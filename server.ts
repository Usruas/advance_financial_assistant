import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import express, { NextFunction, Request, Response } from "express";
import multer from "multer";
import { chunkText, extractPdfText, parseCompanyYear } from "./src/pdf.js";
import {
  answerQuestion,
  embedTexts,
  extractFinancialMetrics,
  resolveChatScope,
  retrieveChunks,
} from "./src/rag.js";
import {
  createJob,
  getAllMetadata,
  getHistory,
  getJob,
  getMetrics,
  listMessages,
  listReports,
  replaceReportChunks,
  ReportChunk,
  saveMetrics,
  saveTurn,
  updateJob,
} from "./src/store.js";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const devFrontend = path.join(__dirname, "frontend");
const FRONTEND = fs.existsSync(devFrontend)
  ? devFrontend
  : path.resolve(__dirname, "..", "frontend");
const PORT = Number(process.env.PORT || "3000");
const MAX_UPLOAD_MB = Number(process.env.MAX_UPLOAD_MB || "50");
const MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use("/static", express.static(FRONTEND));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_BYTES },
});

// Sequential ingestion lock matching INGEST_LOCK in original Python backend
let ingestQueue: Promise<void> = Promise.resolve();

function enqueueIngestion(task: () => Promise<void>): void {
  ingestQueue = ingestQueue.then(task, task);
}

async function processReportJob(
  jobId: string,
  buffer: Buffer,
  filename: string
): Promise<void> {
  try {
    updateJob(jobId, {
      status: "processing",
      stage: "Starting report processing",
    });

    const { company, year } = parseCompanyYear(filename);
    updateJob(jobId, { stage: "Converting PDF to text" });

    const text = await extractPdfText(buffer);
    if (!text) {
      throw new Error(`No readable text was extracted from ${filename}`);
    }

    updateJob(jobId, { stage: "Creating semantic chunks and embeddings" });
    const textChunks = chunkText(text);
    if (textChunks.length === 0) {
      throw new Error(`No readable text was extracted from ${filename}`);
    }

    const embeddings = await embedTexts(textChunks);

    updateJob(jobId, {
      stage: `Saving ${textChunks.length} chunks to ChromaDB`,
    });

    const docs: ReportChunk[] = textChunks.map((content, index) => ({
      id: `${filename}:${index}`,
      pageContent: content,
      embedding: embeddings[index],
      metadata: {
        company,
        year,
        source_file: filename,
        chunk_index: index,
      },
    }));

    replaceReportChunks(filename, docs);

    if (/^\d+$/.test(year)) {
      updateJob(jobId, { stage: "Extracting KPIs" });
      const parsedYear = Number.parseInt(year, 10);
      const kpis = await extractFinancialMetrics(company, parsedYear);
      saveMetrics(company, parsedYear, kpis);
    }

    updateJob(jobId, { status: "completed", stage: "Report ready" });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    updateJob(jobId, {
      status: "failed",
      stage: "Processing failed",
      error: message.slice(0, 1000),
    });
  }
}

app.get("/", (_req: Request, res: Response) => {
  res.sendFile(path.join(FRONTEND, "index.html"));
});

app.get("/api/health", (_req: Request, res: Response) => {
  res.json({
    status: "ok",
    ollama_url: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
    chat_model: process.env.OLLAMA_CHAT_MODEL || "llama3.1",
    embedding_model: process.env.OLLAMA_EMBED_MODEL || "nomic-embed-text",
  });
});

app.get("/api/metrics", (_req: Request, res: Response) => {
  res.json(getMetrics());
});

app.get("/api/reports", (_req: Request, res: Response) => {
  res.json(listReports());
});

app.post(
  "/api/reports",
  upload.single("file"),
  (req: Request, res: Response) => {
    const file = req.file;
    if (!file || path.extname(file.originalname).toLowerCase() !== ".pdf") {
      return res
        .status(400)
        .json({ detail: "Upload an annual report as a PDF file." });
    }
    const safeName = path.basename(file.originalname);
    const jobId = crypto.randomUUID();
    createJob(jobId, safeName);
    const bufferCopy = Buffer.from(file.buffer);
    enqueueIngestion(() => processReportJob(jobId, bufferCopy, safeName));
    return res.json({
      message: "Upload accepted; processing started.",
      job_id: jobId,
      filename: safeName,
    });
  }
);

app.get("/api/jobs/:job_id", (req: Request, res: Response) => {
  const job = getJob(String(req.params.job_id));
  if (!job) {
    return res.status(404).json({ detail: "Processing job not found." });
  }
  return res.json(job);
});

app.post("/api/chat", async (req: Request, res: Response) => {
  try {
    const body = (req.body ?? {}) as {
      question?: unknown;
      company?: unknown;
      year?: unknown;
      conversation_id?: unknown;
    };

    const question =
      typeof body.question === "string" ? body.question.trim() : "";
    if (question.length < 2 || question.length > 2000) {
      return res
        .status(422)
        .json({ detail: "Question must be between 2 and 2000 characters." });
    }

    const selectedCompany =
      typeof body.company === "string" && body.company.trim()
        ? body.company.trim()
        : null;
    const requestedYear =
      typeof body.year === "number" && Number.isFinite(body.year)
        ? body.year
        : typeof body.year === "string" && /^\d{4}$/.test(body.year.trim())
        ? Number.parseInt(body.year.trim(), 10)
        : null;
    const conversationId =
      typeof body.conversation_id === "string" &&
      body.conversation_id.length >= 8 &&
      body.conversation_id.length <= 80
        ? body.conversation_id
        : crypto.randomUUID();

    const history = getHistory(conversationId);
    const reportMetadata = getAllMetadata();

    let [companyFilter, companyScope] = resolveChatScope(
      question,
      selectedCompany,
      reportMetadata
    );

    if (companyScope === null) {
      for (let i = history.length - 1; i >= 0; i--) {
        const previous = history[i];
        if (previous.role === "user") {
          const [prevFilter, prevScope] = resolveChatScope(
            previous.content,
            null,
            reportMetadata
          );
          if (prevFilter === "__missing__" || prevFilter) {
            companyFilter = prevFilter;
            companyScope = prevScope;
            break;
          }
        }
      }
    }

    if (companyFilter === "__missing__") {
      const available = Array.from(
        new Set(reportMetadata.map((m) => m.company).filter(Boolean))
      ).sort();
      const answer = `I don't have an indexed report for ${companyScope}. Available companies are: ${
        available.length ? available.join(", ") : "none yet"
      }.`;
      saveTurn(conversationId, question, answer);
      return res.json({
        answer,
        conversation_id: conversationId,
        sources: [],
      });
    }

    let yearFilter = requestedYear;
    if (yearFilter === null) {
      let yearMatch = question.match(/\b(19\d{2}|20\d{2})\b/);
      if (!yearMatch) {
        for (let i = history.length - 1; i >= 0; i--) {
          if (history[i].role === "user") {
            const m = history[i].content.match(/\b(19\d{2}|20\d{2})\b/);
            if (m) {
              yearMatch = m;
              break;
            }
          }
        }
      }
      if (yearMatch) {
        yearFilter = Number.parseInt(yearMatch[1], 10);
      }
    }

    const priorQuestions = history
      .filter((t) => t.role === "user")
      .slice(-2)
      .map((t) => t.content);
    const retrievalQuery = [...priorQuestions, question].join("\n");

    const docs = await retrieveChunks(retrievalQuery, {
      company: companyFilter,
      year: yearFilter,
      topK: 6,
    });

    if (docs.length === 0) {
      const scopeText = companyScope ? ` for ${companyScope}` : "";
      const answer = `I couldn't find a matching report${scopeText} in the knowledge base. Upload the report or adjust the filters.`;
      saveTurn(conversationId, question, answer);
      return res.json({
        answer,
        conversation_id: conversationId,
        sources: [],
      });
    }

    const answer = await answerQuestion({
      question,
      docs,
      history,
      companyScope,
    });

    saveTurn(conversationId, question, answer);
    return res.json({
      answer,
      conversation_id: conversationId,
      scope: { company: companyScope, year: yearFilter },
      sources: docs.map((d) => ({
        file: d.metadata.source_file || "report",
        company: d.metadata.company,
        year: d.metadata.year,
        chunk: d.metadata.chunk_index,
      })),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return res
      .status(503)
      .json({ detail: `Chat service unavailable: ${message}` });
  }
});

app.get("/api/chat/:conversation_id", (req: Request, res: Response) => {
  const conversationId = String(req.params.conversation_id);
  if (conversationId.length > 80) {
    return res.status(400).json({ detail: "Invalid conversation ID." });
  }
  return res.json({
    conversation_id: conversationId,
    messages: listMessages(conversationId),
  });
});

app.post("/api/metrics/extract", async (req: Request, res: Response) => {
  try {
    const company =
      typeof req.query.company === "string" ? req.query.company.trim() : "";
    const yearRaw =
      typeof req.query.year === "string" ? req.query.year.trim() : "";
    const year = /^\d+$/.test(yearRaw) ? Number.parseInt(yearRaw, 10) : null;
    if (!company || year === null) {
      return res
        .status(400)
        .json({ detail: "Both company and numeric year are required." });
    }
    const result = await extractFinancialMetrics(company, year);
    saveMetrics(company, year, result);
    return res.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return res
      .status(503)
      .json({ detail: `KPI extraction failed: ${message}` });
  }
});

// Handle multer file-size errors and unexpected errors cleanly
app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
    return res
      .status(413)
      .json({ detail: "PDF exceeds the configured upload limit." });
  }
  const message = err instanceof Error ? err.message : String(err);
  return res.status(500).json({ detail: message || "Internal server error" });
});

if (!process.env.VERCEL) {
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Ledgerline server listening on http://0.0.0.0:${PORT}`);
  });
}

export default app;
