export interface ReportChunk {
  id: string;
  pageContent: string;
  embedding?: number[];
  metadata: {
    company: string;
    year: string;
    source_file: string;
    chunk_index: number;
  };
}

export interface ReportSummary {
  source_file: string;
  company: string;
  year: string;
}

export interface MetricRow {
  id: number;
  company: string;
  year: string | null;
  revenue: string | null;
  net_income: string | null;
  operating_income: string | null;
  cash_flow: string | null;
  total_assets: string | null;
  total_liabilities: string | null;
  risk_factors: string | null;
  growth_drivers: string | null;
  created_at: string;
}

export interface ExtractedKPIs {
  Revenue: string | number | null;
  "Net Income": string | number | null;
  "Operating Income": string | number | null;
  "Cash Flow from Operating Activities": string | number | null;
  "Total Assets": string | number | null;
  "Total Liabilities": string | number | null;
  "Top Risk Factors": string | string[] | null;
  "Top Growth Drivers": string | string[] | null;
}

export interface IngestionJob {
  job_id: string;
  filename: string;
  status: "queued" | "processing" | "completed" | "failed";
  stage: string;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

// MOCKED in-memory stores replacing SQLite and ChromaDB
const chunksStore: ReportChunk[] = [];
const metricsStore: MetricRow[] = [];
const jobsStore = new Map<string, IngestionJob>();
const conversationsStore = new Map<string, ChatMessage[]>();
let nextMetricId = 1;

export function getAllChunks(): ReportChunk[] {
  return chunksStore;
}

export function getAllMetadata(): ReportChunk["metadata"][] {
  return chunksStore.map((c) => c.metadata);
}

export function listReports(): ReportSummary[] {
  const byFile = new Map<string, ReportSummary>();
  for (const chunk of chunksStore) {
    const name = chunk.metadata.source_file;
    if (name) {
      byFile.set(name, {
        source_file: name,
        company: chunk.metadata.company,
        year: chunk.metadata.year,
      });
    }
  }
  return Array.from(byFile.values());
}

export function replaceReportChunks(
  sourceFile: string,
  newChunks: ReportChunk[]
): void {
  for (let i = chunksStore.length - 1; i >= 0; i--) {
    if (chunksStore[i].metadata.source_file === sourceFile) {
      chunksStore.splice(i, 1);
    }
  }
  chunksStore.push(...newChunks);
}

function formatValue(val: unknown): string | null {
  if (val === null || val === undefined) return null;
  if (Array.isArray(val)) {
    const joined = val.map((item) => String(item)).join("\n").trim();
    return joined || null;
  }
  const str = String(val).trim();
  return str || null;
}

export function saveMetrics(
  company: string,
  year: number | null,
  metrics: ExtractedKPIs | Record<string, unknown>
): void {
  const record = metrics as Record<string, unknown>;
  const row: MetricRow = {
    id: nextMetricId++,
    company,
    year: year !== null && !Number.isNaN(year) ? String(year) : null,
    revenue: formatValue(record["Revenue"] ?? record["revenue"]),
    net_income: formatValue(record["Net Income"] ?? record["net_income"]),
    operating_income: formatValue(
      record["Operating Income"] ?? record["operating_income"]
    ),
    cash_flow: formatValue(
      record["Cash Flow from Operating Activities"] ?? record["cash_flow"]
    ),
    total_assets: formatValue(
      record["Total Assets"] ?? record["total_assets"]
    ),
    total_liabilities: formatValue(
      record["Total Liabilities"] ?? record["total_liabilities"]
    ),
    risk_factors: formatValue(
      record["Top Risk Factors"] ?? record["risk_factors"]
    ),
    growth_drivers: formatValue(
      record["Top Growth Drivers"] ?? record["growth_drivers"]
    ),
    created_at: new Date().toISOString(),
  };
  metricsStore.push(row);
}

export function getMetrics(): MetricRow[] {
  return [...metricsStore].sort((a, b) => {
    const cmp = a.company.localeCompare(b.company);
    if (cmp !== 0) return cmp;
    const yearCmp = (b.year ?? "").localeCompare(a.year ?? "");
    if (yearCmp !== 0) return yearCmp;
    return b.created_at.localeCompare(a.created_at);
  });
}

export function createJob(jobId: string, filename: string): void {
  const now = new Date().toISOString();
  jobsStore.set(jobId, {
    job_id: jobId,
    filename,
    status: "queued",
    stage: "Waiting for worker",
    error: null,
    created_at: now,
    updated_at: now,
  });
}

export function updateJob(
  jobId: string,
  patch: Partial<Pick<IngestionJob, "status" | "stage" | "error">>
): void {
  const existing = jobsStore.get(jobId);
  if (!existing) return;
  if (patch.status !== undefined) existing.status = patch.status;
  if (patch.stage !== undefined) existing.stage = patch.stage;
  if (patch.error !== undefined) existing.error = patch.error;
  existing.updated_at = new Date().toISOString();
}

export function getJob(jobId: string): IngestionJob | null {
  return jobsStore.get(jobId) ?? null;
}

export function getHistory(
  conversationId: string,
  limit = 12
): ChatMessage[] {
  const messages = conversationsStore.get(conversationId) ?? [];
  return messages.slice(-limit);
}

export function listMessages(
  conversationId: string,
  limit = 24
): ChatMessage[] {
  const messages = conversationsStore.get(conversationId) ?? [];
  return messages.slice(-limit);
}

export function saveTurn(
  conversationId: string,
  question: string,
  answer: string
): void {
  const messages = conversationsStore.get(conversationId) ?? [];
  messages.push(
    { role: "user", content: question },
    { role: "assistant", content: answer }
  );
  if (messages.length > 24) {
    conversationsStore.set(conversationId, messages.slice(-24));
  } else {
    conversationsStore.set(conversationId, messages);
  }
}
