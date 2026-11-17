import path from "node:path";
import { extractText } from "unpdf";

export function parseCompanyYear(filename: string): {
  company: string;
  year: string;
} {
  const stem = path.basename(filename, path.extname(filename));
  const leadingYear = stem.match(/^(\d{4})[_ -]+(.+)$/);
  if (leadingYear) {
    return {
      year: leadingYear[1],
      company: leadingYear[2].replace(/[_-]+/g, " ").trim(),
    };
  }
  const trailingYear = stem.match(/^(.+?)[_ -]+(\d{4})$/);
  if (trailingYear) {
    return {
      company: trailingYear[1].replace(/[_-]+/g, " ").trim(),
      year: trailingYear[2],
    };
  }
  const parts = stem.split("_");
  if (parts.length >= 2 && /^\d+$/.test(parts[0])) {
    return { year: parts[0], company: parts.slice(1).join(" ") };
  }
  if (parts.length >= 2) {
    return { company: parts[0], year: parts[1] };
  }
  return { company: stem, year: "" };
}

function extractFallbackPdfText(buffer: Buffer): string {
  const raw = buffer.toString("latin1");
  const matches: string[] = [];
  const tjRegex = /\(([^()\\]*(?:\\.[^()\\]*)*)\)\s*T[jJ]/g;
  let match: RegExpExecArray | null;
  while ((match = tjRegex.exec(raw)) !== null) {
    const cleaned = match[1]
      .replace(/\\n/g, "\n")
      .replace(/\\r/g, " ")
      .replace(/\\t/g, " ")
      .replace(/\\([()\\])/g, "$1")
      .replace(/[^\x20-\x7E\n]/g, " ")
      .trim();
    if (cleaned) matches.push(cleaned);
  }
  return matches.join(" ").replace(/\s+/g, " ").trim();
}

export async function extractPdfText(buffer: Buffer): Promise<string> {
  try {
    const result = await extractText(new Uint8Array(buffer), {
      mergePages: true,
    });
    const text = (result.text ?? "").trim();
    if (text.length > 0) {
      return text;
    }
  } catch (err) {
    console.warn("unpdf extraction failed, trying fallback:", err);
  }
  return extractFallbackPdfText(buffer);
}

export function chunkText(
  text: string,
  targetSize = 1100,
  overlap = 150
): string[] {
  const normalized = text.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  if (!normalized) return [];

  const paragraphs = normalized
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  let current = "";

  const pushCurrent = () => {
    const trimmed = current.trim();
    if (trimmed) chunks.push(trimmed);
  };

  for (const para of paragraphs) {
    if (para.length > targetSize) {
      if (current) {
        pushCurrent();
        current = "";
      }
      const sentences = para.split(/(?<=[.!?])\s+/);
      for (const sentence of sentences) {
        if ((current + " " + sentence).trim().length > targetSize && current) {
          pushCurrent();
          const tail = current.slice(-overlap).trim();
          current = tail ? `${tail} ${sentence}` : sentence;
        } else {
          current = current ? `${current} ${sentence}` : sentence;
        }
      }
      if (current.length > targetSize) {
        for (let i = 0; i < current.length; i += targetSize - overlap) {
          const slice = current.slice(i, i + targetSize).trim();
          if (slice) chunks.push(slice);
        }
        current = "";
      }
      continue;
    }

    if ((current + "\n\n" + para).trim().length > targetSize && current) {
      pushCurrent();
      const tail = current.slice(-overlap).trim();
      current = tail ? `${tail}\n\n${para}` : para;
    } else {
      current = current ? `${current}\n\n${para}` : para;
    }
  }

  pushCurrent();
  return chunks;
}
