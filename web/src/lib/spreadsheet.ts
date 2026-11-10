/**
 * Spreadsheet preview data for the workspace side panel: delimited text
 * (CSV / TSV / PSV) and workbooks (XLSX / XLS / ODS …). SheetJS is loaded on
 * first use so the main bundle does not carry it.
 */

/** Delimiter per suffix; `null` lets SheetJS detect `,` vs `;` in a .csv. */
const DELIMITED: Record<string, string | null> = { csv: null, tsv: "\t", tab: "\t", psv: "|" };
export const DELIMITED_EXTS = new Set(Object.keys(DELIMITED));
export const WORKBOOK_EXTS = new Set(["xlsx", "xlsm", "xlsb", "xls", "ods", "fods"]);

/** Rendered rows/columns per sheet; the rest is counted, not drawn. */
export const MAX_ROWS = 1000;
export const MAX_COLS = 200;
/** Workbooks above this are parsed in the main thread at the user's expense; refuse them. */
export const MAX_WORKBOOK_BYTES = 20 * 1024 * 1024;

export type SheetData = {
  name: string;
  rows: string[][];
  totalRows: number;
  totalCols: number;
};

type XLSXLib = typeof import("xlsx");
type WorkSheet = import("xlsx").WorkSheet;

const loadLib = (): Promise<XLSXLib> => import("xlsx");

function extOf(name: string): string {
  const lower = name.toLowerCase();
  const dot = lower.lastIndexOf(".");
  return dot > 0 ? lower.slice(dot + 1) : "";
}

function sheetData(X: XLSXLib, name: string, ws: WorkSheet): SheetData {
  const ref = ws["!fullref"] || ws["!ref"];
  const all = X.utils.sheet_to_json<unknown[]>(ws, {
    header: 1, raw: false, defval: "", blankrows: true, dateNF: "yyyy-mm-dd",
  });
  const range = ref ? X.utils.decode_range(ref) : null;
  const totalRows = range ? range.e.r + 1 : all.length;
  const totalCols = range ? range.e.c + 1 : Math.max(0, ...all.map((r) => r.length));
  const cols = Math.min(totalCols, MAX_COLS);
  const rows = all.slice(0, MAX_ROWS).map((r) => {
    const out = new Array<string>(cols).fill("");
    for (let i = 0; i < cols && i < r.length; i++) out[i] = String(r[i] ?? "");
    return out;
  });
  return { name, rows, totalRows, totalCols };
}

/**
 * Parse CSV/TSV text. `truncated` means the gateway cut the file at its preview
 * limit: the partial last line is dropped so it does not show as a broken row.
 */
export async function parseDelimited(text: string, fileName: string, truncated: boolean): Promise<SheetData> {
  const X = await loadLib();
  let body = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  if (truncated) {
    const nl = body.lastIndexOf("\n");
    if (nl > 0) body = body.slice(0, nl);
  }
  const fs = DELIMITED[extOf(fileName)];
  const wb = X.read(body, { type: "string", raw: true, ...(fs ? { FS: fs } : {}) });
  const first = wb.SheetNames[0];
  return sheetData(X, fileName, wb.Sheets[first]!);
}

/** Parse a workbook's bytes; every sheet, first MAX_ROWS rows each. */
export async function parseWorkbook(bytes: ArrayBuffer): Promise<SheetData[]> {
  const X = await loadLib();
  const wb = X.read(new Uint8Array(bytes), { type: "array", sheetRows: MAX_ROWS, dateNF: "yyyy-mm-dd" });
  return wb.SheetNames.map((n) => sheetData(X, n, wb.Sheets[n]!));
}

/** Excel-style column name: 0 -> A, 25 -> Z, 26 -> AA. */
export function columnName(i: number): string {
  let s = "";
  for (let n = i + 1; n > 0; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(65 + ((n - 1) % 26)) + s;
  return s;
}
