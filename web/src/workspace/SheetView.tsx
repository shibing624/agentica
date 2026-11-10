/**
 * Table view for CSV / TSV text and XLSX / XLS / ODS workbooks in the file
 * preview: sticky column letters and row numbers, one tab per sheet, and a
 * note when rows or columns past the render cap are hidden.
 */
import { useEffect, useState } from "react";
import { useStrings } from "../i18n";
import { fmtFileSize } from "../lib/format";
import {
  MAX_WORKBOOK_BYTES, columnName, parseDelimited, parseWorkbook, type SheetData,
} from "../lib/spreadsheet";

type Loaded = { sheets: SheetData[] } | { error: string };

function SheetTable({ sheet }: { sheet: SheetData }) {
  const cols = sheet.rows[0]?.length ?? 0;
  return (
    <div className="sheet-scroll">
      <table className="sheet-table">
        <thead>
          <tr>
            <th className="sheet-corner" />
            {Array.from({ length: cols }, (_, i) => <th key={i}>{columnName(i)}</th>)}
          </tr>
        </thead>
        <tbody>
          {sheet.rows.map((row, r) => (
            <tr key={r}>
              <th className="sheet-rownum">{r + 1}</th>
              {row.map((cell, c) => <td key={c} title={cell.length > 40 ? cell : undefined}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** `text` for delimited files (already fetched as preview text), `src` for workbooks. */
export function SheetView({ name, text, truncated, src }: {
  name: string;
  text?: string;
  truncated?: boolean;
  src?: string;
}) {
  const S = useStrings();
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    let live = true;
    setLoaded(null);
    setActive(0);
    const run = async (): Promise<Loaded> => {
      if (text !== undefined) return { sheets: [await parseDelimited(text, name, !!truncated)] };
      const res = await fetch(src!);
      if (!res.ok) return { error: S.files.sheetLoadFailed };
      const len = Number(res.headers.get("content-length") || 0);
      if (len > MAX_WORKBOOK_BYTES) return { error: S.files.sheetTooLarge(fmtFileSize(MAX_WORKBOOK_BYTES)) };
      return { sheets: await parseWorkbook(await res.arrayBuffer()) };
    };
    run()
      .then((v) => { if (live) setLoaded(v); })
      .catch((e) => { if (live) setLoaded({ error: `${S.files.sheetLoadFailed}: ${e?.message || e}` }); });
    return () => { live = false; };
  }, [name, text, truncated, src]);

  if (!loaded) return <p className="ws-muted">{S.common.loading}</p>;
  if ("error" in loaded) return <p className="ws-muted">{loaded.error}</p>;
  const sheet = loaded.sheets[Math.min(active, loaded.sheets.length - 1)];
  if (!sheet) return <p className="ws-muted">{S.files.sheetEmpty}</p>;
  const shownRows = sheet.rows.length;
  const shownCols = sheet.rows[0]?.length ?? 0;
  const clipped = shownRows < sheet.totalRows || shownCols < sheet.totalCols;
  return (
    <div className="sheet-view">
      {loaded.sheets.length > 1 && (
        <div className="sheet-tabs" role="tablist">
          {loaded.sheets.map((s, i) => (
            <button key={s.name + i} type="button" role="tab" aria-selected={i === active}
                    className={"sheet-tab" + (i === active ? " active" : "")} onClick={() => setActive(i)}>
              {s.name}
            </button>
          ))}
        </div>
      )}
      {shownRows ? <SheetTable sheet={sheet} /> : <p className="ws-muted">{S.files.sheetEmpty}</p>}
      <div className="sheet-foot">
        {S.files.sheetSize(sheet.totalRows, sheet.totalCols)}
        {clipped && <span> · {S.files.sheetClipped(shownRows, shownCols)}</span>}
      </div>
    </div>
  );
}
