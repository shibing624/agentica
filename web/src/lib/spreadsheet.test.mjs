import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const vite = await createServer({
  root: fileURLToPath(new URL("../..", import.meta.url)),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true },
});
const SP = await vite.ssrLoadModule("/src/lib/spreadsheet.ts");
const X = await vite.ssrLoadModule("xlsx");
after(() => vite.close());

test("csv keeps quoted commas and newlines, strips the BOM, and pads ragged rows", async () => {
  const s = await SP.parseDelimited('\uFEFFname,desc,n\n"a,b","two\nlines",007\nshort\n', "x.csv", false);
  assert.deepEqual(s.rows, [["name", "desc", "n"], ["a,b", "two\nlines", "007"], ["short", "", ""]]);
  assert.equal(s.totalRows, 3);
  assert.equal(s.totalCols, 3);
});

test("tsv splits on tabs only, so commas stay inside cells", async () => {
  const s = await SP.parseDelimited("a\tb,c\n1\t2,3\n", "x.tsv", false);
  assert.deepEqual(s.rows, [["a", "b,c"], ["1", "2,3"]]);
});

test("a preview truncated by the gateway drops its partial last line", async () => {
  const s = await SP.parseDelimited("a,b\n1,2\n3,4\n5,", "x.csv", true);
  assert.deepEqual(s.rows, [["a", "b"], ["1", "2"], ["3", "4"]]);
});

test("rows past MAX_ROWS are counted but not rendered", async () => {
  const text = Array.from({ length: SP.MAX_ROWS + 5 }, (_, i) => `${i},x`).join("\n");
  const s = await SP.parseDelimited(text, "big.csv", false);
  assert.equal(s.rows.length, SP.MAX_ROWS);
  assert.equal(s.totalRows, SP.MAX_ROWS + 5);
});

test("workbooks keep every sheet, show dates as dates, and report the full size when clipped", async () => {
  const wb = X.utils.book_new();
  X.utils.book_append_sheet(wb, X.utils.aoa_to_sheet([["名字", "日期"], ["张三", new Date(Date.UTC(2026, 0, 2))]]), "表1");
  const tall = Array.from({ length: SP.MAX_ROWS + 3 }, (_, i) => [i]);
  X.utils.book_append_sheet(wb, X.utils.aoa_to_sheet(tall), "tall");
  for (const bookType of ["xlsx", "biff8", "ods"]) {
    const bytes = X.write(wb, { type: "array", bookType });
    const sheets = await SP.parseWorkbook(bytes);
    assert.deepEqual(sheets.map((s) => s.name), ["表1", "tall"], bookType);
    // A date cell with no format of its own shows as ISO; ODS stores the
    // writer's format with the cell, and the preview honors it like a spreadsheet app.
    assert.deepEqual(sheets[0].rows, [["名字", "日期"], ["张三", bookType === "ods" ? "1/2/26" : "2026-01-02"]], bookType);
    assert.equal(sheets[1].rows.length, SP.MAX_ROWS, bookType);
    assert.equal(sheets[1].totalRows, SP.MAX_ROWS + 3, bookType);
  }
});

test("column names follow Excel lettering", () => {
  assert.deepEqual([0, 25, 26, 701, 702].map(SP.columnName), ["A", "Z", "AA", "ZZ", "AAA"]);
});
