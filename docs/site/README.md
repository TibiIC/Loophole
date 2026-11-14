# Project site (`docs/site/`)

A static site, no build step. It renders the repo's markdown (`README.md`, `REFERENCES.md`,
`docs/*.md`) in the browser at view time, so it never copies or edits those files: whatever is
on the branch is what the site shows.

| file | role |
|---|---|
| `index.html` | landing page: 2-up video slot, the two tasks, results table, scoring summary, document index |
| `doc.html?d=<KEY>` | markdown viewer (e.g. `doc.html?d=RESULTS`), with table of contents and cross-doc links |
| `results.json` | **the only source of the landing-page results table and the video** |
| `site.js`, `site.css` | renderer and styles; the document list is `DOCS` at the top of `site.js` |
| `vendor/` | marked 12.0.2 (MIT) and DOMPurify 3.1.6 (Apache-2.0 / MPL-2.0), vendored so it works offline |

## View locally

The pages use `fetch()`, which browsers block on `file://`. From the repo root:

```bash
python3 -m http.server 8000
# open http://localhost:8000/docs/site/
```

A document that is not on the current branch (e.g. `docs/RATTRAP_DESIGN.md` before the
`rattrap` merge) shows a notice with a GitHub link instead of failing.

If the site is hosted with `docs/` as the web root (GitHub Pages "/docs"), `README.md` and
`REFERENCES.md` at the repo root are not served; their pages fall back to the GitHub link.
Serving the repo root avoids that.

## Filling in `results.json`

Nothing in the HTML hardcodes a result. Until `status` is `"final"`, the table carries a
"results pending" badge; with no rows it shows an empty-state row.

```json
{
  "status": "final",
  "updated": "YYYY-MM-DD",
  "caption": "optional sentence under the table",
  "columns": [
    { "key": "task", "label": "Task" },
    { "key": "score", "label": "Score (mean ± sd)", "align": "right" }
  ],
  "rows": [
    { "task": "...", "score": "...", "highlight": true }
  ],
  "footnotes": ["optional", "notes"],
  "video": { "src": "../media/<file>.mp4", "poster": "../media/<file>.png", "caption": "..." }
}
```

* `columns`: order and headers of the table; `align: "right"` for numeric columns; optional
  `digits` formats numeric cell values with that many decimals (strings are shown as-is).
* `rows`: one object per row, keyed by column `key`; a missing key shows "—";
  `highlight: true` tints the row.
* `video.src`: path relative to `docs/site/`. While it is `null` the landing page shows the
  marked placeholder.
