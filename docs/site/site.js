// Loophole docs site. Static, no build step. Markdown is fetched from the repo at view
// time and never copied, so the site always shows the current docs/*.md.

const REPO_URL = "https://github.com/TibiIC/Loophole";

// Paths are relative to docs/site/. `repoPath` is used for "view on GitHub" links.
const DOCS = [
  { key: "README", title: "Overview", file: "../../README.md", repoPath: "README.md",
    blurb: "The environment, the Alpha 1.0 task, observation and action spaces, goal and rewards." },
  { key: "RESULTS", title: "Results", file: "../RESULTS.md", repoPath: "docs/RESULTS.md",
    blurb: "Every evaluation run: scores, strategy classes, behaviour axes and plots." },
  { key: "ANOMALY", title: "Anomaly", file: "../ANOMALY.md", repoPath: "docs/ANOMALY.md",
    blurb: "Why some agent runs out-score the oracle, diagnosed from the logs." },
  { key: "EXPLOITS", title: "Reward red-team", file: "../EXPLOITS.md", repoPath: "docs/EXPLOITS.md",
    blurb: "Exploit attempts against the reward and the harness, and what stopped each one." },
  { key: "HARNESS_NOTES", title: "Harness notes", file: "../HARNESS_NOTES.md", repoPath: "docs/HARNESS_NOTES.md",
    blurb: "Task scope, geometry provenance and licence, and harness behaviour." },
  { key: "RATTRAP_DESIGN", title: "Rat Trap design", file: "../RATTRAP_DESIGN.md", repoPath: "docs/RATTRAP_DESIGN.md",
    blurb: "The cord-and-frame puzzle: reference solve, score and checkpoint progress." },
  { key: "VARIANTS", title: "Harder variants", file: "../VARIANTS.md", repoPath: "docs/VARIANTS.md",
    blurb: "Candidate harder puzzles from the published archive (prepared, not switched)." },
  { key: "TRAINING", title: "Training", file: "../TRAINING.md", repoPath: "docs/TRAINING.md",
    blurb: "Training protocol: sampled starts, shaping, and how evaluation stays held out." },
  { key: "REFERENCES", title: "References", file: "../../REFERENCES.md", repoPath: "REFERENCES.md",
    blurb: "Every citation, checked against its DOI, arXiv or publisher record." },
];

const byKey = Object.fromEntries(DOCS.map((d) => [d.key, d]));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

function servedOverFile() {
  return location.protocol === "file:";
}

function fileNotice() {
  return el("div", { class: "notice" },
    "This page loads files with fetch(), which browsers block on file:// URLs. Serve the repo over HTTP, e.g. ",
    el("code", { text: "python3 -m http.server" }),
    " from the repo root, then open ",
    el("code", { text: "http://localhost:8000/docs/site/" }), ".");
}

// ---------- shared nav ----------

function renderNav(currentKey) {
  const nav = document.getElementById("nav");
  if (!nav) return;
  nav.append(el("a", { href: "index.html", "aria-current": currentKey ? null : "page", text: "Home" }));
  for (const d of DOCS) {
    nav.append(el("a", {
      href: `doc.html?d=${d.key}`,
      "aria-current": d.key === currentKey ? "page" : null,
      text: d.title,
    }));
  }
}

// ---------- landing page ----------

function renderDocCards(container) {
  for (const d of DOCS) {
    container.append(el("a", { class: "card", href: `doc.html?d=${d.key}` },
      el("div", { class: "file", text: d.repoPath }),
      el("h3", { text: d.title }),
      el("p", { text: d.blurb })));
  }
}

function formatCell(value, col) {
  if (value == null || value === "") return "—";
  if (typeof value === "number" && Number.isInteger(col.digits)) return value.toFixed(col.digits);
  return String(value);
}

function renderVideo(slot, video) {
  if (!video || !video.src) return; // keep the placeholder markup from the HTML
  const fig = el("figure", { class: "video-real" });
  fig.style.margin = "0";
  const v = el("video", { src: video.src, poster: video.poster || null, controls: "", playsinline: "", preload: "metadata" });
  fig.append(v);
  if (video.caption) fig.append(el("figcaption", { text: video.caption }));
  slot.replaceWith(fig);
}

function renderResults(root, data) {
  const cols = Array.isArray(data.columns) ? data.columns : [];
  const rows = Array.isArray(data.rows) ? data.rows : [];
  const final = data.status === "final";

  const head = root.querySelector(".results-head");
  if (!final) head.append(el("span", { class: "placeholder-tag placeholder", text: "Placeholder: results pending" }));
  if (data.updated) head.append(el("span", { class: "results-meta", text: `Updated ${data.updated}` }));

  const table = el("table");
  const thead = el("thead");
  const htr = el("tr");
  for (const c of cols) htr.append(el("th", { class: c.align === "right" ? "num" : null, text: c.label ?? c.key }));
  thead.append(htr);
  table.append(thead);

  const tbody = el("tbody");
  if (rows.length === 0) {
    tbody.append(el("tr", {}, el("td", {
      class: "empty",
      colspan: String(Math.max(cols.length, 1)),
      text: "No results yet. Final numbers will appear here once docs/site/results.json is filled in.",
    })));
  } else {
    for (const r of rows) {
      const tr = el("tr", { class: r.highlight ? "highlight" : null });
      for (const c of cols) tr.append(el("td", { class: c.align === "right" ? "num" : null, text: formatCell(r[c.key], c) }));
      tbody.append(tr);
    }
  }
  table.append(tbody);
  root.querySelector(".table-scroll").append(table);

  if (data.caption) root.append(el("p", { class: "results-meta", text: data.caption }));
  if (Array.isArray(data.footnotes) && data.footnotes.length) {
    const ul = el("ul", { class: "footnotes" });
    for (const f of data.footnotes) ul.append(el("li", { text: f }));
    root.append(ul);
  }
}

async function initHome() {
  renderNav(null);
  renderDocCards(document.getElementById("doc-cards"));
  const resultsRoot = document.getElementById("results");
  if (servedOverFile()) {
    resultsRoot.append(fileNotice());
    return;
  }
  try {
    const res = await fetch("results.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderResults(resultsRoot, data);
    renderVideo(document.getElementById("video-slot"), data.video);
  } catch (err) {
    resultsRoot.append(el("div", { class: "notice", text: `Could not load results.json (${err.message}).` }));
  }
}

// ---------- document viewer ----------

// GitHub-style heading ids, so in-doc links like #goal-free-escape-by-translation work.
function slugger() {
  const counts = new Map();
  return (text) => {
    const base = text.trim().toLowerCase()
      .replace(/[^\p{L}\p{N}\s_-]/gu, "")
      .replace(/\s/g, "-");
    let n = counts.get(base);
    if (n === undefined) { counts.set(base, 1); return base; }
    let slug;
    do slug = `${base}-${n++}`; while (counts.has(slug));
    counts.set(base, n);
    counts.set(slug, 1);
    return slug;
  };
}

function docForUrl(url) {
  const path = url.pathname;
  return DOCS.find((d) => new URL(d.file, location.href).pathname === path) || null;
}

// Plain-text mentions such as "docs/ANOMALY.md" or `RESULTS.md` become viewer links.
const DOC_MENTION = /\b(?:docs\/)?([A-Z][A-Z_]+)\.md\b/g;

function linkifyMentions(root) {
  for (const code of root.querySelectorAll("code")) {
    if (code.closest("pre, a")) continue;
    const m = code.textContent.trim().match(/^(?:docs\/)?([A-Z][A-Z_]+)\.md$/);
    if (m && byKey[m[1]]) code.replaceWith(el("a", { href: `doc.html?d=${m[1]}` }, code.cloneNode(true)));
  }
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (n) => (n.parentElement.closest("a, code, pre") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  });
  const targets = [];
  const probe = new RegExp(DOC_MENTION.source);
  while (walker.nextNode()) if (probe.test(walker.currentNode.nodeValue)) targets.push(walker.currentNode);
  for (const node of targets) {
    const frag = document.createDocumentFragment();
    let last = 0;
    const text = node.nodeValue;
    for (const m of text.matchAll(DOC_MENTION)) {
      if (!byKey[m[1]]) continue;
      frag.append(text.slice(last, m.index));
      frag.append(el("a", { href: `doc.html?d=${m[1]}`, text: m[0] }));
      last = m.index + m[0].length;
    }
    frag.append(text.slice(last));
    node.replaceWith(frag);
  }
}

function fixLinksAndMedia(root, mdUrl) {
  for (const a of root.querySelectorAll("a[href]")) {
    const href = a.getAttribute("href");
    if (href.startsWith("#")) continue;
    let url;
    try { url = new URL(href, mdUrl); } catch { continue; }
    if (url.origin !== location.origin) { a.target = "_blank"; a.rel = "noopener"; continue; }
    const target = docForUrl(url);
    if (target) a.href = `doc.html?d=${target.key}${url.hash}`;
    else a.href = url.href;
  }
  for (const img of root.querySelectorAll("img[src]")) {
    img.src = new URL(img.getAttribute("src"), mdUrl).href;
    img.loading = "lazy";
  }
  for (const t of root.querySelectorAll("table")) {
    const wrap = el("div", { class: "table-scroll" });
    t.replaceWith(wrap);
    wrap.append(t);
  }
}

function buildToc(root, tocList) {
  const slug = slugger();
  const items = [];
  for (const h of root.querySelectorAll("h1, h2, h3, h4")) {
    if (!h.id) h.id = slug(h.textContent);
    if (h.tagName === "H2" || h.tagName === "H3") {
      const li = el("li", { class: h.tagName === "H3" ? "lvl3" : null },
        el("a", { href: `#${h.id}`, text: h.textContent }));
      tocList.append(li);
      items.push([h, li.firstChild]);
    }
  }
  if (!items.length) return;
  const obs = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      for (const [, a] of items) a.classList.remove("active");
      const hit = items.find(([h]) => h === e.target);
      if (hit) hit[1].classList.add("active");
    }
  }, { rootMargin: "-60px 0px -70% 0px" });
  for (const [h] of items) obs.observe(h);
}

function renderOtherDocs(list, currentKey) {
  for (const d of DOCS) {
    list.append(el("li", {}, el("a", {
      href: `doc.html?d=${d.key}`,
      class: d.key === currentKey ? "active" : null,
      text: d.title,
    })));
  }
}

async function initDoc() {
  const key = new URLSearchParams(location.search).get("d") || "README";
  const doc = byKey[key];
  renderNav(doc ? doc.key : null);
  const body = document.getElementById("doc-body");
  const source = document.getElementById("doc-source");
  renderOtherDocs(document.getElementById("doc-list"), key);

  const toggle = document.querySelector(".toc-toggle");
  toggle.addEventListener("click", () => document.querySelector(".toc").classList.toggle("open"));

  if (!doc) {
    body.append(el("div", { class: "notice", text: `Unknown document "${key}".` }));
    return;
  }
  document.title = `${doc.title} · Loophole`;
  source.append(doc.repoPath, " · ", el("a", { href: `${REPO_URL}/blob/HEAD/${doc.repoPath}`, target: "_blank", rel: "noopener", text: "view on GitHub" }));

  if (servedOverFile()) { body.append(fileNotice()); return; }

  const mdUrl = new URL(doc.file, location.href);
  let text;
  try {
    const res = await fetch(mdUrl, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    text = await res.text();
  } catch (err) {
    body.append(el("div", { class: "notice" },
      `${doc.repoPath} could not be loaded here (${err.message}). It may not be on this branch yet, or this host may not serve that path. `,
      el("a", { href: `${REPO_URL}/blob/HEAD/${doc.repoPath}`, target: "_blank", rel: "noopener", text: "Open it on GitHub" }), "."));
    return;
  }

  const html = marked.parse(text, { gfm: true });
  body.innerHTML = DOMPurify.sanitize(html, { ADD_TAGS: ["video", "source"], ADD_ATTR: ["controls", "poster", "target"] });
  fixLinksAndMedia(body, mdUrl);
  linkifyMentions(body);
  buildToc(body, document.getElementById("toc-list"));

  const h1 = body.querySelector("h1");
  if (h1) document.title = `${h1.textContent} · Loophole`;
  if (location.hash) {
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) target.scrollIntoView();
  }
}
