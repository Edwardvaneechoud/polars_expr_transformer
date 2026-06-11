/* ============================================================
   Flowfile Expressions — playground & documentation app
   Runs polars-expr-transformer in the browser via Pyodide.
   ============================================================ */

"use strict";

// Pinned to the 0.27.x line: it is the latest Pyodide release that ships a
// polars build (polars was removed from the 0.28/0.29 distributions, see
// https://blog.pyodide.org/posts/0.28-release/).
const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";

/* ----------------------------------------------------------------
   Sample datasets. The dtype tags are interpreted by runtime.py
   (date/datetime values are ISO strings parsed into real temporals).
   ---------------------------------------------------------------- */
const DATASETS = {
  employees: {
    label: "Employees",
    columns: [
      { name: "first_name", dtype: "str", values: ["John", "Jane", "Bob", "Alice", "Mehmet", "Sofia", "Liam", "Yuki"] },
      { name: "last_name", dtype: "str", values: ["Doe", "Smith", "Johnson", "Williams", "Yilmaz", "Rossi", "O'Brien", "Tanaka"] },
      { name: "age", dtype: "int", values: [30, 25, 45, 38, 52, 29, 33, 41] },
      { name: "salary", dtype: "float", values: [50000, 60000, 75000, 82000, 91000, 58500, 67250, 73900] },
      { name: "department", dtype: "str", values: ["Sales", "Engineering", "Engineering", "Marketing", "Sales", "Engineering", "Finance", "Marketing"] },
      { name: "hire_date", dtype: "date", values: ["2021-03-15", "2019-07-01", "2010-11-23", "2015-02-09", "2008-06-30", "2022-01-10", "2018-09-17", "2012-04-05"] },
      { name: "email", dtype: "str", values: ["john.doe@acme.com", null, "bob.j@acme.com", "alice.w@acme.com", null, "sofia.r@acme.com", "liam.ob@acme.com", "yuki.t@acme.com"] },
    ],
  },
  orders: {
    label: "Orders",
    columns: [
      { name: "order_id", dtype: "str", values: ["ORD-0001", "ORD-0002", "ORD-0003", "ORD-0004", "ORD-0005", "ORD-0006", "ORD-0007", "ORD-0008"] },
      { name: "product", dtype: "str", values: ["Laptop Pro", "Wireless Mouse", "USB-C Hub", "Monitor 27in", "Mechanical Keyboard", "Webcam HD", "Laptop Pro", "Desk Lamp"] },
      { name: "category", dtype: "str", values: ["Computers", "Accessories", "Accessories", "Displays", "Accessories", "Accessories", "Computers", "Office"] },
      { name: "price", dtype: "float", values: [1299.99, 24.5, 49.95, 329.0, 119.99, 79.9, 1299.99, 39.95] },
      { name: "quantity", dtype: "int", values: [1, 4, 2, 2, 1, 3, 2, 5] },
      { name: "discount", dtype: "float", values: [0.1, null, 0.05, null, 0.15, null, 0.2, 0.05] },
      { name: "order_date", dtype: "datetime", values: ["2024-01-15 10:30:00", "2024-01-17 14:05:12", "2024-02-02 09:12:45", "2024-02-14 18:44:03", "2024-03-01 11:00:00", "2024-03-08 16:27:39", "2024-03-21 08:55:21", "2024-04-02 13:10:08"] },
      { name: "status", dtype: "str", values: ["shipped", "shipped", "pending", "shipped", "cancelled", "pending", "shipped", "pending"] },
    ],
  },
  events: {
    label: "Events",
    columns: [
      { name: "event", dtype: "str", values: ["Kickoff Meeting", "Design Sprint", "Tech Conference", "Team Offsite", "Product Launch", "Retrospective"] },
      { name: "city", dtype: "str", values: ["Amsterdam", "Berlin", "Lisbon", "Utrecht", "Amsterdam", "Remote"] },
      { name: "start", dtype: "datetime", values: ["2024-05-06 09:00:00", "2024-05-13 09:30:00", "2024-06-18 08:00:00", "2024-07-05 10:00:00", "2024-09-02 17:00:00", "2024-09-27 15:00:00"] },
      { name: "end", dtype: "datetime", values: ["2024-05-06 10:30:00", "2024-05-17 17:00:00", "2024-06-20 18:00:00", "2024-07-06 16:00:00", "2024-09-02 21:30:00", "2024-09-27 16:00:00"] },
      { name: "attendees", dtype: "int", values: [12, 8, 450, 35, null, 9] },
    ],
  },
};

const EXAMPLES = [
  { label: "Full name", dataset: "employees", expr: 'concat([first_name], " ", [last_name])' },
  { label: "Seniority", dataset: "employees", expr: 'if [age] >= 45 then "Senior"\nelseif [age] >= 30 then "Mid"\nelse "Junior" endif' },
  { label: "10% raise", dataset: "employees", expr: "round([salary] * 1.1, 2) // add 10%" },
  { label: "Email fallback", dataset: "employees", expr: 'coalesce([email], concat(lowercase([first_name]), "@acme.com"))' },
  { label: "Years employed", dataset: "employees", expr: "floor(date_diff_days(today(), [hire_date]) / 365)" },
  { label: "Order total", dataset: "orders", expr: "round([price] * [quantity] * (1 - ifnull([discount], 0)), 2)" },
  { label: "Big order?", dataset: "orders", expr: '[price] * [quantity] > 500 and [status] != "cancelled"' },
  { label: "Order month", dataset: "orders", expr: 'format_date([order_date], "%B %Y")' },
  { label: "Duration (h)", dataset: "events", expr: "datetime_diff_seconds([end], [start]) / 3600" },
  { label: "Weekend?", dataset: "events", expr: "weekday([start]) >= 6" },
];

const KEYWORDS = ["if", "then", "elseif", "else", "endif", "and", "or", "not", "true", "false"];

/* ---------------- state ---------------- */
const state = {
  pyodide: null,
  runFn: null,
  ready: false,
  dataset: "employees",
  functionIndex: [], // [{name, signature, description, example_call}]
  reference: null,
  runPending: false,
  runQueued: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ----------------------------------------------------------------
   Generic helpers
   ---------------------------------------------------------------- */
function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function copyText(text, button) {
  navigator.clipboard.writeText(text).then(() => {
    if (!button) return;
    button.classList.add("copied");
    const original = button.textContent;
    button.textContent = "✓";
    setTimeout(() => {
      button.classList.remove("copied");
      button.textContent = original;
    }, 1200);
  });
}

function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

/* ----------------------------------------------------------------
   Expression syntax highlighting
   ---------------------------------------------------------------- */
const TOKEN_RE = new RegExp(
  [
    "(\\/\\/[^\\n]*)",                                  // 1 comment
    "(\"(?:[^\"\\\\\\n]|\\\\.)*\"?|'(?:[^'\\\\\\n]|\\\\.)*'?)", // 2 string
    "(\\[[^\\]\\n]*\\]?)",                              // 3 column ref
    "(\\b\\d+(?:\\.\\d+)?\\b)",                         // 4 number
    `(\\b(?:${KEYWORDS.join("|")})\\b)`,                // 5 keyword
    "([A-Za-z_][A-Za-z0-9_]*(?=\\s*\\())",              // 6 function call
    "([+\\-*/%]|==|!=|>=|<=|=|>|<)",                    // 7 operator
  ].join("|"),
  "g"
);

const TOKEN_CLASSES = ["tok-com", "tok-str", "tok-col", "tok-num", "tok-kw", "tok-fn", "tok-op"];

function highlightExpression(src) {
  let html = "";
  let last = 0;
  TOKEN_RE.lastIndex = 0;
  let match;
  while ((match = TOKEN_RE.exec(src)) !== null) {
    html += escapeHtml(src.slice(last, match.index));
    for (let group = 1; group <= 7; group++) {
      if (match[group] !== undefined) {
        html += `<span class="${TOKEN_CLASSES[group - 1]}">${escapeHtml(match[group])}</span>`;
        break;
      }
    }
    last = TOKEN_RE.lastIndex;
  }
  html += escapeHtml(src.slice(last));
  return html;
}

function highlightPython(src) {
  const re = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?\b)|(\b(?:pl|ff|datetime)\b)|(\.[A-Za-z_][A-Za-z0-9_]*(?=\())/g;
  let html = "";
  let last = 0;
  let match;
  while ((match = re.exec(src)) !== null) {
    html += escapeHtml(src.slice(last, match.index));
    if (match[1] !== undefined) html += `<span class="tok-str">${escapeHtml(match[1])}</span>`;
    else if (match[2] !== undefined) html += `<span class="tok-num">${escapeHtml(match[2])}</span>`;
    else if (match[3] !== undefined) html += `<span class="tok-kw">${escapeHtml(match[3])}</span>`;
    else html += `<span class="tok-fn">${escapeHtml(match[4])}</span>`;
    last = re.lastIndex;
  }
  html += escapeHtml(src.slice(last));
  return html;
}

/* ----------------------------------------------------------------
   Editor (textarea + highlight overlay + autocomplete)
   ---------------------------------------------------------------- */
const editor = $("#expression-input");
const editorHighlight = $("#editor-highlight");
const autocompleteBox = $("#autocomplete");
let acItems = [];
let acSelected = 0;
let acContext = null; // {start, end, mode}

function syncHighlight() {
  let value = editor.value;
  if (value.endsWith("\n")) value += " ";
  editorHighlight.innerHTML = highlightExpression(value);
  editorHighlight.scrollTop = editor.scrollTop;
  editorHighlight.scrollLeft = editor.scrollLeft;
}

/* Measure caret pixel position with a hidden mirror element. */
function caretPosition() {
  const mirror = document.createElement("div");
  const style = getComputedStyle(editor);
  for (const prop of ["fontFamily", "fontSize", "lineHeight", "padding", "border", "boxSizing", "whiteSpace", "wordWrap", "overflowWrap", "tabSize"]) {
    mirror.style[prop] = style[prop];
  }
  mirror.style.position = "absolute";
  mirror.style.visibility = "hidden";
  mirror.style.width = editor.clientWidth + "px";
  mirror.style.whiteSpace = "pre-wrap";
  mirror.textContent = editor.value.slice(0, editor.selectionStart);
  const marker = document.createElement("span");
  marker.textContent = "​";
  mirror.appendChild(marker);
  editor.parentElement.appendChild(mirror);
  const top = marker.offsetTop - editor.scrollTop;
  const left = marker.offsetLeft - editor.scrollLeft;
  mirror.remove();
  return { top, left };
}

function autocompleteCandidates() {
  const caret = editor.selectionStart;
  const before = editor.value.slice(0, caret);

  // Inside an unclosed [column ref]?
  const bracketStart = before.lastIndexOf("[");
  if (bracketStart !== -1 && !before.slice(bracketStart).includes("]")) {
    const prefix = before.slice(bracketStart + 1).toLowerCase();
    if (!prefix.includes("\n")) {
      const columns = DATASETS[state.dataset].columns
        .filter((c) => c.name.toLowerCase().startsWith(prefix))
        .map((c) => ({ insert: `${c.name}]`, name: c.name, detail: c.dtype, kind: "col" }));
      return { items: columns, start: bracketStart + 1, end: caret };
    }
  }

  // Otherwise complete function names from the current word.
  const wordMatch = before.match(/[A-Za-z_][A-Za-z0-9_]*$/);
  if (!wordMatch) return null;
  const word = wordMatch[0].toLowerCase();
  if (word.length < 2 || KEYWORDS.includes(word)) return null;
  const fns = state.functionIndex
    .filter((f) => f.name.toLowerCase().startsWith(word))
    .slice(0, 8)
    .map((f) => ({ insert: `${f.name}(`, name: f.name, detail: f.signature, kind: "fn" }));
  return { items: fns, start: caret - word.length, end: caret };
}

function showAutocomplete() {
  const result = autocompleteCandidates();
  if (!result || result.items.length === 0) return hideAutocomplete();
  acItems = result.items;
  acContext = result;
  acSelected = 0;
  autocompleteBox.innerHTML = acItems
    .map(
      (item, index) => `
      <div class="ac-item ${index === 0 ? "selected" : ""}" data-index="${index}">
        <span class="ac-name ${item.kind === "col" ? "ac-col" : ""}">${escapeHtml(item.name)}</span>
        <span class="ac-detail">${escapeHtml(item.detail || "")}</span>
      </div>`
    )
    .join("");
  const pos = caretPosition();
  autocompleteBox.style.top = pos.top + 26 + "px";
  autocompleteBox.style.left = Math.min(pos.left, editor.clientWidth - 120) + "px";
  autocompleteBox.classList.remove("hidden");
}

function hideAutocomplete() {
  autocompleteBox.classList.add("hidden");
  acItems = [];
  acContext = null;
}

function acceptAutocomplete(index) {
  if (!acContext || !acItems[index]) return;
  const item = acItems[index];
  const value = editor.value;
  editor.value = value.slice(0, acContext.start) + item.insert + value.slice(acContext.end);
  const caret = acContext.start + item.insert.length;
  editor.setSelectionRange(caret, caret);
  hideAutocomplete();
  syncHighlight();
  scheduleRun();
}

function updateAcSelection() {
  $$(".ac-item").forEach((el, index) => el.classList.toggle("selected", index === acSelected));
}

editor.addEventListener("input", () => {
  syncHighlight();
  showAutocomplete();
  scheduleRun();
});
editor.addEventListener("scroll", syncHighlight);
editor.addEventListener("blur", () => setTimeout(hideAutocomplete, 150));
editor.addEventListener("keydown", (event) => {
  if (!autocompleteBox.classList.contains("hidden") && acItems.length) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      acSelected = (acSelected + 1) % acItems.length;
      return updateAcSelection();
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      acSelected = (acSelected - 1 + acItems.length) % acItems.length;
      return updateAcSelection();
    }
    if (event.key === "Tab" || event.key === "Enter") {
      event.preventDefault();
      return acceptAutocomplete(acSelected);
    }
    if (event.key === "Escape") {
      event.preventDefault();
      return hideAutocomplete();
    }
  }
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    runExpression();
  }
});
autocompleteBox.addEventListener("mousedown", (event) => {
  const item = event.target.closest(".ac-item");
  if (item) {
    event.preventDefault();
    acceptAutocomplete(Number(item.dataset.index));
  }
});

function setExpression(expr, { run = true } = {}) {
  editor.value = expr;
  syncHighlight();
  hideAutocomplete();
  if (run) scheduleRun();
}

/* ----------------------------------------------------------------
   Dataset rendering & selection
   ---------------------------------------------------------------- */
function renderDatasetTabs() {
  const tabs = $("#dataset-tabs");
  tabs.innerHTML = Object.entries(DATASETS)
    .map(
      ([key, ds]) =>
        `<button class="dataset-tab ${key === state.dataset ? "active" : ""}" data-ds="${key}">${ds.label}</button>`
    )
    .join("");
  tabs.querySelectorAll(".dataset-tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      state.dataset = tab.dataset.ds;
      renderDatasetTabs();
      renderInputTable();
      scheduleRun();
    })
  );
}

function tableHtml(columns, rows, { resultColumn = null } = {}) {
  const header = columns
    .map(
      (col) =>
        `<th class="${col.name === resultColumn ? "result-col" : ""}">${escapeHtml(col.name)}<span class="dtype">${escapeHtml(col.dtype)}</span></th>`
    )
    .join("");
  const body = rows
    .map(
      (row) =>
        "<tr>" +
        row
          .map((value, index) => {
            const cls = columns[index].name === resultColumn ? "result-col" : "";
            if (value === null || value === undefined) return `<td class="null ${cls}">null</td>`;
            return `<td class="${cls}">${escapeHtml(value)}</td>`;
          })
          .join("") +
        "</tr>"
    )
    .join("");
  return `<table class="data-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderInputTable() {
  const ds = DATASETS[state.dataset];
  const rowCount = ds.columns[0].values.length;
  const rows = [];
  for (let i = 0; i < rowCount; i++) {
    rows.push(ds.columns.map((col) => (col.values[i] === null ? null : String(col.values[i]))));
  }
  $("#input-table").innerHTML = tableHtml(
    ds.columns.map((c) => ({ name: c.name, dtype: c.dtype })),
    rows
  );
}

function renderExamples() {
  $("#example-chips").innerHTML = EXAMPLES.map(
    (example, index) => `<button class="example-chip" data-index="${index}" title="${escapeHtml(example.expr)}">${escapeHtml(example.label)}</button>`
  ).join("");
  $$(".example-chip").forEach((chip) =>
    chip.addEventListener("click", () => {
      const example = EXAMPLES[Number(chip.dataset.index)];
      if (example.dataset !== state.dataset) {
        state.dataset = example.dataset;
        renderDatasetTabs();
        renderInputTable();
      }
      setExpression(example.expr);
    })
  );
}

/* ----------------------------------------------------------------
   Pyodide bootstrap
   ---------------------------------------------------------------- */
function setRuntimeStatus(text, kind = "loading") {
  const box = $("#runtime-status");
  box.className = `runtime-status ${kind}`;
  $("#runtime-status-text").innerHTML = text;
}

async function bootRuntime(wheelPath) {
  try {
    setRuntimeStatus("Loading the playground…");
    const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });
    state.pyodide = pyodide;

    setRuntimeStatus("Loading Polars (about 15 MB on first visit, cached afterwards)…");
    await pyodide.loadPackage(["micropip", "polars", "pydantic"]);

    setRuntimeStatus("Installing <code>polars-expr-transformer</code>…");
    const wheelUrl = new URL(`assets/${wheelPath}`, window.location.href).href;
    await pyodide.runPythonAsync(
      `import micropip\nawait micropip.install("${wheelUrl}", deps=False)`
    );

    setRuntimeStatus("Starting…");
    const runtimeSource = await (await fetch("assets/runtime.py")).text();
    pyodide.runPython(runtimeSource);
    state.runFn = pyodide.globals.get("run_expression");

    state.ready = true;
    const version = state.reference ? ` ${state.reference.version}` : "";
    setRuntimeStatus(
      `Ready (polars-expr-transformer${version}, Polars ` +
        `${pyodide.runPython("import polars; polars.__version__")}).`,
      "ready"
    );
    $("#run-btn").disabled = false;
    runExpression();
  } catch (error) {
    console.error(error);
    setRuntimeStatus(
      `Could not start the playground: ${escapeHtml(error.message || String(error))}. ` +
        "It needs WebAssembly and access to cdn.jsdelivr.net; the rest of the page works without it.",
      "error"
    );
  }
}

/* ----------------------------------------------------------------
   Running expressions
   ---------------------------------------------------------------- */
const scheduleRun = debounce(() => runExpression(), 350);

function runExpression() {
  if (!state.ready) return;
  if (state.runPending) {
    state.runQueued = true;
    return;
  }
  const expr = editor.value.trim();
  if (!expr) {
    $("#run-status").textContent = "";
    return;
  }
  state.runPending = true;
  const started = performance.now();
  let payload;
  try {
    payload = state.runFn(
      JSON.stringify({ expression: editor.value, dataset: DATASETS[state.dataset] })
    );
  } catch (error) {
    state.runPending = false;
    showError(`Unexpected runtime error: ${error.message || error}`);
    return;
  }
  state.runPending = false;
  const elapsed = Math.max(1, Math.round(performance.now() - started));
  renderRunResult(JSON.parse(payload), elapsed);
  if (state.runQueued) {
    state.runQueued = false;
    runExpression();
  }
}

function showError(message) {
  const box = $("#error-box");
  box.textContent = message;
  box.classList.remove("hidden");
  const status = $("#run-status");
  status.textContent = "error";
  status.className = "run-status err";
}

function renderRunResult(result, elapsedMs) {
  const errorBox = $("#error-box");
  const status = $("#run-status");

  if (result.polars_code) {
    $("#polars-code").innerHTML = highlightPython(result.polars_code);
    $("#flowframe-code").innerHTML = highlightPython(result.flowframe_code);
  } else {
    const note = result.codegen_error
      ? `<span class="placeholder">Code generation failed: ${escapeHtml(result.codegen_error)}</span>`
      : '<span class="placeholder">—</span>';
    $("#polars-code").innerHTML = note;
    $("#flowframe-code").innerHTML = note;
  }

  if (result.ok) {
    errorBox.classList.add("hidden");
    $("#result-table").innerHTML = tableHtml(result.columns, result.rows, { resultColumn: "result" });
    const resultDtype = result.columns[result.columns.length - 1].dtype;
    status.textContent = `ok · ${resultDtype} · ${elapsedMs} ms`;
    status.className = "run-status ok";
  } else {
    const stageLabel = result.stage === "parse" ? "Could not parse expression" : "Expression failed while executing";
    errorBox.textContent = `${stageLabel}:\n${result.error}`;
    errorBox.classList.remove("hidden");
    if (result.stage === "parse") {
      $("#result-table").innerHTML = '<div class="placeholder">Fix the expression to see results.</div>';
    }
    status.textContent = "error";
    status.className = "run-status err";
  }
}

/* ----------------------------------------------------------------
   Output tabs / copy / share
   ---------------------------------------------------------------- */
function setupOutputTabs() {
  const tabs = $$(".output-tab");
  const activate = (tab) => {
    tabs.forEach((t) => {
      const selected = t === tab;
      t.classList.toggle("active", selected);
      t.setAttribute("aria-selected", String(selected));
      t.tabIndex = selected ? 0 : -1;
    });
    $$(".output-pane").forEach((pane) =>
      pane.classList.toggle("hidden", pane.dataset.pane !== tab.dataset.tab)
    );
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : tabs.length - 1;
      const next = tabs[(index + step) % tabs.length];
      next.focus();
      activate(next);
    });
  });
}

function setupCopyButtons() {
  document.body.addEventListener("click", (event) => {
    const button = event.target.closest(".copy-btn");
    if (!button) return;
    if (button.dataset.copy) return copyText(button.dataset.copy, button);
    if (button.dataset.copyTarget) {
      const target = document.getElementById(button.dataset.copyTarget);
      if (target) copyText(target.textContent, button);
    }
  });
}

function setupShare() {
  $("#share-btn").addEventListener("click", (event) => {
    const url =
      window.location.origin +
      window.location.pathname +
      `#ds=${encodeURIComponent(state.dataset)}&expr=${encodeURIComponent(editor.value)}`;
    copyText(url, event.currentTarget);
  });
}

function restoreFromHash() {
  if (!window.location.hash.startsWith("#ds=")) return false;
  const params = new URLSearchParams(window.location.hash.slice(1));
  const ds = params.get("ds");
  const expr = params.get("expr");
  if (ds && DATASETS[ds]) state.dataset = ds;
  if (expr) {
    setExpression(expr, { run: false });
    return true;
  }
  return false;
}

/* ----------------------------------------------------------------
   Function reference
   ---------------------------------------------------------------- */
const refState = { search: "", category: "all" };

function renderReferenceControls(reference) {
  const pills = [
    { key: "all", label: "All", count: reference.total_functions },
    ...reference.categories.map((category) => ({
      key: category.key,
      label: category.label,
      count: category.functions.length,
    })),
  ];
  $("#category-pills").innerHTML = pills
    .map(
      (pill) =>
        `<button class="category-pill ${pill.key === refState.category ? "active" : ""}" data-cat="${pill.key}">
          ${escapeHtml(pill.label)} <span class="count">${pill.count}</span>
        </button>`
    )
    .join("");
  $$(".category-pill").forEach((pill) =>
    pill.addEventListener("click", () => {
      refState.category = pill.dataset.cat;
      $$(".category-pill").forEach((p) => p.classList.toggle("active", p === pill));
      renderReference();
    })
  );
}

function functionMatches(fn, category) {
  if (refState.category !== "all" && refState.category !== category.key) return false;
  if (!refState.search) return true;
  const haystack = `${fn.name} ${fn.description} ${category.label}`.toLowerCase();
  return refState.search
    .toLowerCase()
    .split(/\s+/)
    .every((term) => haystack.includes(term));
}

function fnCardHtml(fn) {
  const tryButton = fn.try_expression
    ? `<button class="fn-try" data-expr="${escapeHtml(fn.try_expression)}" data-ds="${escapeHtml(fn.try_dataset || "")}">Try it ▸</button>`
    : "";
  const example = fn.example_call
    ? `<div class="fn-example">${highlightExpression(fn.example_call)}` +
      (fn.example_result ? `<span class="arrow">→</span><span class="res">${escapeHtml(fn.example_result)}</span>` : "") +
      (fn.example_context ? `<span class="ctx">when ${escapeHtml(fn.example_context)}</span>` : "") +
      "</div>"
    : "";
  const params = fn.parameters.length
    ? `<details class="fn-more"><summary>Parameters &amp; return value</summary>
        <ul class="fn-params">
          ${fn.parameters
            .map(
              (p) =>
                `<li><code>${escapeHtml(p.name)}</code> <em>(${escapeHtml(p.type)}${p.default !== undefined ? `, default ${escapeHtml(p.default)}` : ""})</em>${p.description ? " — " + escapeHtml(p.description) : ""}</li>`
            )
            .join("")}
          ${fn.returns ? `<li><strong>returns</strong> — ${escapeHtml(fn.returns)}</li>` : ""}
        </ul>
      </details>`
    : fn.returns
      ? `<p class="fn-params"><strong>returns</strong> — ${escapeHtml(fn.returns)}</p>`
      : "";
  return `<div class="fn-card">
      <div class="fn-card-head"><span class="fn-sig">${escapeHtml(fn.signature)}</span>${tryButton}</div>
      <p class="fn-desc">${escapeHtml(fn.description)}</p>
      ${example}
      ${params}
    </div>`;
}

function renderReference() {
  const reference = state.reference;
  if (!reference) return;
  let html = "";
  let shown = 0;
  for (const category of reference.categories) {
    const functions = category.functions.filter((fn) => functionMatches(fn, category));
    if (!functions.length) continue;
    shown += functions.length;
    html += `<h3 class="fn-category-title">${escapeHtml(category.label)} <small>(${functions.length})</small></h3>
      <div class="fn-grid">${functions.map(fnCardHtml).join("")}</div>`;
  }
  $("#fn-list").innerHTML = html;
  $("#fn-empty").classList.toggle("hidden", shown > 0);
  $$(".fn-try").forEach((button) =>
    button.addEventListener("click", () => {
      const ds = button.dataset.ds;
      if (ds && DATASETS[ds] && ds !== state.dataset) {
        state.dataset = ds;
        renderDatasetTabs();
        renderInputTable();
      }
      setExpression(button.dataset.expr);
      $("#playground").scrollIntoView({ behavior: "smooth" });
    })
  );
}

/* ----------------------------------------------------------------
   Theme toggle (light-first, matching the Flowfile docs theme)
   ---------------------------------------------------------------- */
function setupThemeToggle() {
  const root = document.documentElement;
  const button = $("#theme-toggle");
  const apply = (theme) => {
    root.dataset.theme = theme;
    button.textContent = theme === "dark" ? "☀️" : "🌙";
  };
  apply(root.dataset.theme || "light"); // set by the inline head script
  button.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    localStorage.setItem("fx-theme", next);
    apply(next);
  });
}

function setupSearch() {
  $("#fn-search").addEventListener(
    "input",
    debounce((event) => {
      refState.search = event.target.value.trim();
      renderReference();
    }, 120)
  );
}

/* ----------------------------------------------------------------
   Init
   ---------------------------------------------------------------- */
async function init() {
  renderDatasetTabs();
  renderInputTable();
  renderExamples();
  setupOutputTabs();
  setupCopyButtons();
  setupShare();
  setupSearch();
  setupThemeToggle();

  $("#run-btn").addEventListener("click", () => runExpression());

  const restored = restoreFromHash();
  if (restored) {
    renderDatasetTabs();
    renderInputTable();
  } else {
    setExpression(EXAMPLES[0].expr, { run: false });
  }

  let wheelPath = "wheel/polars_expr_transformer-0.5.5-py3-none-any.whl";
  try {
    const reference = await (await fetch("assets/functions.json")).json();
    state.reference = reference;
    wheelPath = reference.wheel || wheelPath;
    state.functionIndex = reference.categories.flatMap((category) =>
      category.functions.map((fn) => ({
        name: fn.name,
        signature: fn.signature,
        description: fn.description,
        example_call: fn.example_call,
      }))
    );
    $("#hero-fn-count").textContent = reference.total_functions;
    $$(".fn-count").forEach((el) => (el.textContent = reference.total_functions));
    renderReferenceControls(reference);
    renderReference();
  } catch (error) {
    console.error("Could not load function reference:", error);
    $("#fn-list").innerHTML = '<p class="placeholder">Could not load the function reference data.</p>';
  }

  bootRuntime(wheelPath);
}

init();
