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
   Natural language → expression (optional, in-browser via WebLLM)

   A small instruction-tuned model runs fully client-side on WebGPU and
   drafts an expression from a plain-English description. The model is
   only downloaded when the user first asks for it, so the default
   playground load is unaffected. Whatever the model produces is checked
   by the same parser the playground already runs (run_expression); on a
   failure its error is fed back for one repair attempt.
   ---------------------------------------------------------------- */
// Default model; users switch sizes via the #ai-model picker. q4f16_1 needs
// shader-f16 support (most modern GPUs) — swap the ids in index.html to the
// "…-q4f32_1-MLC" variants for GPUs without it.
const WEBLLM_MODEL = "Qwen2.5-1.5B-Instruct-q4f16_1-MLC";
const WEBLLM_ESM = "https://esm.run/@mlc-ai/web-llm";

const ai = {
  engine: null,
  loading: false,
  ready: false,
  busy: false,
  lessons: [],
  model: WEBLLM_MODEL,   // currently selected model id
  loadedModel: null,     // model id actually loaded into the engine
  useGrammar: true,      // constrain decoding to the formula grammar (XGrammar)
};

function aiStatus(html, kind = "") {
  const el = $("#ai-status");
  el.className = `ai-status ${kind}`;
  el.innerHTML = html;
}

/* Remember formulas the parser rejected so later prompts can avoid
   repeating them. Kept bounded so the prompt stays small. */
function rememberLesson(expr, error) {
  const note = `${expr} — ${String(error).split("\n")[0]}`;
  if (ai.lessons.includes(note)) return;
  ai.lessons.push(note);
  if (ai.lessons.length > 6) ai.lessons.shift();
}

/* Build the system prompt from the live function reference plus the
   columns of the active dataset. Mirrors the catalog shown in the
   reference, so it stays in sync as functions change. */
function buildAiSystemPrompt() {
  const catalog = [];
  for (const category of state.reference.categories) {
    catalog.push(`\n[${category.label}]`);
    for (const fn of category.functions) {
      const desc = (fn.description || "").replace(/\s+/g, " ").split(". ")[0].replace(/\.$/, "");
      catalog.push(`- ${fn.signature} — ${desc}`);
    }
  }
  const columns = DATASETS[state.dataset].columns.map((c) => `[${c.name}]`).join(", ");
  const parts = [
    "You translate a user's natural-language request into a SINGLE polars-expr-transformer formula.",
    "",
    "Output rules:",
    "- Reply with ONLY the formula, on one line. No explanation, no markdown, no code fences.",
    "- Use the functions listed below with their names spelled exactly as shown (for example uppercase, not upper; lowercase, not lower). Never invent function names.",
    "- If no function fits, build the result from operators and conditionals.",
    "- Reference columns with square brackets, e.g. [first_name]. Spaces are allowed: [Order Date].",
    "",
    "Syntax:",
    `- String literals use single or double quotes: "hello", 'world'. Numbers and booleans are bare: 42, 3.14, -7, true, false.`,
    "- Conditionals: if <condition> then <value> elseif <condition> then <value> else <value> endif (elseif may repeat or be omitted; else is required).",
    "- Operators: + - * / % (arithmetic; + also concatenates text) | = == != (equality) | > >= < <= (comparison) | and or (boolean) | ( ) for grouping.",
    "- There is no [..] indexing or slicing. For the last character of text use right([col], 1); for the first, left([col], 1); for the middle, mid([col], start, length).",
    "- Functions may be nested: uppercase(left([last_name], 3)).",
    "- For missing/null values use is_empty, is_not_empty, coalesce or ifnull.",
    "- To read a part of a date use year([col]), month([col]), day([col]), hour([col]), weekday([col]), etc. to_date / to_datetime only convert text into a date — never use them to extract a part.",
    "",
    `Columns in the current dataset: ${columns}. Prefer these names; only use a different [name] if the request clearly refers to one.`,
    "",
    "Available functions:",
    catalog.join("\n"),
    "",
    "Examples (request -> formula):",
    `"Combine first and last name with a space between them" -> concat([first_name], " ", [last_name])`,
    `"Label anyone older than 30 as Senior, everyone else Junior" -> if [age] > 30 then "Senior" else "Junior" endif`,
    `"Multiply price by quantity and round to 2 decimals" -> round([price] * [quantity], 2)`,
    `"Uppercase the first three letters of the last name" -> uppercase(left([last_name], 3))`,
    `"Last letter of the first name" -> right([first_name], 1)`,
    `"Get the year from the start date" -> year([start])`,
    `"Use the nickname, or the full name when there is no nickname" -> coalesce([nickname], [name])`,
    `"Grade: A for 90 or above, B for 80 or above, otherwise C" -> if [score] >= 90 then "A" elseif [score] >= 80 then "B" else "C" endif`,
  ];
  if (ai.lessons.length) {
    parts.push("", "Avoid these formulas the parser rejected earlier in this session:");
    for (const lesson of ai.lessons) parts.push(`- ${lesson}`);
  }
  parts.push("", "Reply with only the formula.");
  return parts.join("\n");
}

/* Build an EBNF grammar for the formula language from the live function
   list and the active dataset's columns. Passed to WebLLM as a grammar
   response_format so the model can only emit a syntactically valid formula
   that uses real function and column names — hallucinated names, [..]
   indexing and the like become impossible at decode time. Validated with
   XGrammar; the downstream parser remains the semantic backstop. */
function buildFormulaGrammar() {
  const esc = (s) => s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  const fnames = state.reference.categories.flatMap((c) => c.functions.map((f) => f.name));
  const cols = DATASETS[state.dataset].columns.map((c) => c.name);
  return [
    "root ::= ws expr ws",
    "expr ::= term (ws op ws term)*",
    "term ::= call | cond | group | col | str | num | bool",
    'group ::= "(" ws expr ws ")"',
    'call ::= fname ws "(" ws arglist? ws ")"',
    'arglist ::= expr (ws "," ws expr)*',
    'cond ::= "if" ws expr ws "then" ws expr ws elseifs "else" ws expr ws "endif"',
    'elseifs ::= ("elseif" ws expr ws "then" ws expr ws)*',
    'col ::= "[" colname "]"',
    'str ::= "\\"" dqchar* "\\"" | "\'" sqchar* "\'"',
    'dqchar ::= [^"\\n]',
    "sqchar ::= [^'\\n]",
    'num ::= "-"? digit+ ("." digit+)?',
    "digit ::= [0-9]",
    'bool ::= "true" | "false"',
    'op ::= "==" | "!=" | ">=" | "<=" | "+" | "-" | "*" | "/" | "%" | "=" | ">" | "<" | "and" | "or"',
    "fname ::= " + fnames.map((n) => `"${esc(n)}"`).join(" | "),
    "colname ::= " + cols.map((c) => `"${esc(c)}"`).join(" | "),
    "ws ::= [ \\t\\n]*",
  ].join("\n");
}

/* One model call, grammar-constrained when supported. Falls back to plain
   generation (prompt rules + parser repair loop) if the engine rejects the
   grammar response_format, and remembers that for the session. */
async function aiComplete(messages) {
  const base = { messages, temperature: 0.1, max_tokens: 256 };
  if (ai.useGrammar) {
    try {
      return await ai.engine.chat.completions.create({
        ...base,
        response_format: { type: "grammar", grammar: buildFormulaGrammar() },
      });
    } catch (error) {
      console.warn("Grammar-constrained generation unavailable; using the prompt only.", error);
      ai.useGrammar = false;
    }
  }
  return await ai.engine.chat.completions.create(base);
}

/* Pull a bare expression out of whatever the model returned. */
function cleanModelOutput(text) {
  let out = (text || "").trim();
  const fenced = out.match(/```[a-zA-Z]*\n?([\s\S]*?)```/);
  if (fenced) out = fenced[1].trim();
  out = out.split(/\n\s*\n/)[0].trim();              // first paragraph only
  out = out.replace(/^(?:->|=|expression:)\s*/i, "").trim();
  out = out.replace(/^`+|`+$/g, "").trim();          // stray inline backticks
  return out;
}

/* Run a candidate through the existing runtime; return null if it parses
   and executes on the active dataset, otherwise the error message. */
function checkExpression(expr) {
  if (!state.ready || !state.runFn) return null;     // runtime not up yet: skip
  try {
    const result = JSON.parse(
      state.runFn(JSON.stringify({ expression: expr, dataset: DATASETS[state.dataset] }))
    );
    return result.ok ? null : result.error || "unknown error";
  } catch {
    return null;                                      // never block on a harness hiccup
  }
}

async function ensureAiEngine() {
  if (ai.ready && ai.loadedModel === ai.model) return ai.engine;  // already loaded
  if (ai.loading) return null;
  if (!navigator.gpu) {
    aiStatus(
      "This feature needs WebGPU, which this browser doesn't expose. Try the latest desktop Chrome or Edge.",
      "err"
    );
    return null;
  }
  ai.loading = true;
  try {
    if (ai.engine && typeof ai.engine.unload === "function") {
      await ai.engine.unload();        // free the previous model before switching
    }
    ai.ready = false;
    aiStatus("Loading WebLLM…");
    const webllm = await import(WEBLLM_ESM);
    ai.engine = await webllm.CreateMLCEngine(ai.model, {
      initProgressCallback: (report) => {
        const pct = report.progress ? ` (${Math.round(report.progress * 100)}%)` : "";
        aiStatus(`${escapeHtml(report.text || "Loading model…")}${pct}`);
      },
    });
    ai.ready = true;
    ai.loadedModel = ai.model;
    return ai.engine;
  } catch (error) {
    console.error(error);
    aiStatus(`Could not load the model: ${escapeHtml(error.message || String(error))}`, "err");
    return null;
  } finally {
    ai.loading = false;
  }
}

async function generateFromNl() {
  const nl = $("#ai-input").value.trim();
  if (!nl) return $("#ai-input").focus();
  if (ai.busy) return;
  if (!state.reference) {
    return aiStatus("The function reference hasn't loaded yet — try again in a moment.", "err");
  }

  const engine = await ensureAiEngine();
  if (!engine) return;

  ai.busy = true;
  $("#ai-generate").disabled = true;
  try {
    if (typeof engine.resetChat === "function") await engine.resetChat();  // fresh session each generation

    const messages = [
      { role: "system", content: buildAiSystemPrompt() },
      { role: "user", content: nl },
    ];
    let expr = "";
    let problem = null;
    for (let attempt = 0; attempt < 3; attempt++) {
      aiStatus(attempt === 0 ? "Generating…" : "Adjusting the formula…");
      const reply = await aiComplete(messages);
      expr = cleanModelOutput(reply.choices?.[0]?.message?.content);
      problem = expr ? checkExpression(expr) : "empty response";
      if (!problem) break;
      rememberLesson(expr, problem);                  // carry the mistake into later prompts
      // Hand the parser's own error back so the model can correct itself.
      messages.push({ role: "assistant", content: expr });
      messages.push({
        role: "user",
        content: `That formula is not valid. The parser said:\n${problem}\nReturn a corrected formula. Reply with only the formula.`,
      });
    }
    if (!expr) {
      return aiStatus("The model didn't return a formula — try rephrasing.", "err");
    }
    setExpression(expr);                              // fills the editor and runs it
    aiStatus(
      problem
        ? "Generated a formula, but it didn't pass the parser — review and fix it below."
        : `Generated from “${escapeHtml(nl)}”. Review it above and tweak as needed.`,
      problem ? "err" : "ok"
    );
  } catch (error) {
    console.error(error);
    aiStatus(`Generation failed: ${escapeHtml(error.message || String(error))}`, "err");
  } finally {
    ai.busy = false;
    $("#ai-generate").disabled = false;
  }
}

function setupAi() {
  const toggle = $("#ai-toggle");
  const panel = $("#ai-panel");
  if (!toggle || !panel) return;
  toggle.addEventListener("click", () => {
    const open = !panel.classList.toggle("hidden");
    toggle.setAttribute("aria-expanded", String(open));
    toggle.classList.toggle("active", open);
    if (open) $("#ai-input").focus();
  });
  const modelSelect = $("#ai-model");
  if (modelSelect) {
    modelSelect.value = ai.model;
    modelSelect.addEventListener("change", () => {
      ai.model = modelSelect.value;
      if (ai.loadedModel && ai.loadedModel !== ai.model) {
        aiStatus("Model changed — it will download on the next generate.");
      }
    });
  }
  $("#ai-generate").addEventListener("click", generateFromNl);
  $("#ai-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      generateFromNl();
    }
  });
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
  setupAi();

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
