# Documentation & playground site

This directory is a static site with documentation and a playground for
`polars-expr-transformer` (also referred to as flowfile expressions).
The playground runs the library in the browser via
[Pyodide](https://pyodide.org), so expressions are parsed and executed
client-side; there is no server component.

## Structure

| Path | Purpose |
|------|---------|
| `index.html` | The single-page app: playground, syntax guide, function reference. |
| `assets/app.js` | UI logic: editor with highlighting + autocomplete, datasets, Pyodide bootstrap, function reference rendering. |
| `assets/style.css` | Styles. |
| `assets/runtime.py` | Python entry point loaded into Pyodide; evaluates expressions and returns results/generated code as JSON. |
| `assets/functions.json` | Function reference data, generated from the library's docstrings by `../generate_docs.py`. |
| `assets/wheel/*.whl` | The library wheel installed into Pyodide, built from this repository. |

## Natural-language input (optional)

The Expression panel has a **Generate flowfile formula** button that drafts a
formula from a plain-English description. A small instruction-tuned
[Qwen2.5](https://huggingface.co/Qwen) model runs entirely client-side on
WebGPU via [WebLLM](https://github.com/mlc-ai/web-llm), loaded on demand
from the `esm.run` CDN — nothing is downloaded until a visitor first uses
the feature, so the default page load is unchanged. A picker offers 0.5B
(~0.4 GB), 1.5B (~1 GB, default) and 3B (~2 GB); switching reloads on the
next generate. It needs a WebGPU browser (desktop Chrome or Edge).

Generation is constrained by an EBNF grammar built at runtime from the
function list and the active dataset's columns and enforced in the browser
by WebLLM/[XGrammar](https://xgrammar.mlc.ai) (`response_format: { type:
"grammar" }`). The model can therefore only emit a syntactically valid
formula that uses real function and column names — hallucinated names,
`[..]` indexing and the like are impossible at decode time; if a model or
engine can't honour the grammar the code falls back to plain generation.
The system prompt is built from the same catalog so it stays in sync.

As a semantic backstop each draft is still run through the
`run_expression` parser; if it fails to execute, the error is fed back for
a repair attempt (up to three), and formulas the parser rejects are
remembered to steer later prompts in the session. The default model is
`WEBLLM_MODEL` in `assets/app.js`; the picker's options (and their
`q4f16_1` quantization) are the `#ai-model` `<option>`s in `index.html`;
the grammar itself is `buildFormulaGrammar()`.

## Developing locally

```bash
# 1. Regenerate the function reference (after changing docstrings)
pip install polars polars-ds pydantic
python generate_docs.py

# 2. Rebuild the wheel served to the browser (after changing library code)
pip wheel . --no-deps -w docs/assets/wheel/

# 3. Serve the site (any static file server works)
python -m http.server --directory docs 8000
```

Then open <http://localhost:8000>. Note that the playground downloads
Pyodide and Polars (~15 MB) from the jsDelivr CDN on first load, so an
internet connection is required even when serving locally.

`runtime.py` is plain Python and can be tested against a local
interpreter — `run_expression` takes and returns JSON strings, see its
docstring.

## Deployment

`.github/workflows/deploy-docs.yml` regenerates `functions.json`,
rebuilds the wheel and publishes this directory to GitHub Pages on every
push to `main` (it can also be started manually from the Actions tab).
The workflow tries to enable Pages on its first run; if that fails, set
Settings → Pages → Source to "GitHub Actions" once and re-run it.
