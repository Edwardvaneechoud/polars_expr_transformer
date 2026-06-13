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

The system prompt is built at runtime from the function reference plus the
active dataset's column names, so it stays in sync with the catalog. Each
draft is run through the same `run_expression` parser the playground uses;
if it fails, the parser's error is fed back to the model for one repair
attempt before the result is shown. The default model is `WEBLLM_MODEL` in
`assets/app.js`; the picker's options (and their `q4f16_1` quantization)
are the `#ai-model` `<option>`s in `index.html`.

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
