"""
payload_report.py
==================

Render a list of InsightsPayload objects (`payload_results`) as a clean,
INDEXED HTML report. Each dataset is a numbered card; inside it the
"Model signals" section is collapsible, and every individual model is its own
collapsible block (so you can hide/unhide the whole dump or any single model).

Input is exactly the object you iterate in your notebook:

    for pr in payload_results:
        print(pr)                                   # full payload
        print(f"{pr.dataset_name}: {pr.model_signals}")

Each payload must expose `.to_json()` (used to read columns + model_signals).

Usage in a Databricks notebook
------------------------------
Most reliable load (sidesteps /Workspace import issues):

    import importlib.util, os
    _path = os.path.join(_repo_root, "payload_report.py")
    spec = importlib.util.spec_from_file_location("payload_report", _path)
    payload_report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(payload_report)

    payload_report.show_payload_report(payload_results, displayHTML)

Or, if normal imports work:

    from payload_report import show_payload_report
    show_payload_report(payload_results, displayHTML)

Get the HTML string (e.g. to save to a file):

    from payload_report import payloads_to_html
    html = payloads_to_html(payload_results)

Per dataset (numbered card):
  * measure columns (count + names)        -- schema subtype=='measure', else by_measure
  * Measure -> Model mapping               -- which models modeled each measure
  * Candidates (model -> candidate_window) -- which model produced which window
  * Model signals                          -- collapsible; each model indexed & collapsible
A summary table at the top indexes every dataset and links to its card.
A toolbar provides Expand all / Collapse all for the signal sections.
"""

import json
import html as _html

__all__ = ["payloads_to_html", "show_payload_report"]


# ----------------------------------------------------------------------
# value formatting
# ----------------------------------------------------------------------
def _fmt_scalar(v):
    if isinstance(v, bool):  return str(v)
    if isinstance(v, float): return f"{v:,.4g}"
    if isinstance(v, int):   return f"{v:,}"
    return _html.escape(str(v))


def _is_long_num_list(v):
    return isinstance(v, list) and len(v) > 12 and all(
        x is None or isinstance(x, (int, float)) for x in v)


def _render(value):
    """Recursively render an arbitrary JSON-ish value as compact HTML."""
    if isinstance(value, dict):
        if not value:
            return '<span class="muted">{}</span>'
        rows = "".join(
            f'<tr><th>{_html.escape(str(k))}</th><td>{_render(v)}</td></tr>'
            for k, v in value.items())
        return f'<table class="kv">{rows}</table>'
    if isinstance(value, list):
        if _is_long_num_list(value):
            head = ", ".join(_fmt_scalar(x) for x in value[:5])
            tail = ", ".join(_fmt_scalar(x) for x in value[-3:])
            return (f'<span class="series">[{head}, … , {tail}]</span>'
                    f' <span class="muted">({len(value)} values)</span>')
        if not value:
            return '<span class="muted">[]</span>'
        return '<ul class="lst">' + "".join(f"<li>{_render(x)}</li>" for x in value) + '</ul>'
    if value is None:
        return '<span class="muted">null</span>'
    return _fmt_scalar(value)


# ----------------------------------------------------------------------
# measure columns + measure -> model mapping
# ----------------------------------------------------------------------
def _by_measure_containers(sig):
    """by_measure can live on the signal directly or under signal['signals']."""
    out = [sig.get("by_measure")]
    inner = sig.get("signals")
    if isinstance(inner, dict):
        out.append(inner.get("by_measure"))
    return out


def resolve_measures(data):
    """(names, source).
    Authoritative: schema columns with subtype == 'measure'.
    Fallback: measures actually modeled (keys of by_measure across model_signals).
    """
    schema = [c.get("name") for c in data.get("columns", [])
              if isinstance(c, dict) and c.get("subtype") == "measure" and c.get("name")]
    if schema:
        return schema, "schema"
    modeled = {}
    for sig in data.get("model_signals", {}).values():
        if isinstance(sig, dict):
            for bm in _by_measure_containers(sig):
                if isinstance(bm, dict):
                    for k in bm:
                        modeled[k] = True
                elif isinstance(bm, list):
                    for k in bm:
                        modeled[str(k)] = True
    names = list(modeled)
    return names, ("modeled" if names else "none")


def measure_model_map(data):
    """measure -> sorted list of models that modeled it (from by_measure).
    Schema measures with no model are included with an empty list."""
    mapping = {}
    for model, sig in data.get("model_signals", {}).items():
        if not isinstance(sig, dict):
            continue
        for bm in _by_measure_containers(sig):
            if isinstance(bm, dict):
                for measure in bm:
                    mapping.setdefault(measure, set()).add(model)
            elif isinstance(bm, list):
                for measure in bm:
                    mapping.setdefault(str(measure), set()).add(model)
    mapping = {k: sorted(v) for k, v in mapping.items()}
    schema, src = resolve_measures(data)
    if src == "schema":
        for m in schema:
            mapping.setdefault(m, [])
    return mapping


# ----------------------------------------------------------------------
# candidates
# ----------------------------------------------------------------------
def collect_candidates(signals):
    """Each candidate = a model signal that produced a candidate_window."""
    out = []
    for model, sig in signals.items():
        if not isinstance(sig, dict):
            continue
        cw = sig.get("candidate_window")
        if not cw:
            continue
        out.append({
            "model":   model,
            "label":   f'{cw.get("years")}yr  {cw.get("cutoff_date")} → {cw.get("max_date")}',
            "rows":    cw.get("n_rows"),
            "ran":     sig.get("ran"),
            "quality": sig.get("quality_decision"),
        })
    return out


# ----------------------------------------------------------------------
# small render helpers
# ----------------------------------------------------------------------
def _stat(label, val, title=""):
    shown = "?" if val is None else val
    cls = "stat-unknown" if val is None else "stat"
    t = f' title="{_html.escape(title)}"' if title else ""
    return f'<span class="{cls}"{t}><b>{shown}</b> {label}</span>'


def _num(v):
    return f'{v:,}' if isinstance(v, int) else _html.escape(str(v))


def _models_cell(models):
    if not models:
        return '<span class="muted">not modeled</span>'
    return "".join(f'<span class="mdl-chip">{_html.escape(str(x))}</span>' for x in models)


def _measure_model_block(data):
    mmap = measure_model_map(data)
    _, src = resolve_measures(data)
    tag = {"schema": "schema columns", "modeled": "from by_measure"}.get(src, src)
    if not mmap:
        return '<div class="mmap"><span class="muted">no measure → model mapping</span></div>'
    rows = "".join(
        f'<tr><td class="mname">{_html.escape(str(m))}</td><td>{_models_cell(models)}</td></tr>'
        for m, models in sorted(mmap.items()))
    return ('<div class="mmap"><div class="mmap-h">Measure → Model &nbsp;'
            f'<span class="muted">({tag})</span></div>'
            '<table class="mmap-tbl"><thead><tr>'
            '<th>Measure column</th><th>Modeled by</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>')


def _candidates_block(cands):
    if not cands:
        return ('<div class="cands"><span class="muted">'
                'no candidates (no model produced a candidate_window)</span></div>')
    rows = "".join(
        f'<tr><td class="cmodel">{_html.escape(str(c["model"]))}</td>'
        f'<td>{_html.escape(c["label"])}</td>'
        f'<td class="num">{_num(c["rows"])}</td>'
        f'<td>{"ran" if c["ran"] else "—"}</td>'
        f'<td>{_html.escape(str(c["quality"]))}</td></tr>'
        for c in cands)
    return ('<div class="cands"><div class="cands-h">Candidates &nbsp;'
            '<span class="muted">model → window</span></div>'
            '<table class="cand-tbl"><thead><tr>'
            '<th>From model</th><th>Candidate window</th><th>Rows</th><th>Ran</th><th>Quality</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>')


def _model_signals_block(card_idx, sigs):
    """Collapsible 'Model signals' section; each model is its own indexed,
    collapsible <details> so it can be hidden/unhidden individually."""
    if not sigs:
        return ('<details class="raw" open><summary>Model signals</summary>'
                '<div class="raw-body"><span class="muted">no model signals</span></div></details>')
    model_blocks = ""
    for j, (mn, sig) in enumerate(sigs.items(), 1):
        model_blocks += (
            f'<details class="model" open id="card-{card_idx}-m{j}">'
            f'<summary><span class="midx">{card_idx}.{j}</span>'
            f'<span class="model-name">{_html.escape(str(mn))}</span></summary>'
            f'<div class="model-body">{_render(sig)}</div>'
            f'</details>')
    return (f'<details class="raw" open><summary>Model signals '
            f'<span class="muted">({len(sigs)} model(s))</span></summary>'
            f'<div class="raw-body">{model_blocks}</div></details>')


# ----------------------------------------------------------------------
# input parsing
# ----------------------------------------------------------------------
def _to_dict(pr):
    """Turn one payload object into a plain dict.
    Prefers .to_json(); falls back to attribute access if needed."""
    try:
        return json.loads(pr.to_json())
    except Exception:
        return {
            "dataset_name": getattr(pr, "dataset_name", "payload"),
            "model_signals": getattr(pr, "model_signals", {}) or {},
            "columns": getattr(pr, "columns", []) or [],
        }


# ----------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------
def payloads_to_html(payload_results):
    """Build the full HTML report string from a list of payload objects."""
    parsed = []
    for pr in payload_results:
        data = _to_dict(pr)
        name = data.get("dataset_name", getattr(pr, "dataset_name", "payload"))
        sigs = data.get("model_signals", {}) or {}
        m_names, m_source = resolve_measures(data)
        cands = collect_candidates(sigs)
        parsed.append((name, data, sigs, m_names, m_source, cands))

    # ---- indexed summary table ----
    srows = ""
    for i, (n, _, _, mn, _src, cands) in enumerate(parsed, 1):
        models = ", ".join(c["model"] for c in cands) or "—"
        mnames = ", ".join(map(str, mn)) if mn else "—"
        srows += (
            f'<tr><td class="num">{i}</td>'
            f'<td><a href="#card-{i}" class="ds-link">{_html.escape(str(n))}</a></td>'
            f'<td class="num">{len(mn) if mn else "?"}</td>'
            f'<td class="mnames">{_html.escape(mnames)}</td>'
            f'<td class="num">{len(cands)}</td>'
            f'<td class="cmodels">{_html.escape(models)}</td></tr>')
    summary = (
        '<table class="summary"><thead><tr>'
        '<th>#</th><th>Dataset</th><th>Measures</th><th>Measure columns</th>'
        '<th>Candidates</th><th>Candidate models</th>'
        f'</tr></thead><tbody>{srows}</tbody></table>')

    # ---- detail cards ----
    cards = []
    for i, (name, data, sigs, m_names, m_source, cands) in enumerate(parsed, 1):
        m_count = len(m_names) if m_names else None
        cards.append(
            f'<section class="card" id="card-{i}">'
            f'<h2><span class="idx">#{i}</span> {_html.escape(str(name))}</h2>'
            f'<div class="stats">'
            f'{_stat("measure columns", m_count, ", ".join(map(str, m_names)))}'
            f'{_stat("candidates found", len(cands))}</div>'
            f'{_measure_model_block(data)}'
            f'{_candidates_block(cands)}'
            f'{_model_signals_block(i, sigs)}'
            f'</section>')

    toolbar = ('<div class="toolbar">'
               '<button onclick="prToggle(true)">Expand all signals</button>'
               '<button onclick="prToggle(false)">Collapse all signals</button>'
               '</div>')
    script = ('<script>function prToggle(open){'
              'document.querySelectorAll(".pr-wrap details.raw, .pr-wrap details.model")'
              '.forEach(function(d){d.open=open;});}</script>')

    style = """
    <style>
      .pr-wrap{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a1a;max-width:1000px}
      .pr-wrap .toolbar{margin:0 0 12px;display:flex;gap:8px}
      .pr-wrap .toolbar button{font-size:12px;padding:5px 12px;border:1px solid #0b3d91;background:#fff;color:#0b3d91;border-radius:6px;cursor:pointer}
      .pr-wrap .toolbar button:hover{background:#eef2fb}
      .pr-wrap .card{border:1px solid #e2e2e2;border-radius:10px;padding:18px 20px;margin:0 0 18px;
                     background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
      .pr-wrap h2{margin:0 0 4px;font-size:16px;color:#0b3d91;word-break:break-all;display:flex;align-items:center;gap:10px}
      .pr-wrap .idx{background:#0b3d91;color:#fff;font-size:11px;font-weight:700;border-radius:6px;padding:2px 8px}
      .pr-wrap .stats{margin:8px 0 12px;display:flex;gap:10px;flex-wrap:wrap}
      .pr-wrap .stat,.pr-wrap .stat-unknown{font-size:12px;padding:3px 10px;border-radius:20px;background:#eef2fb;color:#0b3d91;cursor:default}
      .pr-wrap .stat-unknown{background:#fbeeee;color:#a11}
      .pr-wrap .stat b,.pr-wrap .stat-unknown b{font-size:13px}
      .pr-wrap .mmap,.pr-wrap .cands{margin:0 0 16px}
      .pr-wrap .mmap-h,.pr-wrap .cands-h{font-size:13px;font-weight:600;color:#333;margin:14px 0 6px}
      .pr-wrap table.mmap-tbl,.pr-wrap table.cand-tbl{border-collapse:collapse;width:100%;font-size:12px}
      .pr-wrap table.mmap-tbl th,.pr-wrap table.cand-tbl th{text-align:left;background:#f4f6fa;color:#555;font-weight:500;padding:5px 10px;border-bottom:1px solid #e2e8f2}
      .pr-wrap table.mmap-tbl td,.pr-wrap table.cand-tbl td{padding:5px 10px;border-bottom:1px solid #f0f0f0}
      .pr-wrap table.mmap-tbl td.mname{font-family:ui-monospace,Menlo,monospace;color:#234;font-weight:600}
      .pr-wrap table.cand-tbl td.cmodel{font-weight:600;color:#0b3d91}
      .pr-wrap table.cand-tbl td.num{text-align:right;font-variant-numeric:tabular-nums}
      .pr-wrap .mdl-chip{display:inline-block;background:#eef2fb;color:#0b3d91;border-radius:5px;padding:1px 8px;margin:0 4px 2px 0;font-size:11px;font-weight:600}

      /* collapsible Model signals section */
      .pr-wrap details.raw{margin-top:10px}
      .pr-wrap details.raw>summary{cursor:pointer;font-size:13px;font-weight:600;color:#333;list-style:none;padding:6px 0}
      .pr-wrap details.raw>summary::-webkit-details-marker{display:none}
      .pr-wrap details.raw>summary:before{content:"▸ ";color:#0b3d91}
      .pr-wrap details.raw[open]>summary:before{content:"▾ "}
      .pr-wrap .raw-body{padding-top:4px}

      /* collapsible per-model block (indexed) */
      .pr-wrap details.model{border-left:3px solid #0b3d91;background:#fbfcfe;border-radius:6px;margin:10px 0}
      .pr-wrap details.model>summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;padding:7px 12px;font-weight:600;font-size:13px;color:#333}
      .pr-wrap details.model>summary::-webkit-details-marker{display:none}
      .pr-wrap details.model>summary:before{content:"▸";color:#0b3d91;font-size:11px}
      .pr-wrap details.model[open]>summary:before{content:"▾"}
      .pr-wrap .midx{background:#eef2fb;color:#0b3d91;font-size:10px;font-weight:700;border-radius:5px;padding:1px 6px;font-variant-numeric:tabular-nums}
      .pr-wrap .model-name{font-family:ui-monospace,Menlo,monospace}
      .pr-wrap .model-body{padding:2px 12px 12px 14px}

      .pr-wrap table.kv{border-collapse:collapse;width:100%;font-size:13px;margin:2px 0}
      .pr-wrap table.kv th{text-align:left;vertical-align:top;color:#555;font-weight:500;padding:3px 12px 3px 0;white-space:nowrap;width:1%}
      .pr-wrap table.kv td{padding:3px 0;vertical-align:top}
      .pr-wrap table.kv table.kv{border-left:1px solid #eee;padding-left:10px}
      .pr-wrap ul.lst{margin:2px 0;padding-left:18px}
      .pr-wrap .series{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#444}
      .pr-wrap .muted{color:#aaa;font-style:italic}
      .pr-wrap table.summary{border-collapse:collapse;width:100%;font-size:13px;margin:0 0 22px}
      .pr-wrap table.summary th{background:#0b3d91;color:#fff;text-align:left;padding:8px 12px}
      .pr-wrap table.summary td{padding:7px 12px;border-bottom:1px solid #eee;vertical-align:top}
      .pr-wrap table.summary td.num{text-align:right;font-variant-numeric:tabular-nums}
      .pr-wrap table.summary td.mnames,.pr-wrap table.summary td.cmodels{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#234}
      .pr-wrap .ds-link{color:#0b3d91;text-decoration:none}
      .pr-wrap .ds-link:hover{text-decoration:underline}
    </style>
    """
    return style + '<div class="pr-wrap">' + toolbar + summary + "".join(cards) + script + "</div>"


def show_payload_report(payload_results, display_fn=None):
    """Build the report and render it.

    Parameters
    ----------
    payload_results : list
        The list of payload objects (each must expose .to_json()).
    display_fn : callable, optional
        Pass the notebook's `displayHTML`. If omitted, the HTML string is
        returned instead of rendered.

    Returns
    -------
    None if rendered via display_fn, else the HTML string.
    """
    html = payloads_to_html(payload_results)
    if display_fn is not None:
        display_fn(html)
        return None
    return html