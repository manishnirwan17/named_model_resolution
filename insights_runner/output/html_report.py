"""
HTML debug dashboard for InsightsPayload.

Single public function: build_html(d: dict) -> str

Produces a self-contained HTML string (no external dependencies) that renders
an InsightsPayload dict as a human-scannable dashboard:
  - Header card: dataset name, routing, quality gate summary
  - Column classification table: color-coded subtype pills
  - Quality gate cards: PASS/WARN/FAIL badges per model per check
  - Model signals: BOCPD changepoints table + SVG sparkline, MMM fit stats
  - Transform context: collapsed <details> element
  - Warnings card

Usage:
    displayHTML(payload.to_html())                           # Databricks
    from IPython.display import HTML, display
    display(HTML(payload.to_html()))                         # Jupyter
    Path("debug.html").write_text(payload.to_html())        # file
"""

from __future__ import annotations

from html import escape as _esc

# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

_SUBTYPE_COLORS: dict[str, str] = {
    "date":                "#3b82f6",
    "measure":             "#16a34a",
    "channel":             "#ea580c",
    "segment":             "#7c3aed",
    "geography":           "#0891b2",
    "key":                 "#6b7280",
    "flag":                "#ca8a04",
    "unclassified_metric": "#dc2626",
    "dimension_attribute": "#9ca3af",
    "unknown":             "#1f2937",
}

_DECISION_COLORS: dict[str, str] = {
    "PASS": "#16a34a",
    "WARN": "#ca8a04",
    "FAIL": "#dc2626",
}

_SUBTYPE_ORDER = [
    "date", "measure", "channel", "segment", "geography",
    "key", "flag", "unclassified_metric", "dimension_attribute", "unknown",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h(v: object) -> str:
    """Escape and stringify — returns em-dash for None/empty."""
    if v is None or v == "":
        return "&mdash;"
    return _esc(str(v))


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
        f'background:{color};color:#fff;font-size:12px;font-weight:600">'
        f"{_esc(text)}</span>"
    )


def _decision_badge(decision: str | None) -> str:
    if not decision:
        return "&mdash;"
    color = _DECISION_COLORS.get(str(decision), "#6b7280")
    return _badge(str(decision), color)


def _subtype_pill(subtype: str | None) -> str:
    if not subtype:
        return "&mdash;"
    color = _SUBTYPE_COLORS.get(str(subtype), "#6b7280")
    return (
        f'<span style="display:inline-block;padding:1px 7px;border-radius:4px;'
        f'background:{color};color:#fff;font-size:11px;font-weight:600">'
        f"{_esc(str(subtype))}</span>"
    )


def _fmt_pct(v: object) -> str:
    if v is None:
        return "&mdash;"
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return _h(v)


def _fmt_f(v: object, dec: int = 3) -> str:
    if v is None:
        return "&mdash;"
    try:
        return f"{float(v):.{dec}f}"
    except (TypeError, ValueError):
        return _h(v)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 0 }
body { background: #f9fafb; padding: 24px; color: #111827 }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px }
h1 { font-size: 22px; font-weight: 700 }
h2 { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #374151 }
h3 { font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #374151 }
table { border-collapse: collapse; width: 100% }
td, th { padding: 6px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: middle }
th { background: #f3f4f6; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #6b7280; font-weight: 600 }
tr:last-child td { border-bottom: none }
.warn-text { background: #fffbeb; border-left: 3px solid #f59e0b; padding: 8px 12px; border-radius: 4px; font-size: 13px; white-space: pre-wrap; margin-top: 6px }
details summary { cursor: pointer; font-size: 14px; font-weight: 600; color: #374151; padding: 4px 0 }
details[open] summary { margin-bottom: 10px }
.mono { font-family: ui-monospace, 'Cascadia Code', monospace; font-size: 12px }
"""


# ---------------------------------------------------------------------------
# Section: Header
# ---------------------------------------------------------------------------

def _section_header(d: dict) -> str:
    name = _h(d.get("dataset_name", "Unknown dataset"))
    generated_at = _h(d.get("generated_at", ""))

    meta = d.get("metadata") or {}
    routing = meta.get("routing") or {}
    top_model = _h(routing.get("top_model"))
    conf = routing.get("confidence")
    conf_str = _fmt_pct(conf) if conf is not None else "&mdash;"

    qg = d.get("quality_gate") or {}
    overall = qg.get("overall_decision", "")
    gate_badge = _decision_badge(overall)

    candidates = routing.get("all_candidates") or []
    cand_parts = [
        f'{_h(c.get("model"))} {_fmt_pct(c.get("confidence"))}'
        for c in candidates
    ]
    cand_str = " &nbsp;|&nbsp; ".join(cand_parts) if cand_parts else "&mdash;"

    kb = d.get("knowledge_base_context") or {}
    matched = kb.get("matched_datamart")
    desc = kb.get("description")
    kb_html = ""
    if matched:
        kb_html = (
            f'<div style="margin-top:8px;font-size:13px">'
            f"KB match: <b>{_h(matched)}</b>"
            + (f" &mdash; {_h(desc)}" if desc else "")
            + "</div>"
        )

    star = meta.get("star_schema") or {}
    dim_tables = star.get("dimension_tables") or []
    star_html = ""
    if dim_tables:
        star_html = (
            f'<div style="margin-top:6px;font-size:12px;color:#6b7280">'
            f'Star schema dims: {_esc(", ".join(str(t) for t in dim_tables))}</div>'
        )

    table_type = _h(meta.get("table_type", ""))

    return f"""
<div class="card">
  <h1>{name}</h1>
  <div style="display:flex;gap:16px;margin-top:8px;flex-wrap:wrap;align-items:center">
    <span>Table type: <b>{table_type}</b></span>
    <span>Top model: <b>{top_model}</b> ({conf_str})</span>
    <span>Quality gate: {gate_badge}</span>
    <span style="font-size:12px;color:#9ca3af">{generated_at}</span>
  </div>
  <div style="margin-top:8px;font-size:13px;color:#6b7280">Routing candidates: {cand_str}</div>
  {kb_html}
  {star_html}
</div>
"""


# ---------------------------------------------------------------------------
# Section: Column classification
# ---------------------------------------------------------------------------

def _section_columns(d: dict) -> str:
    cols = d.get("columns") or []
    if not cols:
        return ""

    order_map = {s: i for i, s in enumerate(_SUBTYPE_ORDER)}
    cols_sorted = sorted(
        cols,
        key=lambda c: order_map.get(c.get("subtype", "unknown"), len(_SUBTYPE_ORDER)),
    )

    rows = []
    for c in cols_sorted:
        name = _h(c.get("name"))
        dtype = _h(c.get("dtype"))
        subtype = c.get("subtype", "unknown")
        pill = _subtype_pill(subtype)

        conf = c.get("confidence")
        conf_color = "#d97706" if conf is not None and float(conf) < 0.8 else "#111827"
        conf_str = (
            f'<span style="color:{conf_color}">{_fmt_f(conf, 2)}</span>'
            if conf is not None else "&mdash;"
        )

        ms = c.get("match_source", "")
        ms_color = "#dc2626" if ms == "guardrail_metric" else "#374151"
        ms_str = f'<span style="color:{ms_color}">{_h(ms)}</span>'

        p = c.get("profile") or {}
        null_pct = _fmt_pct(p.get("null_pct"))
        unique = _h(p.get("unique_count"))
        grain = _h(p.get("date_grain"))
        skew = _fmt_f(p.get("skewness"), 2)

        rows.append(
            f"<tr>"
            f"<td class='mono'>{name}</td>"
            f"<td>{dtype}</td>"
            f"<td>{pill}</td>"
            f"<td>{conf_str}</td>"
            f"<td>{ms_str}</td>"
            f"<td>{null_pct}</td>"
            f"<td>{unique}</td>"
            f"<td>{grain}</td>"
            f"<td>{skew}</td>"
            f"</tr>"
        )

    return f"""
<div class="card">
  <h2>Column Classification ({len(cols)} columns)</h2>
  <table>
    <thead><tr>
      <th>Column</th><th>dtype</th><th>Subtype</th>
      <th>Conf</th><th>Match source</th>
      <th>Null %</th><th>Unique #</th><th>Grain</th><th>Skewness</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>
"""


# ---------------------------------------------------------------------------
# Section: Quality gate
# ---------------------------------------------------------------------------

def _section_quality_gate(d: dict) -> str:
    qg = d.get("quality_gate") or {}
    per_model = qg.get("per_model") or {}
    if not per_model:
        return ""

    parts = []
    for model_name, report in per_model.items():
        decision = report.get("decision", "")
        badge = _decision_badge(decision)
        skip_reason = report.get("skip_reason")
        checks = report.get("checks") or {}

        if skip_reason:
            body = f'<div class="warn-text">{_h(skip_reason)}</div>'
        elif checks:
            rows = []
            for check_name, res in checks.items():
                status = res.get("status", "")
                c_badge = _decision_badge(status)
                detail = _h(res.get("detail"))
                metric = res.get("metric")
                metric_str = _fmt_f(metric, 4) if metric is not None else "&mdash;"
                rows.append(
                    f"<tr>"
                    f"<td class='mono'>{_h(check_name)}</td>"
                    f"<td>{c_badge}</td>"
                    f"<td style='font-size:12px;color:#374151'>{detail}</td>"
                    f"<td class='mono'>{metric_str}</td>"
                    f"</tr>"
                )
            body = (
                "<table><thead><tr>"
                "<th>Check</th><th>Status</th><th>Detail</th><th>Metric</th>"
                f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
            )
        else:
            body = '<p style="font-size:13px;color:#6b7280">No checks recorded.</p>'

        parts.append(
            f'<div style="margin-bottom:16px">'
            f"<h3>{_h(model_name)} {badge}</h3>"
            f"{body}"
            f"</div>"
        )

    return f"""
<div class="card">
  <h2>Quality Gate</h2>
  {"".join(parts)}
</div>
"""


# ---------------------------------------------------------------------------
# Model signals helpers
# ---------------------------------------------------------------------------

def _get_measure_key(series: list[dict]) -> str | None:
    """Find the raw (non-log) measure column key in a cp_context_window series row."""
    if not series:
        return None
    for key in series[0]:
        if key in ("date", "cp_prob", "exp_run_length"):
            continue
        if key.startswith("log_"):
            continue
        return key
    return None


def _bocpd_sparkline(cp_probs_series: list[dict], cp_candidates: list[dict] | None) -> str:
    """Inline SVG sparkline of cp_prob over time with vertical CP markers."""
    if not cp_probs_series or len(cp_probs_series) < 2:
        return ""

    probs = [float(e.get("cp_prob", 0)) for e in cp_probs_series]
    dates = [str(e.get("date", "")) for e in cp_probs_series]
    n = len(probs)

    W, H = 600, 60
    PL, PR, PT, PB = 30, 10, 5, 18
    CW = W - PL - PR
    CH = H - PT - PB
    max_p = max(probs) if max(probs) > 0 else 1.0

    def xi(i: int) -> float:
        return PL + i * CW / (n - 1)

    def yi(p: float) -> float:
        return PT + CH - (p / max_p) * CH

    points = " ".join(f"{xi(i):.1f},{yi(p):.1f}" for i, p in enumerate(probs))

    cp_indices: set[int] = set()
    for cp in (cp_candidates or []):
        idx = cp.get("week_idx")
        if idx is not None:
            try:
                cp_indices.add(int(idx))
            except (TypeError, ValueError):
                pass

    markers = "".join(
        f'<line x1="{xi(idx):.1f}" y1="{PT}" x2="{xi(idx):.1f}" y2="{PT + CH}"'
        f' stroke="#dc2626" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.8"/>'
        for idx in sorted(cp_indices)
        if 0 <= idx < n
    )

    step = max(1, n // 7)
    labels = "".join(
        f'<text x="{xi(i):.1f}" y="{H - 2}" font-size="9" text-anchor="middle" fill="#9ca3af">'
        f"{_esc(dates[i][:7])}</text>"
        for i in range(0, n, step)
    )

    y_label = (
        f'<text x="4" y="{PT + CH // 2}" font-size="9" fill="#9ca3af" dominant-baseline="middle">CP</text>'
    )

    return (
        f'<svg width="{W}" height="{H}" style="display:block;margin-top:8px;overflow:visible">'
        f'<rect x="{PL}" y="{PT}" width="{CW}" height="{CH}" fill="#f9fafb" stroke="#e5e7eb" rx="2"/>'
        f"{markers}"
        f'<polyline points="{points}" fill="none" stroke="#3b82f6" stroke-width="1.5"/>'
        f"{labels}"
        f"{y_label}"
        f"</svg>"
    )


def _bocpd_signals_html(signals: dict) -> str:
    n_cp = signals.get("n_changepoints", 0)
    cp_windows = signals.get("cp_context_windows") or []
    cp_candidates = signals.get("cp_candidates") or []
    cp_probs_series = signals.get("cp_probs_series") or []

    # Determine measure key from first window's series
    mkey = None
    if cp_windows:
        mkey = _get_measure_key(cp_windows[0].get("series") or [])
    mkey_label = mkey or "measure"

    # Build changepoints table
    cp_rows = []
    for win in cp_windows:
        cp_date = _h(win.get("changepoint_date"))
        cp_prob = win.get("cp_prob", 0.0)
        bar_w = max(0, min(100, int(float(cp_prob) * 100)))
        prob_bar = (
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="width:60px;height:8px;background:#e5e7eb;border-radius:4px">'
            f'<div style="width:{bar_w}%;height:8px;background:#3b82f6;border-radius:4px"></div>'
            f"</div><span>{_fmt_pct(cp_prob)}</span></div>"
        )
        series = win.get("series") or []
        if mkey and len(series) >= 2:
            before_val = series[0].get(mkey)
            after_val = series[-1].get(mkey)
            before_str = _fmt_f(before_val, 1)
            after_str = _fmt_f(after_val, 1)
            if before_val is not None and after_val is not None:
                delta = float(after_val) - float(before_val)
                delta_color = "#16a34a" if delta >= 0 else "#dc2626"
                delta_str = (
                    f'<span style="color:{delta_color}">{_fmt_f(delta, 1)}</span>'
                )
            else:
                delta_str = "&mdash;"
        else:
            before_str = after_str = delta_str = "&mdash;"

        cp_rows.append(
            f"<tr>"
            f"<td class='mono'>{cp_date}</td>"
            f"<td>{prob_bar}</td>"
            f"<td class='mono'>{before_str}</td>"
            f"<td class='mono'>{after_str}</td>"
            f"<td class='mono'>{delta_str}</td>"
            f"</tr>"
        )

    cp_table = ""
    if cp_rows:
        cp_table = (
            f"<table><thead><tr>"
            f"<th>Date</th><th>CP Prob</th>"
            f"<th>{_h(mkey_label)} (before)</th>"
            f"<th>{_h(mkey_label)} (after)</th>"
            f"<th>Delta</th>"
            f"</tr></thead><tbody>{''.join(cp_rows)}</tbody></table>"
        )

    sparkline = _bocpd_sparkline(cp_probs_series, cp_candidates)
    sparkline_html = ""
    if sparkline:
        sparkline_html = (
            f'<div style="margin-top:14px">'
            f'<p style="font-size:12px;color:#6b7280">CP probability series ({len(cp_probs_series)} periods)</p>'
            f"{sparkline}"
            f"</div>"
        )

    return (
        f'<p style="font-size:13px;margin-bottom:10px"><b>{n_cp} changepoint(s)</b> detected</p>'
        + (cp_table if cp_table else '<p style="font-size:13px;color:#6b7280">No changepoint windows available.</p>')
        + sparkline_html
    )


def _mmm_signals_html(signals: dict) -> str:
    fit = signals.get("model_fit") or {}
    rows = []
    if fit:
        mape = fit.get("in_sample_mape")
        rhat = fit.get("rhat_max")
        if mape is not None:
            rows.append(f"<tr><td>In-sample MAPE</td><td class='mono'>{_fmt_pct(mape)}</td></tr>")
        if rhat is not None:
            rhat_f = float(rhat)
            rhat_color = "#16a34a" if rhat_f <= 1.01 else "#dc2626"
            rows.append(
                f"<tr><td>R-hat max</td>"
                f"<td class='mono' style='color:{rhat_color}'>{_fmt_f(rhat, 4)}</td></tr>"
            )

    # Channel contributions
    contributions = signals.get("channel_contributions") or signals.get("contributions")
    if contributions and isinstance(contributions, dict):
        for ch, v in contributions.items():
            rows.append(f"<tr><td>Contribution: {_h(ch)}</td><td class='mono'>{_fmt_pct(v)}</td></tr>")

    if not rows:
        return '<p style="font-size:13px;color:#6b7280">No model fit statistics available.</p>'

    return f"<table><tbody>{''.join(rows)}</tbody></table>"


def _generic_signals_html(signals: dict) -> str:
    rows = []
    for k, v in signals.items():
        if isinstance(v, (dict, list)):
            v_str = f'<span class="mono" style="font-size:11px">{_esc(str(v)[:300])}</span>'
        else:
            v_str = _h(v)
        rows.append(f"<tr><td>{_h(k)}</td><td>{v_str}</td></tr>")
    if not rows:
        return ""
    return f"<table><tbody>{''.join(rows)}</tbody></table>"


def _candidate_window_html(cw: dict) -> str:
    years = _h(cw.get("years"))
    cutoff = _h(cw.get("cutoff_date"))
    max_d = _h(cw.get("max_date"))
    n_rows = _h(cw.get("n_rows"))
    return (
        f'<p style="font-size:12px;color:#6b7280;margin-bottom:10px">'
        f"Window: <b>{years}yr</b> &nbsp; {cutoff} &rarr; {max_d} &nbsp; ({n_rows} rows)</p>"
    )


# ---------------------------------------------------------------------------
# Section: Model signals
# ---------------------------------------------------------------------------

def _section_model_signals(d: dict) -> str:
    model_signals = d.get("model_signals") or {}
    ran_models = {m: sig for m, sig in model_signals.items() if sig.get("ran")}
    if not ran_models:
        return ""

    parts = []
    for model_name, sig in ran_models.items():
        cw = sig.get("candidate_window")
        cw_html = _candidate_window_html(cw) if cw else ""
        signals = sig.get("signals") or {}
        quality_decision = sig.get("quality_decision", "")
        qd_badge = _decision_badge(quality_decision)

        if model_name == "BOCPD":
            body = _bocpd_signals_html(signals)
        elif model_name == "MMM":
            body = _mmm_signals_html(signals)
        else:
            body = (
                _generic_signals_html(signals) if signals
                else '<p style="font-size:13px;color:#6b7280">No signals.</p>'
            )

        note = sig.get("note")
        note_html = (
            f'<p class="warn-text" style="margin-top:10px">{_h(note)}</p>'
            if note else ""
        )

        parts.append(
            f'<div style="margin-bottom:20px">'
            f"<h3>{_h(model_name)} {qd_badge}</h3>"
            f"{cw_html}"
            f"{body}"
            f"{note_html}"
            f"</div>"
        )

    return f"""
<div class="card">
  <h2>Model Signals</h2>
  {"".join(parts)}
</div>
"""


# ---------------------------------------------------------------------------
# Section: Transform context
# ---------------------------------------------------------------------------

def _section_transform_context(d: dict) -> str:
    transform_context = d.get("transform_context") or []
    if not transform_context:
        return ""

    rows = []
    for entry in transform_context:
        col = _h(entry.get("column"))
        suggestions = entry.get("suggestions") or []
        sug_str = _esc(", ".join(str(s) for s in suggestions))
        rows.append(
            f"<tr><td class='mono'>{col}</td>"
            f"<td style='font-size:12px'>{sug_str}</td></tr>"
        )

    return f"""
<div class="card">
  <details>
    <summary>Transform Context ({len(transform_context)} columns with suggestions)</summary>
    <table>
      <thead><tr><th>Column</th><th>Suggested Transforms</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </details>
</div>
"""


# ---------------------------------------------------------------------------
# Section: Warnings
# ---------------------------------------------------------------------------

def _section_warnings(d: dict) -> str:
    warnings = d.get("warnings") or []
    if not warnings:
        return ""

    items = "".join(
        f'<li style="margin-bottom:4px;font-size:13px">{_h(w)}</li>'
        for w in warnings
    )
    return f"""
<div class="card" style="border-left:4px solid #ca8a04">
  <h3>Warnings ({len(warnings)})</h3>
  <ul style="margin-top:8px;padding-left:20px">{items}</ul>
</div>
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_html(d: dict) -> str:
    """
    Convert an InsightsPayload dict (from dataclasses.asdict()) to a
    self-contained HTML string suitable for browser viewing or Databricks
    displayHTML().

    Parameters
    ----------
    d : dict
        The payload dict, typically from ``dataclasses.asdict(payload)``.

    Returns
    -------
    str
        A complete ``<!DOCTYPE html>`` document with inline CSS, no external
        dependencies.
    """
    title = _esc(str(d.get("dataset_name", "InsightsPayload")))
    body = "".join([
        _section_header(d),
        _section_columns(d),
        _section_quality_gate(d),
        _section_model_signals(d),
        _section_transform_context(d),
        _section_warnings(d),
    ])

    return (
        f"<!DOCTYPE html>\n"
        f'<html lang="en">\n'
        f"<head>\n"
        f'<meta charset="UTF-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"<title>InsightsPayload \u2014 {title}</title>\n"
        f"<style>{_CSS}</style>\n"
        f"</head>\n"
        f"<body>\n"
        f"{body}\n"
        f"</body>\n"
        f"</html>"
    )
