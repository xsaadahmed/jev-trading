"""Build HTML + Markdown experiment reports from a manifest only (Phase 11).

No model recomputation: all numbers/plots come from the manifest payload and
optional sibling artifact files referenced by relative paths.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from jevlab.evaluation.metrics import power_statement
from jevlab.reporting.conclusion import ConclusionResult, generate_conclusion
from jevlab.reporting.plots import (
    equity_drawdown_figure,
    placebo_histogram_figure,
    reliability_figure,
    trade_distribution_figure,
)
from jevlab.reporting.statements import (
    CONTAMINATION_LABEL,
    HISTORICAL_CONTAMINATION,
    HISTORICAL_PASS_CONTAMINATION_CAVEAT,
    LIVE_EXECUTION_EXPERIMENT,
    WARNING_SIGNAL_NOT_PROOF,
    assert_mandatory_present,
)
from jevlab.utils.hashing import sha256_of

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


@dataclass
class ReportSection:
    id: str
    title: str
    markdown: str = ""
    html: str = ""
    missing: str | None = None


@dataclass
class BuiltReport:
    experiment_id: str
    conclusion: ConclusionResult
    sections: list[ReportSection] = field(default_factory=list)
    markdown: str = ""
    html: str = ""
    manifest_digest: str = ""

    def write(self, out_dir: Path) -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "report.md"
        html_path = out_dir / "report.html"
        md_path.write_text(self.markdown, encoding="utf-8")
        html_path.write_text(self.html, encoding="utf-8")
        meta = {
            "experiment_id": self.experiment_id,
            "conclusion": self.conclusion.to_dict(),
            "manifest_digest": self.manifest_digest,
            "contamination_label": CONTAMINATION_LABEL,
        }
        meta_path = out_dir / "report_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return {"markdown": md_path, "html": html_path, "meta": meta_path}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(manifest: Path | Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        return dict(manifest)
    path = Path(manifest)
    data = _load_json(path)
    if not isinstance(data, dict):
        raise TypeError("manifest must be a JSON object")
    data.setdefault("_manifest_path", str(path.resolve()))
    return data


def _resolve_ref(manifest: Mapping[str, Any], key: str) -> Any | None:
    """Return embedded ``manifest[key]`` or load ``manifest[key + '_path']``."""
    if key in manifest and manifest[key] is not None:
        return manifest[key]
    path_key = f"{key}_path"
    if path_key not in manifest or not manifest[path_key]:
        return None
    base = Path(str(manifest.get("_manifest_path", "."))).parent
    ref = Path(str(manifest[path_key]))
    path = ref if ref.is_absolute() else (base / ref)
    if not path.is_file():
        return None
    return _load_json(path)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _section_strategy(manifest: Mapping[str, Any]) -> ReportSection:
    strat = manifest.get("strategy") or {}
    if not strat:
        return ReportSection(
            "strategy",
            "Strategy definition",
            missing="Strategy block not present in manifest.",
            markdown="_Strategy block not present in manifest._",
            html="<p class='missing'>Strategy block not present in manifest.</p>",
        )
    rows = [
        [str(k), json.dumps(v, default=str) if not isinstance(v, str) else v]
        for k, v in strat.items()
    ]
    md = _md_table(["field", "value"], rows)
    html = md.replace("\n", "<br/>")  # simple; full table below
    html_rows = "".join(f"<tr><th>{k}</th><td><code>{v}</code></td></tr>" for k, v in rows)
    html = f"<table>{html_rows}</table>"
    return ReportSection("strategy", "Strategy definition", markdown=md, html=html)


def _section_dataset(manifest: Mapping[str, Any]) -> ReportSection:
    ds = manifest.get("dataset") or manifest.get("splits") or {}
    if not ds:
        return ReportSection(
            "dataset",
            "Dataset and splits",
            missing="Dataset/splits not present in manifest.",
            markdown="_Dataset/splits not present in manifest._",
            html="<p class='missing'>Dataset/splits not present in manifest.</p>",
        )
    lines = [
        "Partitions and contamination status:",
        "",
    ]
    html_parts = ["<table><tr><th>partition</th><th>range</th><th>contamination</th></tr>"]
    partitions = ds.get("partitions") or []
    if partitions:
        rows: list[list[str]] = []
        for p in partitions:
            name = str(p.get("name", ""))
            rng = str(p.get("range", p.get("dates", "")))
            cont = str(p.get("contamination_status", CONTAMINATION_LABEL))
            rows.append([name, rng, cont])
            html_parts.append(f"<tr><td>{name}</td><td>{rng}</td><td><code>{cont}</code></td></tr>")
        lines.append(_md_table(["partition", "range", "contamination"], rows))
    else:
        for key in ("walk_forward", "sealed", "interim", "forward", "train"):
            if key in ds:
                cont = (
                    "clean_forward"
                    if key == "forward"
                    else ("sealed_held_out" if key == "sealed" else CONTAMINATION_LABEL)
                )
                lines.append(f"- **{key}**: `{ds[key]}` - contamination: `{cont}`")
                html_parts.append(
                    f"<tr><td>{key}</td><td><code>{ds[key]}</code></td>"
                    f"<td><code>{cont}</code></td></tr>"
                )
    html_parts.append("</table>")
    note = ds.get("note") or (
        "Sealed/interim/historical Jev-involving partitions are potentially training-contaminated; "
        "only forward sessions after deployment_timestamp are clean."
    )
    lines.extend(["", note])
    return ReportSection(
        "dataset",
        "Dataset and splits",
        markdown="\n".join(lines),
        html="".join(html_parts) + f"<p>{note}</p>",
    )


def _section_features(manifest: Mapping[str, Any]) -> ReportSection:
    feats = manifest.get("features") or {}
    cols_a = feats.get("model_a_columns") or feats.get("MODEL_A_COLUMNS") or []
    cols_jev = feats.get("jev_feature_columns") or feats.get("JEV_FEATURE_COLUMNS") or []
    if not cols_a and not cols_jev:
        return ReportSection(
            "features",
            "Feature definitions",
            missing="Feature definitions not present in manifest.",
            markdown="_Feature definitions not present in manifest._",
            html="<p class='missing'>Feature definitions not present in manifest.</p>",
        )
    md = [
        f"Model A columns ({len(cols_a)}):",
        "",
        ", ".join(f"`{c}`" for c in cols_a) or "_(none)_",
        "",
        f"Jev feature columns ({len(cols_jev)}):",
        "",
        ", ".join(f"`{c}`" for c in cols_jev) or "_(none)_",
    ]
    html = (
        f"<p><strong>Model A</strong> ({len(cols_a)}): "
        + ", ".join(f"<code>{c}</code>" for c in cols_a)
        + "</p>"
        + f"<p><strong>Jev features</strong> ({len(cols_jev)}): "
        + ", ".join(f"<code>{c}</code>" for c in cols_jev)
        + "</p>"
    )
    return ReportSection("features", "Feature definitions", markdown="\n".join(md), html=html)


def _section_qset(manifest: Mapping[str, Any]) -> ReportSection:
    qset = _resolve_ref(manifest, "qset") or manifest.get("jev_questions") or {}
    if not qset:
        return ReportSection(
            "qset",
            "Jev question definitions",
            missing="Qset / Jev questions not present in manifest.",
            markdown="_Qset / Jev questions not present in manifest._",
            html="<p class='missing'>Qset / Jev questions not present in manifest.</p>",
        )
    qhash = str(
        qset.get("content_hash") or qset.get("qset_hash") or manifest.get("qset_hash") or ""
    )
    version = str(qset.get("qset_version") or "")
    questions = qset.get("questions") or {}
    lines = [f"qset_version: `{version}`", f"qset_hash: `{qhash}`", ""]
    html_parts = [
        f"<p>qset_version: <code>{version}</code><br/>qset_hash: <code>{qhash}</code></p>"
    ]
    for qid, body in questions.items():
        text = body.get("instructions") if isinstance(body, dict) else str(body)
        lines.append(f"### {qid}")
        lines.append("")
        lines.append(str(text))
        lines.append("")
        html_parts.append(f"<h3>{qid}</h3><p>{text}</p>")
    return ReportSection(
        "qset",
        "Jev question definitions",
        markdown="\n".join(lines),
        html="".join(html_parts),
    )


def _section_equity(manifest: Mapping[str, Any]) -> ReportSection:
    equity = _resolve_ref(manifest, "equity") or manifest.get("equity_curve")
    fig = equity_drawdown_figure(equity if isinstance(equity, list) else None)
    if equity is None:
        return ReportSection(
            "equity",
            "Equity and drawdown curves",
            missing="Equity curve not present in manifest.",
            markdown="_Equity curve not present in manifest._",
            html=fig,
        )
    return ReportSection(
        "equity",
        "Equity and drawdown curves",
        markdown="_See HTML report for equity/drawdown figure._",
        html=fig,
    )


def _section_monthly(manifest: Mapping[str, Any]) -> ReportSection:
    monthly = manifest.get("monthly_returns") or []
    if not monthly:
        return ReportSection(
            "monthly",
            "Monthly return table",
            missing="Monthly returns not present in manifest.",
            markdown="_Monthly returns not present in manifest._",
            html="<p class='missing'>Monthly returns not present in manifest.</p>",
        )
    headers = list(monthly[0].keys())
    rows = [[str(r.get(h, "")) for h in headers] for r in monthly]
    md = _md_table(headers, rows)
    html_rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    html = (
        "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>" + html_rows + "</table>"
    )
    return ReportSection("monthly", "Monthly return table", markdown=md, html=html)


def _section_trades(manifest: Mapping[str, Any]) -> ReportSection:
    trades = manifest.get("trade_pnls") or manifest.get("trades")
    pnls: list[float] | None = None
    if isinstance(trades, list) and trades:
        if isinstance(trades[0], (int, float)):
            pnls = [float(x) for x in trades]
        elif isinstance(trades[0], dict):
            key = "pnl_usd" if "pnl_usd" in trades[0] else "pnl"
            pnls = [float(t[key]) for t in trades if key in t]
    fig = trade_distribution_figure(pnls)
    if pnls is None:
        return ReportSection(
            "trades",
            "Trade distribution",
            missing="Trade distribution not present in manifest.",
            markdown="_Trade distribution not present in manifest._",
            html=fig,
        )
    return ReportSection(
        "trades",
        "Trade distribution",
        markdown=f"n_trades={len(pnls)} (see HTML histogram).",
        html=fig,
    )


def _section_comparison(manifest: Mapping[str, Any], *, key: str, title: str) -> ReportSection:
    data = manifest.get(key) or {}
    if not data:
        return ReportSection(
            key,
            title,
            missing=f"{title} not present in manifest.",
            markdown=f"_{title} not present in manifest._",
            html=f"<p class='missing'>{title} not present in manifest.</p>",
        )
    if isinstance(data, list):
        headers = list(data[0].keys()) if data else ["metric"]
        rows = [[str(r.get(h, "")) for h in headers] for r in data]
        md = _md_table(headers, rows)
        html = f"<pre>{json.dumps(data, indent=2, default=str)}</pre>"
    else:
        rows = [[str(k), str(v)] for k, v in data.items()]
        md = _md_table(["metric", "value"], rows)
        html = f"<pre>{json.dumps(data, indent=2, default=str)}</pre>"
    return ReportSection(key, title, markdown=md, html=html)


def _section_calibration(manifest: Mapping[str, Any]) -> ReportSection:
    cal = manifest.get("calibration") or manifest.get("reliability")
    fig = reliability_figure(cal if isinstance(cal, dict) else None)
    if cal is None:
        return ReportSection(
            "calibration",
            "Calibration / reliability",
            missing="Calibration/reliability not present in manifest.",
            markdown="_Calibration/reliability not present in manifest._",
            html=fig,
        )
    return ReportSection(
        "calibration",
        "Calibration / reliability",
        markdown="_See HTML reliability diagram._",
        html=fig,
    )


def _section_placebo(manifest: Mapping[str, Any]) -> ReportSection:
    placebo = _resolve_ref(manifest, "placebo") or {}
    if not placebo:
        return ReportSection(
            "placebo",
            "Placebo histograms",
            missing="Placebo results not present in manifest.",
            markdown="_Placebo results not present in manifest._",
            html="<p class='missing'>Placebo results not present in manifest.</p>",
        )
    parts_md: list[str] = []
    parts_html: list[str] = []
    for pid in ("P1", "P2", "P3", "P4"):
        block = placebo.get(pid)
        if not isinstance(block, dict):
            continue
        samples = block.get("placebo_delta_log_loss_gains") or block.get("samples")
        observed = block.get("observed_delta_log_loss_gain")
        if observed is None:
            observed = block.get("observed")
        pval = block.get("pvalue_log_loss")
        parts_md.append(
            f"- **{pid}**: observed={observed}, p={pval}, n={block.get('n', len(samples or []))}"
        )
        parts_html.append(f"<h3>{pid}</h3>")
        parts_html.append(
            placebo_histogram_figure(
                samples if isinstance(samples, list) else None,
                float(observed) if observed is not None else None,
                title=f"{pid} placebo ΔlogLoss gain",
            )
        )
    if not parts_md:
        parts_md.append("_No P1-P4 blocks with samples._")
        parts_html.append("<p class='missing'>No P1-P4 blocks with samples.</p>")
    return ReportSection(
        "placebo",
        "Placebo histograms",
        markdown="\n".join(parts_md),
        html="".join(parts_html),
    )


def _section_cost(manifest: Mapping[str, Any]) -> ReportSection:
    return _section_comparison(manifest, key="cost_sensitivity", title="Cost sensitivity")


def _section_robustness(manifest: Mapping[str, Any]) -> ReportSection:
    rob = _resolve_ref(manifest, "robustness") or manifest.get("robustness_summary")
    if rob is None:
        return ReportSection(
            "robustness",
            "Robustness",
            missing="Robustness data not present in manifest.",
            markdown="_Robustness data not present in manifest._",
            html="<p class='missing'>Robustness data not present in manifest.</p>",
        )
    # Data only - no narrative conclusions.
    summary = {
        k: (len(v) if isinstance(v, list) else v)
        for k, v in (rob.items() if isinstance(rob, dict) else [])
        if k != "narrative_conclusions"
    }
    md = "Robustness artifact present (data only; no narrative conclusions).\n\n"
    md += f"```json\n{json.dumps(summary, indent=2, default=str)[:4000]}\n```"
    html = f"<pre>{json.dumps(summary, indent=2, default=str)[:8000]}</pre>"
    return ReportSection("robustness", "Robustness", markdown=md, html=html)


def _section_stats(manifest: Mapping[str, Any]) -> ReportSection:
    stats = _resolve_ref(manifest, "stats_summary") or manifest.get("statistical_uncertainty") or {}
    power = manifest.get("power") or (stats.get("mde") if isinstance(stats, dict) else None) or {}
    blurb = power_statement()
    if isinstance(power, dict) and power:
        blurb = str(power.get("statement") or blurb)
    lines = [
        "## Statistical uncertainty",
        "",
        blurb,
        "",
    ]
    if stats:
        lines.append("```json")
        lines.append(json.dumps(stats, indent=2, default=str)[:5000])
        lines.append("```")
    else:
        lines.append("_stats_summary not present; power statement still required._")
    stats_json = json.dumps(stats, indent=2, default=str)[:8000] if stats else ""
    html = f"<p>{blurb}</p><pre>{stats_json}</pre>"
    return ReportSection(
        "stats",
        "Statistical uncertainty (incl. power statement)",
        markdown="\n".join(lines),
        html=html,
        missing=None if stats else "stats_summary optional; power statement included.",
    )


def _section_contamination(manifest: Mapping[str, Any]) -> ReportSection:
    diag = _resolve_ref(manifest, "contamination_diagnostics") or manifest.get("audit") or {}
    lines = [
        f"Each diagnostic below is labeled **{WARNING_SIGNAL_NOT_PROOF}**.",
        "",
        HISTORICAL_CONTAMINATION,
        "",
    ]
    html_parts = [
        f"<p class='warning'>Each diagnostic below is labeled "
        f"<strong>{WARNING_SIGNAL_NOT_PROOF}</strong>.</p>",
        f"<p>{HISTORICAL_CONTAMINATION}</p>",
    ]
    items = diag.get("sections") if isinstance(diag, dict) else None
    if isinstance(items, list) and items:
        for sec in items:
            name = sec.get("name", "diagnostic") if isinstance(sec, dict) else str(sec)
            lines.append(f"- {name}: *{WARNING_SIGNAL_NOT_PROOF}*")
            html_parts.append(
                f"<div class='warning'><strong>{name}</strong> - {WARNING_SIGNAL_NOT_PROOF}</div>"
            )
    elif isinstance(diag, dict) and diag:
        for name in diag:
            if name in {"sections", "meta"}:
                continue
            lines.append(f"- {name}: *{WARNING_SIGNAL_NOT_PROOF}*")
            html_parts.append(
                f"<div class='warning'><strong>{name}</strong> - {WARNING_SIGNAL_NOT_PROOF}</div>"
            )
    else:
        lines.append("_No contamination-diagnostics artifact attached; banner still applies._")
        html_parts.append(
            "<p class='missing'>No contamination-diagnostics artifact attached; "
            "banner still applies.</p>"
        )
    return ReportSection(
        "contamination",
        "Contamination-diagnostics summary",
        markdown="\n".join(lines),
        html="".join(html_parts),
    )


def _section_gates(manifest: Mapping[str, Any]) -> ReportSection:
    gates = manifest.get("gates") or []
    rows_in: list[Mapping[str, Any]] = []
    if isinstance(gates, list):
        rows_in = [g for g in gates if isinstance(g, Mapping)]
    elif isinstance(gates, Mapping):
        for gid, body in gates.items():
            if isinstance(body, Mapping):
                rows_in.append({"id": gid, **body})
            else:
                rows_in.append(
                    {
                        "id": gid,
                        "passed": body if isinstance(body, bool) else None,
                        "status": (
                            "PASS"
                            if body is True
                            else ("FAIL" if body is False else "NOT EVALUATED")
                        ),
                    }
                )
    if not rows_in:
        return ReportSection(
            "gates",
            "Historical gate table (G1–G10)",
            missing="Gate table not present in manifest.",
            markdown="_Gate table not present in manifest._",
            html="<p class='missing'>Gate table not present in manifest.</p>",
        )
    headers = ["gate", "criterion", "threshold", "observed", "status", "evidence"]
    md_rows: list[list[str]] = []
    html_rows: list[str] = []
    for g in rows_in:
        gid = str(g.get("id") or g.get("gate") or "")
        status = str(g.get("status") or "")
        if not status:
            if g.get("passed") is True:
                status = "PASS"
            elif g.get("passed") is False:
                status = "FAIL"
            else:
                status = "NOT EVALUATED"
        row = [
            gid,
            str(g.get("criterion", "")),
            str(g.get("threshold", "")),
            str(g.get("observed", g.get("detail", ""))),
            status,
            str(g.get("evidence", g.get("artifact", ""))),
        ]
        md_rows.append(row)
        html_rows.append(
            "<tr>"
            + "".join(f"<td>{c}</td>" for c in row)
            + "</tr>"
        )
    note = str(
        manifest.get("gates_note")
        or (
            "Any FAIL among G1–G10 blocks HISTORICAL_RESEARCH_PASS. "
            "G10 NOT EVALUATED means the sealed holdout was intentionally not opened "
            "because upstream mandatory gates already failed under the preregistered protocol."
        )
    )
    md = _md_table(headers, md_rows) + f"\n\n{note}\n"
    html = (
        "<table><tr>"
        + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr>"
        + "".join(html_rows)
        + f"</table><p>{note}</p>"
    )
    return ReportSection("gates", "Historical gate table (G1–G10)", markdown=md, html=html)


def _section_sealed_holdout(manifest: Mapping[str, Any]) -> ReportSection:
    sealed = manifest.get("sealed_holdout") or {}
    if not sealed:
        return ReportSection(
            "sealed_holdout",
            "2024–2025 sealed holdout",
            missing="Sealed-holdout block not present in manifest.",
            markdown="_Sealed-holdout block not present in manifest._",
            html="<p class='missing'>Sealed-holdout block not present in manifest.</p>",
        )
    lines = [
        "**The 2024–2025 sealed holdout remains unopened.**",
        "",
        str(sealed.get("reason", "")),
        "",
        f"- sealed_access_log count: `{sealed.get('sealed_access_count', 'n/a')}`",
        f"- Phase 12 run: `{sealed.get('phase12_run', False)}`",
        f"- 2024–2025 Jev inference present: `{sealed.get('sealed_jev_inference', False)}`",
        f"- HISTORICAL_ACCEPTANCE.json present: `{sealed.get('historical_acceptance_present', False)}`",
    ]
    html = (
        "<div class='warning'><strong>The 2024–2025 sealed holdout remains unopened.</strong>"
        f"<p>{sealed.get('reason', '')}</p>"
        f"<ul><li>sealed_access_log count: <code>{sealed.get('sealed_access_count', 'n/a')}</code></li>"
        f"<li>Phase 12 run: <code>{sealed.get('phase12_run', False)}</code></li>"
        f"<li>2024–2025 Jev inference present: <code>{sealed.get('sealed_jev_inference', False)}</code></li>"
        f"<li>HISTORICAL_ACCEPTANCE.json present: "
        f"<code>{sealed.get('historical_acceptance_present', False)}</code></li></ul></div>"
    )
    return ReportSection(
        "sealed_holdout",
        "2024–2025 sealed holdout",
        markdown="\n".join(lines),
        html=html,
    )


def _section_phase10(manifest: Mapping[str, Any]) -> ReportSection:
    p10 = manifest.get("phase10_interpretation") or {}
    if not p10:
        return ReportSection(
            "phase10",
            "Phase 10 interpretation",
            missing="Phase 10 interpretation not present in manifest.",
            markdown="_Phase 10 interpretation not present in manifest._",
            html="<p class='missing'>Phase 10 interpretation not present in manifest.</p>",
        )
    bullets = p10.get("bullets") or []
    lines = [str(p10.get("summary", "")), ""]
    for b in bullets:
        lines.append(f"- {b}")
    html = "<p>" + str(p10.get("summary", "")) + "</p><ul>"
    html += "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
    return ReportSection(
        "phase10",
        "Phase 10 interpretation",
        markdown="\n".join(lines),
        html=html,
    )


def _section_forward(manifest: Mapping[str, Any]) -> ReportSection:
    fwd = manifest.get("forward_capture") or {}
    if not fwd:
        return ReportSection(
            "forward",
            "Forward capture (Phase 13A)",
            missing="Forward-capture block not present in manifest.",
            markdown="_Forward-capture block not present in manifest._",
            html="<p class='missing'>Forward-capture block not present in manifest.</p>",
        )
    lines = [
        "Existing Phase 13A forward-capture state (inputs only; no outcome scoring):",
        "",
        f"- deployment_timestamp: `{fwd.get('deployment_timestamp', '')}`",
        f"- FORWARD_START_DATE: `{fwd.get('FORWARD_START_DATE', fwd.get('forward_start_date', ''))}`",
        f"- note: {fwd.get('note', 'Forward capture does not grant acceptance status.')}",
    ]
    html = (
        "<p>Existing Phase 13A forward-capture state (inputs only; no outcome scoring).</p><ul>"
        f"<li>deployment_timestamp: <code>{fwd.get('deployment_timestamp', '')}</code></li>"
        f"<li>FORWARD_START_DATE: <code>{fwd.get('FORWARD_START_DATE', fwd.get('forward_start_date', ''))}</code></li>"
        f"<li>{fwd.get('note', 'Forward capture does not grant acceptance status.')}</li></ul>"
    )
    return ReportSection(
        "forward",
        "Forward capture (Phase 13A)",
        markdown="\n".join(lines),
        html=html,
    )


def _section_status_table(manifest: Mapping[str, Any]) -> ReportSection:
    statuses = manifest.get("acceptance_statuses") or {}
    rows_spec = [
        ("HISTORICAL_RESEARCH_PASS", "HISTORICAL_ACCEPTANCE.json"),
        ("TINY_LIVE_READY", "TINY_LIVE_ACCEPTANCE.json"),
        ("SCALE_READY", "SCALE_ACCEPTANCE.json"),
    ]
    md_rows: list[list[str]] = []
    html_rows = []
    for status, default_file in rows_spec:
        block = statuses.get(status) or {}
        present = block.get("present")
        digest = block.get("integrity_digest") or block.get("digest") or ""
        path = block.get("path") or ""
        note = str(block.get("note") or ("YES" if present else "NO"))
        if present is None:
            present = bool(digest) or bool(path)
        md_rows.append(
            [
                status,
                "YES" if present else "NO",
                note,
                str(digest)[:16] + ("…" if digest else ""),
                path or default_file,
            ]
        )
        html_rows.append(
            f"<tr><td>{status}</td><td>{'YES' if present else 'NO'}</td>"
            f"<td>{note}</td>"
            f"<td><code>{digest}</code></td><td><code>{path or default_file}</code></td></tr>"
        )
    md = _md_table(["status", "granted", "note", "integrity_digest", "path"], md_rows)
    html = (
        "<table><tr><th>status</th><th>granted</th><th>note</th>"
        "<th>integrity_digest</th><th>path</th></tr>"
        + "".join(html_rows)
        + "</table>"
        + f"<p>{LIVE_EXECUTION_EXPERIMENT}</p>"
    )
    md += f"\n\n{LIVE_EXECUTION_EXPERIMENT}\n"
    # Caveat when historical pass artifact present
    hist = statuses.get("HISTORICAL_RESEARCH_PASS") or {}
    if hist.get("present") or hist.get("integrity_digest"):
        md += f"\n{HISTORICAL_PASS_CONTAMINATION_CAVEAT}\n"
        html += f"<p>{HISTORICAL_PASS_CONTAMINATION_CAVEAT}</p>"
    return ReportSection(
        "statuses",
        "Three-status table",
        markdown=md,
        html=html,
    )


def build_report_sections(manifest: Mapping[str, Any]) -> list[ReportSection]:
    return [
        _section_strategy(manifest),
        _section_dataset(manifest),
        _section_features(manifest),
        _section_qset(manifest),
        _section_equity(manifest),
        _section_monthly(manifest),
        _section_trades(manifest),
        _section_comparison(manifest, key="baseline_comparison", title="Baseline comparison (Model A)"),
        _section_comparison(manifest, key="jev_comparison", title="Jev comparison (Model B)"),
        _section_calibration(manifest),
        _section_placebo(manifest),
        _section_cost(manifest),
        _section_robustness(manifest),
        _section_phase10(manifest),
        _section_stats(manifest),
        _section_contamination(manifest),
        _section_gates(manifest),
        _section_sealed_holdout(manifest),
        _section_forward(manifest),
        _section_status_table(manifest),
    ]


def render_markdown(
    *,
    experiment_id: str,
    sections: list[ReportSection],
    conclusion: ConclusionResult,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    manifest = manifest or {}
    lines = [
        f"# Experiment report: {experiment_id}",
        "",
        "Built from experiment manifest only (no recomputation).",
        "",
        "## Mandatory statements (§16.4)",
        "",
        HISTORICAL_CONTAMINATION,
        "",
        LIVE_EXECUTION_EXPERIMENT,
        "",
        f"Contamination label: `{CONTAMINATION_LABEL}`",
        "",
    ]
    for sec in sections:
        lines.append(f"## {sec.title}")
        lines.append("")
        if sec.missing:
            lines.append(f"_{sec.missing}_")
            lines.append("")
        if sec.markdown:
            lines.append(sec.markdown)
            lines.append("")
    scope = str(
        manifest.get("conclusion_scope")
        or (
            "This conclusion applies to the frozen study (universe, qset, state representation, "
            "model setup, target, and evaluation protocol). It means the preregistered study did "
            "not demonstrate incremental predictive or economic value from the frozen Jev features "
            "over Model A. It does not mean Jev is universally useless, that no different "
            "preregistered study could ever differ, or that training contamination was proven or "
            "disproven. Any future changed qset, features, target, universe, model, or strategy "
            "must be a NEW preregistered study, not a continuation or tuning of this failed one."
        )
    )
    lines.extend(
        [
            "## Final conclusion",
            "",
            conclusion.text,
            "",
            f"Verdict kind: `{conclusion.kind}`",
            "",
            scope,
            "",
        ]
    )
    if conclusion.reasons:
        lines.append("Reasons: " + ", ".join(conclusion.reasons))
        lines.append("")
    return "\n".join(lines)


def render_html(
    *,
    experiment_id: str,
    sections: list[ReportSection],
    conclusion: ConclusionResult,
    manifest: Mapping[str, Any] | None = None,
) -> str:
    manifest = manifest or {}
    scope = str(
        manifest.get("conclusion_scope")
        or (
            "This conclusion applies to the frozen study (universe, qset, state representation, "
            "model setup, target, and evaluation protocol). It means the preregistered study did "
            "not demonstrate incremental predictive or economic value from the frozen Jev features "
            "over Model A. It does not mean Jev is universally useless, that no different "
            "preregistered study could ever differ, or that training contamination was proven or "
            "disproven. Any future changed qset, features, target, universe, model, or strategy "
            "must be a NEW preregistered study, not a continuation or tuning of this failed one."
        )
    )
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
    )
    tmpl = env.get_template("report.html.j2")
    return tmpl.render(
        title=f"Experiment report: {experiment_id}",
        experiment_id=experiment_id,
        historical_contamination=HISTORICAL_CONTAMINATION,
        live_execution_experiment=LIVE_EXECUTION_EXPERIMENT,
        contamination_label=CONTAMINATION_LABEL,
        sections=[
            {
                "id": s.id,
                "title": s.title,
                "html": s.html,
                "missing": s.missing,
            }
            for s in sections
        ],
        conclusion=conclusion.to_dict(),
        conclusion_scope=scope,
    )


def build_report(
    manifest: Path | Mapping[str, Any] | str,
    *,
    out_dir: Path | None = None,
) -> BuiltReport:
    """Assemble HTML+Markdown report from a manifest (and referenced artifacts)."""
    data = load_manifest(manifest)
    experiment_id = str(data.get("experiment_id") or data.get("id") or "unknown")
    force = data.get("force_verdict")
    force_kind = force if force in {"pass", "no_value", "underpowered"} else None
    conclusion = generate_conclusion(
        gates=data.get("gates"),
        underpowered=data.get("underpowered")
        if isinstance(data.get("underpowered"), bool)
        else None,
        power=data.get("power") if isinstance(data.get("power"), dict) else None,
        force_kind=force_kind,
    )
    sections = build_report_sections(data)
    md = render_markdown(
        experiment_id=experiment_id,
        sections=sections,
        conclusion=conclusion,
        manifest=data,
    )
    html = render_html(
        experiment_id=experiment_id,
        sections=sections,
        conclusion=conclusion,
        manifest=data,
    )
    missing = assert_mandatory_present(md)
    if missing:
        raise RuntimeError(f"report missing mandatory statements: {missing}")
    missing_html = assert_mandatory_present(html)
    if missing_html:
        raise RuntimeError(f"HTML report missing mandatory statements: {missing_html}")

    digest_src = {k: v for k, v in data.items() if not str(k).startswith("_")}
    report = BuiltReport(
        experiment_id=experiment_id,
        conclusion=conclusion,
        sections=sections,
        markdown=md,
        html=html,
        manifest_digest=sha256_of(json.dumps(digest_src, sort_keys=True, default=str)),
    )
    if out_dir is not None:
        report.write(Path(out_dir))
    return report
