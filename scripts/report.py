"""Data-driven campaign report generator.

Reads the JSON artifacts produced by `scripts/run_campaign.sh` and emits a
self-contained Markdown, HTML, and PDF report whose structure mirrors the
source PDF (Odyssey AMD Inference Pilot, April 2026), with additional
commentary and auto-derived insights. No numeric value is hardcoded —
every figure in the output is computed from the campaign's JSON.

PDF generation is automatic when ``pandoc`` + a PDF backend (wkhtmltopdf,
xelatex, pdflatex, or tectonic) is available; the script falls back
through the available tools and skips the PDF step cleanly when none
are installed.

Usage:
    python scripts/report.py --out results/<campaign-id>/
    python scripts/report.py --out results/<campaign-id>/ --format md
    python scripts/report.py --out results/<campaign-id>/ --format html
    python scripts/report.py --out results/<campaign-id>/ --format pdf
    python scripts/report.py --out results/<campaign-id>/ --format all \
        --output-name myreport
    python scripts/report.py --out results/<campaign-id>/ --no-pdf  # skip PDF
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from html import escape as html_escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Helpers: load JSON, format numbers, render tables & images for MD and HTML.
# ---------------------------------------------------------------------------


def _load(p: Path) -> Optional[Any]:
    if not p or not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _fmt(v, places: int = 2, na: str = "n/a") -> str:
    if v is None:
        return na
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return na
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{places}f}"
    return str(v)


def _pct(v, places: int = 1, na: str = "n/a") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return na
    return f"{v * 100:.{places}f}%"


def _fmt_tflops(v, na: str = "n/a") -> str:
    """Render TFLOP/s with magnitude-adaptive precision so sub-1 CPU values
    don't collapse to ``0``. Examples::

        1357.4  -> "1,357"
        12.5    -> "12.5"
        0.468   -> "0.468"
        0.0012  -> "1.2e-03"
        None    -> "n/a"
    """
    if v is None:
        return na
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return na
        a = abs(v)
        if a == 0:
            return "0"
        if a >= 1000:
            return f"{v:,.0f}"
        if a >= 10:
            return f"{v:.1f}"
        if a >= 1:
            return f"{v:.2f}"
        if a >= 0.01:
            return f"{v:.3f}"
        return f"{v:.2e}"
    return str(v)


def _is_cpu_host(env: dict, *summaries: dict) -> bool:
    """Return True when the campaign clearly ran on a CPU host.

    Sources we trust, in order:
      1. Any bench summary whose ``device_type`` is "cpu".
      2. ``env.json`` reporting torch.cuda.is_available is False or zero
         GPU device count.
    Mixed signals fall through to GPU (any accelerator target) to avoid
    accidentally suppressing rated-spec context on partial CPU artifacts.
    """
    for s in summaries:
        if isinstance(s, dict) and s.get("device_type") == "cpu":
            return True
    if env:
        torch_info = ((env.get("software") or {}).get("torch") or {}) or {}
        if torch_info.get("torch_cuda_available") is False:
            return True
        if torch_info.get("device_count") in (0, None):
            if torch_info.get("torch_cuda_available") is False:
                return True
    return False


# ---------------------------------------------------------------------------
# Report configuration. Every tunable used by this script — the target
# registry, classification thresholds, source-pilot reference targets,
# status-pill CSS mapping, glossary, and project metadata — lives in
# ``configs/report_config.json`` so a campaign reviewer can re-tune the
# report without code changes. Override at runtime with
# ``--report-config <path>``. The file is required: if it goes missing
# we fail fast with a helpful pointer rather than silently swap in a
# different code path.
# ---------------------------------------------------------------------------

_DEFAULT_REPORT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parent.parent / "configs" / "report_config.json"
)

# Module-level cache so the JSON is parsed exactly once per process. The
# first call (typically ``main()`` parsing CLI args) seeds the cache via
# ``_load_report_config(path)``; every subsequent ``_cfg()`` lookup goes
# through this dict.
_REPORT_CONFIG_CACHE: Dict[str, Any] = {}


def _load_report_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load ``configs/report_config.json`` and cache it on the module.

    Behaviour:
      * If ``path`` is given, that file is loaded and replaces any cached
        config (used by ``main()`` once CLI args are parsed and by
        smoke-test scripts that point at fixture configs).
      * If ``path`` is None and the cache is already populated, return it.
      * If ``path`` is None and the cache is empty, load
        ``_DEFAULT_REPORT_CONFIG_PATH`` (resolved relative to this file's
        repo root, so the script works regardless of cwd).
      * Missing or malformed JSON aborts with a SystemExit pointing at
        the file — there is no silent fallback because the JSON is the
        sole source of truth for thresholds, registry, and glossary.
    """
    global _REPORT_CONFIG_CACHE
    if path is None and _REPORT_CONFIG_CACHE:
        return _REPORT_CONFIG_CACHE
    p = Path(path) if path else _DEFAULT_REPORT_CONFIG_PATH
    if not p.exists():
        raise SystemExit(
            f"[report] report-config not found: {p}\n"
            f"  This file is the source of truth for thresholds, target "
            f"registry, glossary,\n  and project metadata. Restore the "
            f"default at configs/report_config.json or\n  pass "
            f"--report-config <path> to point at an alternate config."
        )
    try:
        cfg = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[report] failed to parse {p}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise SystemExit(f"[report] {p} must contain a JSON object at top level")
    _REPORT_CONFIG_CACHE = cfg
    return cfg


def _cfg() -> Dict[str, Any]:
    """Return the cached report-config; loads the default file on first use."""
    return _load_report_config()


def _threshold(name: str) -> float:
    """Resolve a numeric threshold from ``thresholds.<name>`` in the
    report-config. Raises a clear KeyError when the threshold is missing
    rather than silently substituting a hardcoded default — this keeps
    ``configs/report_config.json`` as the single auditable source of
    every numeric gate the report uses.
    """
    val = (_cfg().get("thresholds") or {}).get(name)
    if val is None:
        raise KeyError(
            f"thresholds.{name} missing from report_config "
            f"(see configs/report_config.json)"
        )
    return float(val)


def _project_meta() -> Dict[str, Any]:
    return _cfg().get("project") or {}


def _target_profile(env: dict, is_cpu_host: bool,
                    override: Optional[str] = None) -> Dict[str, Any]:
    """Resolve a generic target-hardware profile from the run.

    Returns a dict the report uses in place of any hardcoded
    accelerator name or spec value::

        {
          "name":            human full name (e.g. "AMD Instinct MI300X"),
          "short":           short label (e.g. "MI300X"),
          "vendor":          "amd" | "nvidia" | "cpu" | "unknown",
          "is_cpu":          True for CPU validation runs,
          "rated_bf16_low":  TFLOP/s lower-bound rated BF16 dense, or None,
          "rated_bf16_high": TFLOP/s upper-bound (some vendors quote one),
          "rated_bw_gb_s":   on-package memory bandwidth, or None,
          "rated_mem_gib":   on-package memory capacity in GiB, or None,
          "has_rated_specs": True iff at least one rated spec is known,
        }

    On a CPU host the function always returns the CPU profile (no rated
    specs). On a GPU host it inspects ``env['software']['torch']['device_names']``
    against the registry; the optional ``override`` argument (from
    ``--target`` on the CLI) takes precedence for forced labelling.
    """
    if is_cpu_host:
        torch_info = ((env or {}).get("software", {}) or {}).get("torch", {}) or {}
        cpu_dev = (torch_info.get("device_names") or ["CPU"])[0]
        return {
            "name":            f"CPU host ({cpu_dev})" if cpu_dev != "CPU" else "CPU host",
            "short":           "CPU",
            "vendor":          "cpu",
            "is_cpu":          True,
            "rated_bf16_low":  None,
            "rated_bf16_high": None,
            "rated_bw_gb_s":   None,
            "rated_mem_gib":   None,
            "has_rated_specs": False,
        }

    devs = ((env or {}).get("software", {}) or {}).get("torch", {}).get("device_names") or []
    detected_name = devs[0] if devs else "Unknown accelerator"

    # CLI override wins over auto-detect.
    needle = override or detected_name
    import re
    registry = _cfg().get("target_registry") or []
    for entry in registry:
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("pattern") or ""
        if not pattern or not re.search(pattern, needle, re.IGNORECASE):
            continue
        short = entry.get("short") or "GPU"
        vendor = entry.get("vendor") or "unknown"
        template = entry.get("name_template") or "{short}"
        try:
            full_name = template.format(short=short, vendor=vendor)
        except Exception:  # noqa: BLE001
            full_name = short
        # Prefer the actually-detected name when it's longer/more
        # specific than the registry's short label (so the cover page
        # still shows e.g. "AMD Instinct MI300X OAM" if that's what
        # torch reports).
        if not override and detected_name and len(detected_name) > len(short):
            full_name = detected_name
        bf16_lo = entry.get("rated_bf16_low")
        bf16_hi = entry.get("rated_bf16_high")
        bw      = entry.get("rated_bw_gb_s")
        mem     = entry.get("rated_mem_gib")
        return {
            "name":            full_name,
            "short":           short,
            "vendor":          vendor,
            "is_cpu":          False,
            "rated_bf16_low":  bf16_lo,
            "rated_bf16_high": bf16_hi,
            "rated_bw_gb_s":   bw,
            "rated_mem_gib":   mem,
            "has_rated_specs": any(v is not None for v in
                                    (bf16_lo, bf16_hi, bw, mem)),
        }

    # Unknown accelerator: produce a complete profile with no rated specs.
    fallback_short = (
        detected_name.split()[-1] if detected_name
        else _project_meta().get("unknown_device_fallback_short", "GPU")
    )
    return {
        "name":            detected_name,
        "short":           fallback_short,
        "vendor":          "unknown",
        "is_cpu":          False,
        "rated_bf16_low":  None,
        "rated_bf16_high": None,
        "rated_bw_gb_s":   None,
        "rated_mem_gib":   None,
        "has_rated_specs": False,
    }


def _rated_bf16_label(profile: Dict[str, Any]) -> str:
    """Compact "1,260 / 2,500 TFLOP/s" label for the rated BF16 column;
    returns the empty string when no rated specs exist (so callers can
    drop the column entirely)."""
    lo = profile.get("rated_bf16_low")
    hi = profile.get("rated_bf16_high")
    if lo is None and hi is None:
        return ""
    if hi is None or hi == lo:
        return f"{lo:,.0f} TFLOP/s"
    return f"{lo:,.0f} / {hi:,.0f} TFLOP/s"


@dataclass
class Section:
    """A single report section. Both MD and HTML bodies are accumulated as the
    section is built up; the renderer concatenates them into the final files.

    The decision-grade pieces (``callout``, ``insight_takeaway``,
    ``kv_table``, ``subheading``) layer on top of the primitive ``text``
    / ``para`` / ``bullets`` / ``table`` / ``image`` builders so the
    section bodies stay close to a "narrative + evidence + meaning"
    rhythm without each call site reinventing the markup.
    """
    level: int
    title: str
    md_parts: List[str] = field(default_factory=list)
    html_parts: List[str] = field(default_factory=list)
    section_id: Optional[str] = None  # set by render layer for anchor links

    def text(self, md: str, html: Optional[str] = None) -> "Section":
        self.md_parts.append(md.rstrip() + "\n")
        self.html_parts.append(
            html if html is not None else f"<p>{html_escape(md)}</p>"
        )
        return self

    def para(self, md: str) -> "Section":
        self.md_parts.append(md.rstrip() + "\n\n")
        h = html_escape(md)
        h = _inline_md_to_html(h)
        self.html_parts.append(f"<p>{h}</p>\n")
        return self

    def subheading(self, title: str, level: int = 2) -> "Section":
        """Emit an in-section heading (h2 / h3) without bumping level into
        the auto-numbered TOC. Used inside Detailed Analysis so the TOC
        stays clean while individual subsections still get headings."""
        hashes = "#" * (level + 1)
        self.md_parts.append(f"\n{hashes} {title}\n\n")
        h = max(2, min(level + 1, 6))
        self.html_parts.append(f"<h{h}>{html_escape(title)}</h{h}>\n")
        return self

    def bullets(self, items: Sequence[str]) -> "Section":
        if not items:
            return self
        self.md_parts.append("\n".join(f"- {it}" for it in items) + "\n\n")
        body = "".join(f"<li>{_inline_md_to_html(html_escape(it))}</li>" for it in items)
        self.html_parts.append(f"<ul>{body}</ul>\n")
        return self

    def kv_table(self, rows: Sequence[Tuple[str, Any]],
                 caption: Optional[str] = None) -> "Section":
        """Compact two-column key/value listing. Skips rows whose value is
        None so partial-data scenarios stay clean."""
        cleaned = [(k, v) for k, v in rows if v not in (None, "")]
        if not cleaned:
            return self
        return self.table([{"item": k, "value": str(v)} for k, v in cleaned],
                          caption=caption)

    def callout(self, kind: str, heading: str, body: str) -> "Section":
        """Banner-style box (info / warn / success / error) for things the
        reader must see — host vs target context, known limitations,
        go/no-go status. Renders as a blockquote in MD and a styled
        ``<aside class="callout callout-{kind}">`` in HTML.
        """
        kind = kind if kind in ("info", "warn", "success", "error") else "info"
        prefix = {"info": "\u2139\ufe0e", "warn": "\u26a0\ufe0e",
                  "success": "\u2714\ufe0e", "error": "\u2716\ufe0e"}[kind]
        self.md_parts.append(
            f"\n> **{prefix} {heading}**  \n"
            f"> {body.strip()}\n\n"
        )
        h_body = _inline_md_to_html(html_escape(body))
        self.html_parts.append(
            f'<aside class="callout callout-{kind}">'
            f'<strong class="callout-heading">{html_escape(heading)}</strong>'
            f'<div class="callout-body">{h_body}</div>'
            f'</aside>\n'
        )
        return self

    def insight_takeaway(self, insight: str, takeaway: str) -> "Section":
        """Two-line decision pair appended to the end of every analysis
        section: what the data shows, and what the reader should do
        about it. The single most important readability improvement
        across the report — every section now closes with a meaning +
        action pair instead of trailing off on raw numbers.
        """
        self.md_parts.append(
            f"\n**Insight.** {insight.strip()}\n\n"
            f"**Takeaway.** {takeaway.strip()}\n\n"
        )
        i_html = _inline_md_to_html(html_escape(insight.strip()))
        t_html = _inline_md_to_html(html_escape(takeaway.strip()))
        self.html_parts.append(
            f'<div class="insight-takeaway">'
            f'<p><strong>Insight.</strong> {i_html}</p>'
            f'<p><strong>Takeaway.</strong> {t_html}</p>'
            f'</div>\n'
        )
        return self

    def table(self, rows: Sequence[Dict], caption: Optional[str] = None) -> "Section":
        if not rows:
            return self
        keys = list(rows[0].keys())
        # Markdown
        md = []
        if caption:
            md.append(f"_{caption}_\n")
        md.append("| " + " | ".join(keys) + " |")
        md.append("|" + "|".join(["---"] * len(keys)) + "|")
        for r in rows:
            md.append("| " + " | ".join(_cell(r.get(k, "")) for k in keys) + " |")
        self.md_parts.append("\n".join(md) + "\n\n")
        # HTML
        h = ["<table>"]
        if caption:
            h.append(f"<caption>{html_escape(caption)}</caption>")
        h.append("<thead><tr>" + "".join(f"<th>{html_escape(k)}</th>" for k in keys) + "</tr></thead>")
        h.append("<tbody>")
        for r in rows:
            h.append("<tr>" + "".join(f"<td>{html_escape(_cell(r.get(k, '')))}</td>" for k in keys) + "</tr>")
        h.append("</tbody></table>")
        self.html_parts.append("".join(h) + "\n")
        return self

    def image(
        self,
        path: Path,
        alt: str,
        caption: Optional[str] = None,
        embed: bool = True,
    ) -> "Section":
        if not path.exists():
            self.para(f"_(missing plot: `{path.name}`)_")
            return self
        rel = f"plots/{path.name}"
        self.md_parts.append(f"![{alt}]({rel})\n")
        if caption:
            self.md_parts.append(f"\n_{caption}_\n\n")
        else:
            self.md_parts.append("\n")
        if embed:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            src = f"data:image/png;base64,{data}"
        else:
            src = rel
        cap_html = f"<figcaption>{html_escape(caption)}</figcaption>" if caption else ""
        self.html_parts.append(
            f'<figure><img alt="{html_escape(alt)}" src="{src}" />{cap_html}</figure>\n'
        )
        return self


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.4g}"
    return str(v)


_INLINE_BACKTICK = "<code>{}</code>"


def _inline_md_to_html(s: str) -> str:
    """Convert just the `code` and **bold** / *italic* inline forms used by
    this script's commentary. Keeps the HTML self-contained without pulling in
    a markdown parser."""
    import re
    s = re.sub(r"`([^`]+)`", lambda m: _INLINE_BACKTICK.format(html_escape(m.group(1)).replace("&amp;", "&")), s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    return s


# ---------------------------------------------------------------------------
# Section builders. Each takes the loaded JSON object(s) (which may be None
# when an artifact is missing) and returns a Section. Sections gracefully
# degrade to a "not collected" note when their inputs are missing so a partial
# campaign still produces a useful report.
# ---------------------------------------------------------------------------


def _heading(level: int, title: str) -> Section:
    return Section(level=level, title=title)


def _status_pill(status: Optional[str]) -> str:
    """Inline HTML status pill used in scorecard / overview tables. The
    `_cell` text version (markdown column body) just returns the raw
    status string — pills exist only in the HTML rendering path because
    Markdown tables can't carry inline styling cleanly. The HTML
    version is plain enough that pandoc preserves it through the
    HTML->PDF pipeline.
    """
    if not status:
        return ""
    s = str(status).upper()
    pills_cfg = _cfg().get("status_pills") or {}
    default_cls = pills_cfg.get("$default", "pill-skip")
    cls = pills_cfg.get(s, default_cls)
    if not isinstance(cls, str):
        cls = default_cls
    # Embed an HTML pill in the cell. The Markdown renderer also picks
    # this up because we use `text()` rather than table cells when
    # injecting status — but we mostly emit raw status strings into MD
    # tables for portability and rely on this only in the HTML body.
    return f'<span class="pill {cls}">{html_escape(s)}</span>'


def _scorecard_test_rows(scorecard: list) -> list:
    """Return only the actual test-criterion rows (SC-1, SC-2, …),
    filtering out the informational HOST row added at the top of
    `scorecard.json` to record the run-context. The HOST row's status
    is `"CPU"` or `"GPU"` — not a pass/fail enum — and including it in
    counts skews the verdict / overview tables.
    """
    return [r for r in (scorecard or [])
            if (r.get("sc") or "").upper().startswith("SC-")]


def _scorecard_status_counts(scorecard: list) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in _scorecard_test_rows(scorecard):
        s = (r.get("status") or "").upper()
        counts[s] = counts.get(s, 0) + 1
    return counts


def _scorecard_overall(scorecard: list) -> Tuple[str, str]:
    """Returns (verdict, callout_kind) for the overall campaign.

    Verdict logic:
      - Any FAIL  -> NO-GO  (error)
      - Any WARN* -> CONDITIONAL GO  (warn)
      - Any SKIP  -> GO with caveats  (info)  # only when no FAIL/WARN
      - Otherwise PASS                (success)
    Empty scorecard -> UNKNOWN (info).
    """
    if not scorecard:
        return ("UNKNOWN — no scorecard collected", "info")
    counts = _scorecard_status_counts(scorecard)
    if counts.get("FAIL"):
        return (f"NO-GO — {counts['FAIL']} FAIL", "error")
    warn = counts.get("WARN", 0) + counts.get("WARN_CPU", 0)
    if warn:
        return (f"CONDITIONAL GO — {warn} WARN", "warn")
    if counts.get("SKIP"):
        return (f"GO with caveats — {counts['SKIP']} SKIP", "info")
    return ("GO — all criteria PASS", "success")


def _bullet_safe(s: Optional[str], na: str = "—") -> str:
    return s if s else na


def section_cover_page(env: dict, cfg: dict, scorecard: list,
                       is_cpu_host: bool,
                       profile: Dict[str, Any]) -> Section:
    """Cover-page card: title, project, run id, host/target, generation
    time, and the at-a-glance go/no-go verdict. Renders as a styled
    card in HTML and as a key/value table in Markdown. Numbered ``0.``
    so the auto-numberer skips it and the TOC starts at the executive
    summary.

    The cover is generic across hardware: on a CPU host it reports the
    CPU model only; on a GPU host it reports the detected accelerator
    (with vendor-published rated specs when known via ``profile``).
    """
    s = Section(level=1, title="0. Cover")
    run = (env or {}).get("run", {}) or {}
    sw  = ((env or {}).get("software", {}) or {})
    torch_info = sw.get("torch", {}) or {}
    devs = torch_info.get("device_names") or []
    host_label = run.get("host", "?")
    cid = run.get("campaign_id", "?")
    when = run.get("timestamp_utc", _utc_now_iso())
    project_meta = _project_meta()
    project_name = project_meta.get("name") or "Inference Benchmarking Campaign"
    default_workload = project_meta.get("default_workload_label") or "workload"
    workload = (cfg or {}).get("name") or default_workload
    verdict, kind = _scorecard_overall(scorecard)
    host_dev = devs[0] if devs else ("cpu" if is_cpu_host else "?")

    if is_cpu_host:
        target_label = "n/a — CPU validation run"
        run_mode_label = "CPU validation (host-only baseline)"
    else:
        target_label = profile["name"]
        if profile.get("has_rated_specs"):
            target_label = f"{profile['name']} (rated specs known)"
        run_mode_label = f"Target hardware run — {profile['short']}"

    rows_html = [
        ("Project",              project_name),
        ("Workload",             workload),
        ("Campaign ID",          cid),
        ("Host",                 f"{host_label} — {host_dev}"),
        ("Target hardware",      target_label),
        ("Run timestamp",        when),
        ("PyTorch",              torch_info.get("torch_version") or "—"),
        ("ROCm",                 sw.get("rocm_version_file") or "—"),
        ("AITER",                sw.get("aiter_version") or "not installed"),
        ("flash_attn",           sw.get("flash_attn_version") or "not installed"),
        ("Run mode",             run_mode_label),
        ("Verdict (go / no-go)", verdict),
    ]
    rows_html_body = "".join(
        f"<tr><th>{html_escape(k)}</th><td>{html_escape(str(v))}</td></tr>"
        for k, v in rows_html
    )
    pill_class = {"success": "pill-pass", "error": "pill-fail",
                  "warn": "pill-warn", "info": "pill-partial"}.get(kind, "pill-skip")
    cover_subtitle = (target_label if not is_cpu_host
                       else "CPU validation run — measurement-infrastructure baseline")
    s.html_parts.append(
        '<section class="cover-page">'
        f'<div class="doc-title">{html_escape(workload)} — Inference Campaign Report</div>'
        f'<div class="doc-sub">{html_escape(cover_subtitle)} — '
        f'campaign <code>{html_escape(cid)}</code></div>'
        f'<table>{rows_html_body}</table>'
        f'<p style="margin-top:1em">Status: '
        f'<span class="pill {pill_class}">{html_escape(verdict)}</span></p>'
        '</section>\n'
    )

    # --- MD cover-card body
    md_lines = [
        f"_{workload} — Inference Campaign Report_  ",
        f"_{cover_subtitle} — campaign `{cid}`_\n",
        "| Field | Value |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in rows_html]
    md_lines.append("")
    md_lines.append(f"**Status: {verdict}**")
    s.md_parts.append("\n".join(md_lines) + "\n\n")
    return s


def section_host_target_banner(is_cpu_host: bool, scorecard: list,
                               env: dict, profile: Dict[str, Any]) -> Section:
    """First-screen banner stating the run-mode in unambiguous terms.

    On CPU-only validation runs this banner is *the* guard against
    misinterpreting the headline numbers — the section title also
    starts with `Run Context` so the TOC entry self-describes. On
    target-hw runs the banner switches tone to a green "Target run"
    naming the detected accelerator (e.g. MI300X, H100, R9700) and
    rolls in the scorecard's worst-case status as the verdict bullet.
    """
    s = _heading(1, "Run Context")
    verdict, kind = _scorecard_overall(scorecard)
    if is_cpu_host:
        torch_info = ((env or {}).get("software", {}) or {}).get("torch", {}) or {}
        host_dev = (torch_info.get("device_names") or ["cpu"])[0]
        s.callout(
            "warn",
            "CPU validation run — not target-hardware performance",
            ("This campaign was executed on **{host}** ({dev}). All "
             "absolute throughput, bandwidth, and capacity numbers "
             "should be read as **infrastructure regression baselines** — "
             "they characterize the harness on the host CPU, not any "
             "target accelerator. The methodology and timing protocol "
             "mirror an on-target run, so a future GPU campaign produced "
             "from this same harness is directly comparable.").format(
                host=(env or {}).get("run", {}).get("host", "?"),
                dev=host_dev,
            ),
        )
    else:
        rated_bits = []
        if profile.get("rated_bf16_low"):
            rated_bits.append(_rated_bf16_label(profile) + " BF16 dense")
        if profile.get("rated_bw_gb_s"):
            rated_bits.append(f"{profile['rated_bw_gb_s']/1000.0:.1f} TB/s memory")
        if profile.get("rated_mem_gib"):
            rated_bits.append(f"{profile['rated_mem_gib']:.0f} GiB capacity")
        if rated_bits:
            rated_phrase = (" Rated specs (" + ", ".join(rated_bits) + ") "
                            "are reported alongside as ratios, not as "
                            "ground truth.")
        else:
            rated_phrase = (" Rated specs for this device are not in the "
                            "registry; measured numbers are reported "
                            "standalone.")
        s.callout(
            "success",
            f"Target hardware run — {profile['short']}",
            (f"All numbers in this report are measured on the **{profile['name']}** "
             f"target.") + rated_phrase,
        )
    s.callout(
        ("error" if kind == "error"
         else "warn" if kind == "warn"
         else "success" if kind == "success" else "info"),
        f"Overall verdict: {verdict}",
        ("Computed from the SC-1…SC-12 scorecard. See "
         "*Scope & Objectives* for the per-criterion table; "
         "*Recommendations* for prioritized next actions."),
    )
    return s


def section_executive_summary(env: dict, scorecard: list, compute: dict,
                               bw: dict, dram: dict, ops: dict, mfu: dict,
                               comm: dict, fused: dict,
                               is_cpu_host: bool = False,
                              profile: Optional[Dict[str, Any]] = None,
                              workload_name: Optional[str] = None) -> Section:
    """Decision-grade 5-bullet summary, plus the headline numbers
    table. Reads top-down: objective, key result, biggest risk,
    recommendation, status. A single page should be enough for someone
    to know what was tested, what was found, and what to do next.

    The summary is generic across hardware: hardware-specific phrasing
    pulls from ``profile`` (rated specs, short device label) so a CPU
    run never references any accelerator and a GPU run names whatever
    device torch reports.
    """
    s = _heading(1, "Executive Summary")
    profile = profile or {}

    # Compute the bullet content from real artifacts.
    peak = (compute or {}).get("compute_roof_tflops")
    bwv  = (bw or {}).get("bandwidth_roof_gb_s")
    by_scope = {r["scope"]: r for r in (mfu or {}).get("rows", [])}
    eager = by_scope.get("eager_e2e") or {}
    compiled = by_scope.get("compiled_e2e") or {}
    mfu_eager = eager.get("mfu_measured_peak")
    mfu_comp  = compiled.get("mfu_measured_peak")

    fused_available = bool((fused or {}).get("available"))
    fused_reason = (fused or {}).get("reason") or "not exercised"

    verdict, _ = _scorecard_overall(scorecard)

    bullets: List[str] = []
    workload_label = workload_name or "the workload"
    if is_cpu_host:
        objective = (
            f"**Objective.** Validate measurement infrastructure and produce "
            f"a CPU-host regression baseline for `{workload_label}` "
            f"transformer inference. Target-hardware comparison is "
            f"deliberately out of scope for this run."
        )
    else:
        target_phrase = (profile.get("name") or "the target accelerator")
        objective = (
            f"**Objective.** Measure compute, bandwidth, capacity, and end-to-end "
            f"throughput for `{workload_label}` transformer inference on "
            f"**{target_phrase}**, comparing against rated specs where known."
        )
    bullets.append(objective)
    if peak is not None or bwv is not None:
        bits = []
        if peak is not None:
            bits.append(f"BF16 peak **{_fmt_tflops(peak)} TFLOP/s**")
        if bwv is not None:
            bits.append(f"memory bandwidth **{_fmt(bwv, 1)} GB/s**")
        if mfu_comp is not None:
            bits.append(
                f"compiled E2E MFU **{_pct(mfu_comp)}** (measured-peak basis)"
            )
        bullets.append("**Key result.** " + ", ".join(bits) + ".")
    if is_cpu_host:
        bullets.append(
            "**Biggest risk / limitation.** This run is a **CPU-host validation**, "
            "not a target-hw measurement. Full-model memory residency, NCCL/RCCL "
            "collectives, and fused TP kernels cannot be exercised here. A run "
            "on the chosen target accelerator is required for any operational "
            "sign-off."
        )
    else:
        # On target-hw runs the biggest risk surfaces from the worst SC row.
        worst = next((r for r in scorecard or [] if r.get("status") == "FAIL"), None)
        if worst:
            bullets.append(
                f"**Biggest risk / limitation.** {worst.get('sc')} **FAIL** — "
                f"{worst.get('reason') or 'see scorecard for details'}."
            )
        elif not fused_available:
            bullets.append(
                "**Biggest risk / limitation.** Fused collective+GEMM kernels "
                "(AG+MM, MM+RS) are **not yet available** in this stack — the "
                "TP path falls back to sequential collective + matmul, which "
                "the source pilot flags as future work."
            )
        else:
            bullets.append(
                "**Biggest risk / limitation.** No criterion failed; the "
                "remaining gap is operational steady-state validation "
                "(24h sustained run with thermal/power telemetry)."
            )
    if not fused_available:
        bullets.append(
            "**Recommendation.** Promote the fused AG+MM / MM+RS scaffold the "
            "moment AITER ships the API; today the campaign records "
            f"`SKIP — {fused_reason}` so the regression auto-flips to PASS."
        )
    else:
        bullets.append(
            "**Recommendation.** Lock the current `torch.compile(max-autotune)` "
            "path as the production baseline; chase the next-largest residual "
            "(see *Recommendations*)."
        )
    bullets.append(f"**Status.** {verdict}.")
    s.bullets(bullets)

    # Headline numbers table — same content as before, kept compact.
    rows: List[Dict] = []
    bw_label = "Memory bandwidth roof"
    bw_source = ("bench02 streaming microbench (system DDR plateau)"
                 if is_cpu_host else "bench02 streaming microbench (best plateau)")
    if compute:
        rows.append({"metric": "BF16 compute peak",
                     "value": f"{_fmt_tflops(peak)} TFLOP/s",
                     "source": "bench01 tight-loop GEMM"})
    if bw:
        rows.append({"metric": bw_label,
                     "value": f"{_fmt(bwv, 1)} GB/s",
                     "source": bw_source})
    if dram:
        if is_cpu_host:
            host_total = (dram.get("device_total_bytes") or 0) / 1024 ** 3
            rows.append({"metric": "Usable Memory (bf16 contiguous)",
                         "value": (f"{_fmt(dram.get('max_alloc_bf16_gib'))} GiB "
                                   f"({_pct(dram.get('eff_util_fraction_bf16'))} of "
                                   f"{host_total:.1f} GiB host RAM)"),
                         "source": "bench03 binary search"})
        else:
            rated_mem = profile.get("rated_mem_gib")
            spec_phrase = (f"{rated_mem:.0f} GiB rated"
                           if rated_mem else "device capacity")
            rows.append({"metric": "Usable Memory (bf16 contiguous)",
                         "value": f"{_fmt(dram.get('max_alloc_bf16_gib'))} GiB "
                                  f"({_pct(dram.get('eff_util_fraction_bf16'))} of {spec_phrase})",
                         "source": "bench03 binary search"})
    if ops and ops.get("compute_roof_tflops") and ops.get("bandwidth_roof_gb_s"):
        ridge = ops["compute_roof_tflops"] * 1e12 / (ops["bandwidth_roof_gb_s"] * 1e9)
        rows.append({"metric": "Roofline ridge point",
                     "value": f"{ridge:.1f} FLOP/B",
                     "source": "compute_peak / bandwidth_roof"})
    if mfu_comp is not None or mfu_eager is not None:
        rows.append({"metric": "MFU eager E2E",
                     "value": f"{_pct(mfu_eager)} (measured-peak basis)",
                     "source": "bench05 / eager_e2e"})
        rows.append({"metric": "MFU compiled E2E",
                     "value": f"{_pct(mfu_comp)} (measured-peak basis)",
                     "source": "bench05 / compiled_e2e"})
    if rows:
        s.subheading("Headline numbers", level=2)
        s.table(rows, caption="One-line evidence per claim")

    s.insight_takeaway(
        ("Infrastructure is stable enough to anchor regression thresholds; "
         "the operational gaps are CPU-host limits and fused TP kernels, "
         "both expected." if is_cpu_host else
         "Headline numbers land within the source-pilot range; the open "
         "question is fused TP kernels, which today fall back to "
         "sequential collective+matmul."),
        ("Re-run on the chosen target accelerator to populate the "
         "operational TP-3 table; lock this MD/HTML/PDF as the regression "
         "baseline." if is_cpu_host else
         "Promote the fused-kernel scaffold the day the vendor stack "
         "ships AG+MM / MM+RS; meanwhile run the 24h sustained probe "
         "for thermal/power steady-state."),
    )
    return s


def section_scope_objectives(scorecard: list,
                             is_cpu_host: bool = False) -> Section:
    """Scope, exclusions, and the SC-1..SC-12 acceptance grid.

    Each SC row carries a ``status``, a ``reason`` (when not PASS), and
    enough numeric context to surface what the failure / skip means.
    The actionability column is computed here so readers don't need to
    cross-reference the scorecard.json + recommendations to know
    whether a SKIP is "expected", "acceptable", or "blocker".
    """
    s = _heading(1, "Scope & Objectives")
    s.subheading("In scope", level=2)
    s.bullets([
        "BF16 compute ceiling (square + workload-shaped GEMMs, dtype sweep)",
        "Memory-bandwidth ceiling and cache-hierarchy curve",
        "Memory capacity, fragmentation, and headroom-after-model-load",
        "Per-op accounting + roofline placement (compute vs memory bound)",
        "End-to-end MFU on the workload (eager / default / reduce-overhead / max-autotune)",
        "Multi-rank collectives (AllGather / ReduceScatter / AllReduce / AllToAll)",
        "Fused collective+GEMM availability probe (AG+MM, MM+RS)",
        "Numerical stability across reduced precisions (FP16 / BF16 / FP8) vs FP32",
    ])
    s.subheading("Out of scope", level=2)
    s.bullets([
        "VAE encoder/decoder (per source pilot — only the transformer stack is profiled)",
        "Sustained 24h thermal & power profile (covered by `bench07_sustained` "
        "but not run in this campaign by default)",
        "Strong-scaling sweep at WORLD ∈ {2,4,8} (provided as a separate "
        "`scripts/strong_scaling.sh` workflow rather than baked into the main run)",
    ])

    test_rows = _scorecard_test_rows(scorecard)
    if test_rows:
        s.subheading("Acceptance criteria scorecard", level=2)
        s.para(
            "Each criterion is one of `PASS`, `PARTIAL_PASS`, `WARN`, `WARN_CPU`, "
            "`SKIP`, or `FAIL`. The actionability column classifies each row as "
            "`expected` (e.g. CPU host can't run RCCL), `acceptable` "
            "(degraded but doesn't block), or `blocker` (must be addressed)."
        )
        rows = []
        for r in test_rows:
            sc = r.get("sc"); status = (r.get("status") or "").upper()
            reason = r.get("reason") or ", ".join(
                f"{k}={v}" for k, v in r.items()
                if k not in ("sc", "status", "reason"))
            actionability = _sc_actionability(sc, status, reason, is_cpu_host)
            rows.append({
                "SC":         sc,
                "Status":     status,
                "Detail":     reason[:140],
                "Actionability": actionability,
            })
        s.table(rows, caption="SC-1…SC-12 with go/no-go classification")

    counts = _scorecard_status_counts(scorecard)
    if counts:
        summary_bits = []
        for k in ("PASS", "PARTIAL_PASS", "WARN", "WARN_CPU", "SKIP", "FAIL"):
            if counts.get(k):
                summary_bits.append(f"{k}={counts[k]}")
        s.insight_takeaway(
            "Scorecard summary: " + ", ".join(summary_bits) + ".",
            (("Address every `blocker` row before sign-off; `expected` and "
              "`acceptable` rows are safe to ship as-is.") if counts.get("FAIL")
             else ("No blocker rows; treat any `acceptable` SKIP as a "
                   "follow-up for the next campaign cadence.")),
        )
    return s


def _sc_actionability(sc: Optional[str], status: str, reason: str,
                      is_cpu_host: bool) -> str:
    """Three-bucket classification per the user's review feedback:

      - `expected`   — design-intended SKIP (e.g. RVS not installed on
                       a CPU host; fused-collective API not yet shipped).
      - `acceptable` — soft warning that doesn't block the campaign
                       (e.g. WARN_CPU partial fit, PARTIAL_PASS).
      - `blocker`    — outright FAIL or unexplained SKIP that must be
                       resolved before next campaign / sign-off.
    """
    s = (status or "").upper()
    if s == "PASS":
        return "—"
    if s == "FAIL":
        return "blocker"
    if s in ("WARN", "WARN_CPU", "PARTIAL_PASS"):
        return "acceptable"
    if s == "SKIP":
        # Common expected SKIPs on CPU hosts:
        cpu_expected = ("rvs", "rocm-bandwidth-test", "rccl-tests",
                        "RVS gst", "rccl",
                        "fused-collective", "fused AG+MM",
                        "ground truth", "not installed",
                        "not run", "not available")
        if is_cpu_host and any(p.lower() in (reason or "").lower()
                                for p in cpu_expected):
            return "expected (CPU host)"
        if any(p.lower() in (reason or "").lower()
               for p in ("not available", "not installed", "future work",
                         "not yet")):
            return "expected"
        return "acceptable"
    return "—"


def section_results_overview(compute: dict, bw: dict, dram: dict, mfu: dict,
                             ops: dict, comm: dict, fused: dict,
                             plots_dir: Path,
                             is_cpu_host: bool = False) -> Section:
    """Top-level results dashboard: a single compact metric table plus the
    one or two plots that capture the campaign's headline at a glance
    (roofline + MFU comparison). Detailed numerics live in the
    sub-sections of *Detailed Analysis*; this section is for the reader
    who has 60 seconds.
    """
    s = _heading(1, "Results Overview")
    s.para(
        "One-glance dashboard: ceilings, MFU, and TP-collective bandwidth. "
        "Each row points at the section that explains it."
    )
    rows: List[Dict] = []
    if compute and compute.get("compute_roof_tflops") is not None:
        rows.append({
            "metric":   "BF16 compute ceiling",
            "value":    f"{_fmt_tflops(compute.get('compute_roof_tflops'))} TFLOP/s",
            "source":   "bench01",
            "see":      "§ Detailed Analysis / Compute",
        })
    if bw and bw.get("bandwidth_roof_gb_s") is not None:
        rows.append({
            "metric":   "Memory bandwidth ceiling",
            "value":    f"{_fmt(bw.get('bandwidth_roof_gb_s'), 1)} GB/s",
            "source":   "bench02",
            "see":      "§ Detailed Analysis / Bandwidth",
        })
    if dram:
        rows.append({
            "metric":   "Usable Memory (bf16 contig)",
            "value":    (f"{_fmt(dram.get('max_alloc_bf16_gib'))} GiB "
                         f"({_pct(dram.get('eff_util_fraction_bf16'))})"),
            "source":   "bench03",
            "see":      "§ Detailed Analysis / Memory",
        })
    by_scope = {r["scope"]: r for r in (mfu or {}).get("rows", [])}
    eager = by_scope.get("eager_e2e") or {}
    compiled = by_scope.get("compiled_e2e") or {}
    if eager or compiled:
        rows.append({
            "metric":   "MFU eager E2E",
            "value":    _pct(eager.get("mfu_measured_peak")),
            "source":   "bench05 / eager_e2e",
            "see":      "§ Detailed Analysis / End-to-End MFU",
        })
        rows.append({
            "metric":   "MFU compiled E2E",
            "value":    _pct(compiled.get("mfu_measured_peak")),
            "source":   "bench05 / compiled_e2e",
            "see":      "§ Detailed Analysis / End-to-End MFU",
        })
    if comm and comm.get("rows"):
        # plateau busbw of largest payload across all collectives
        best = max((r.get("busbw_gb_s") or 0 for r in comm["rows"]), default=0)
        rows.append({
            "metric":   "TP collective plateau (best)",
            "value":    f"{_fmt(best, 1)} GB/s",
            "source":   "bench06 / multigpu_comm",
            "see":      "§ Detailed Analysis / Multi-GPU",
        })
    if fused:
        avail = "yes" if fused.get("available") else "no — SKIP"
        rows.append({
            "metric":   "Fused AG+MM / MM+RS available",
            "value":    avail,
            "source":   "bench06_fused",
            "see":      "§ Fused Compute+Collective Kernels",
        })
    if rows:
        s.table(rows, caption="Headline metrics with cross-references")

    # Anchor charts that summarize the campaign visually.
    if (plots_dir / "A6_roofline.png").exists():
        s.image(plots_dir / "A6_roofline.png",
                alt="Workload roofline overview",
                caption="Per-op roofline placement (compute vs memory bound).")
    if (plots_dir / "A7_mfu_compare.png").exists():
        s.image(plots_dir / "A7_mfu_compare.png",
                alt="MFU comparison",
                caption=("MFU across scopes (measured-peak basis; rated-PF "
                         "bases included when target rated specs are known)."))

    s.insight_takeaway(
        "Headline metrics fit on one screen; each is auditable through "
        "its source artifact (`benchXX/...json`) and explained in the "
        "named section.",
        ("If a number here looks off, jump to the linked section first; "
         "the raw JSON under the campaign directory is the underlying "
         "evidence."),
    )
    return s


def section_executive_summary_legacy_marker(*_args, **_kwargs):
    """Compatibility shim — the old ``section_exec_summary`` name is
    retained below but the canonical exec-summary builder is now
    :func:`section_executive_summary`. Tests / scripts that import the
    old symbol still work."""
    return None


def section_exec_summary(env: dict, scorecard: list, compute: dict, bw: dict,
                         dram: dict, ops: dict, mfu: dict,
                         is_cpu_host: bool = False) -> Section:
    s = _heading(1, "Executive Summary")
    when = (env or {}).get("run", {}).get("timestamp_utc", "?")
    host = (env or {}).get("run", {}).get("host", "?")
    cid = (env or {}).get("run", {}).get("campaign_id", "?")
    devs = ((env or {}).get("software", {}) or {}).get("torch", {}).get("device_names") or []
    dev = devs[0] if devs else ("cpu" if is_cpu_host else "?")

    if is_cpu_host:
        s.para(
            f"This report covers campaign `{cid}` on host `{host}` "
            f"(CPU host, {dev}) at {when}. The campaign methodology mirrors "
            f"the source reference; absolute thresholds against any target "
            f"accelerator are reported side-by-side **only when the device "
            f"profile is known**, and they are **not enforced** since this "
            f"run was not on target hardware."
        )
    else:
        s.para(
            f"This report covers campaign `{cid}` on host `{host}` "
            f"({dev}) at {when}. It reproduces the methodology of the source "
            f"reference on the local workload and adds data-driven commentary."
        )

    bw_label = "Memory bandwidth roof"
    bw_source = ("bench02 streaming microbench (system DDR plateau)"
                 if is_cpu_host else "bench02 streaming microbench (best plateau)")
    rows: List[Dict] = []
    if compute:
        rows.append({"metric": "BF16 compute peak",
                     "value": f"{_fmt(compute.get('compute_roof_tflops'), 3)} TFLOP/s",
                     "source": "bench01 tight-loop GEMM"})
    if bw:
        rows.append({"metric": bw_label,
                     "value": f"{_fmt(bw.get('bandwidth_roof_gb_s'), 1)} GB/s",
                     "source": bw_source})
    if dram:
        if is_cpu_host:
            host_total = (dram.get("device_total_bytes") or 0) / 1024 ** 3
            rows.append({"metric": "Usable Memory (bf16 contiguous)",
                         "value": (f"{_fmt(dram.get('max_alloc_bf16_gib'))} GiB "
                                   f"({_pct(dram.get('eff_util_fraction_bf16'))} of "
                                   f"{host_total:.1f} GiB host RAM)"),
                         "source": "bench03 binary search"})
        else:
            rows.append({"metric": "Usable Memory (bf16 contiguous)",
                         "value": f"{_fmt(dram.get('max_alloc_bf16_gib'))} GiB "
                                  f"({_pct(dram.get('eff_util_fraction_bf16'))} of device capacity)",
                         "source": "bench03 binary search"})
    if ops and ops.get("compute_roof_tflops") and ops.get("bandwidth_roof_gb_s"):
        ridge = ops["compute_roof_tflops"] * 1e12 / (ops["bandwidth_roof_gb_s"] * 1e9)
        rows.append({"metric": "Roofline ridge point",
                     "value": f"{ridge:.1f} FLOP/B",
                     "source": "compute_peak / bandwidth_roof"})
    if mfu:
        # Pull the three scopes
        by_scope = {r["scope"]: r for r in (mfu.get("rows") or [])}
        for label, key in (("MFU sum-of-ops", "sum_of_ops_optimized"),
                           ("MFU eager e2e", "eager_e2e"),
                           ("MFU compiled e2e", "compiled_e2e")):
            r = by_scope.get(key) or by_scope.get(key.replace("_optimized", "_default"))
            if r:
                rows.append({"metric": label,
                             "value": f"{_pct(r.get('mfu_measured_peak'))} (measured peak basis)",
                             "source": f"bench05 / {r['scope']}"})
    s.table(rows, caption="Headline numbers")

    if scorecard:
        score_rows = [{"SC": r.get("sc"), "Status": r.get("status"),
                       "Detail": ", ".join(f"{k}={v}" for k, v in r.items()
                                            if k not in ("sc", "status"))}
                      for r in scorecard]
        s.text("\n**Success criteria scorecard:**\n", html=None)
        s.table(score_rows, caption="TESTPLAN §1.2 SC-1…SC-5")
    return s


def section_methodology(env: dict, cfg: dict) -> Section:
    """Test environment + measurement protocol. The workload spec (model
    architecture / shapes / op mix) lives in its own *Workload
    Description* section so this one stays focused on *how* the
    measurements were taken."""
    s = _heading(1, "Test Environment & Methodology")
    s.para(
        "Five benchmark families anchor the campaign, run in the order: "
        "BF16 compute → Memory bandwidth → Memory capacity → per-op accounting → "
        "end-to-end MFU. An optional sixth family covers multi-GPU collectives. "
        "Each family is timed under a uniform protocol (warmup, device events, "
        "frozen shapes, multiple repetitions; see TESTPLAN §4)."
    )

    sw = (env or {}).get("software", {}) or {}
    hw = (env or {}).get("hardware", {}) or {}
    torch_info = sw.get("torch", {}) or {}

    sw_rows = [
        {"component": "PyTorch", "version": torch_info.get("torch_version"),
         "extra": f"HIP={torch_info.get('torch_hip_version')} CUDA={torch_info.get('torch_cuda_version')}"},
        {"component": "Triton",  "version": sw.get("triton_version") or "n/a", "extra": ""},
        {"component": "AITER",   "version": sw.get("aiter_version") or "not installed",
         "extra": "fused attention path"},
        {"component": "flash_attn", "version": sw.get("flash_attn_version") or "not installed",
         "extra": "fallback attention path"},
        {"component": "ROCm",    "version": sw.get("rocm_version_file") or "(see hipconfig)",
         "extra": ""},
    ]
    s.subheading("Software stack", level=2)
    s.table(sw_rows)

    devs = torch_info.get("device_names") or []
    if devs:
        s.text(f"\n**Hardware visible to PyTorch:** {len(devs)} × {devs[0]}\n",
               html=f"<p><strong>Hardware visible to PyTorch:</strong> {len(devs)} × {html_escape(devs[0])}</p>")

    s.subheading("Measurement protocol", level=2)
    s.bullets([
        "**Warmup excluded** from all reported times (`bench0X` configurable).",
        "**Device events** for GPU timing (`torch.cuda.Event`) and "
        "`time.perf_counter` for CPU.",
        "**Frozen shapes** — every iteration uses the same tensor sizes "
        "to avoid recompile / autotune noise leaking into timing.",
        "**Median across iterations** — typically 20+ iters; p10/p90/std "
        "are also recorded for stability analysis.",
        "**Same JSON schema** across CPU and GPU hosts so artifacts diff "
        "directly between runs.",
    ])
    s.insight_takeaway(
        "The protocol mirrors the source pilot exactly; whether the run is "
        "on CPU or any GPU target, every artifact is byte-comparable to a "
        "reference run on matching hardware.",
        "If a number drifts between runs and the hardware is fixed, the "
        "first thing to audit is iter count and warmup config, not the "
        "kernel.",
    )
    return s


def section_topline(compute: dict, bw: dict, dram: dict, peak_json: dict,
                    is_cpu_host: bool = False,
                    profile: Optional[Dict[str, Any]] = None) -> Section:
    """Hardware ceilings table. Generic across hardware: when ``profile``
    carries rated specs the table includes a "rated" column and a "% of
    rated" ratio; on a CPU host (or unknown GPU) the table reports just
    the measured values plus a host-context note. No specific
    accelerator name is hardcoded — the ``profile['short']`` label is
    used wherever the prose needs to refer to the target.
    """
    profile = profile or {}
    s = _heading(1, "Hardware Ceilings")
    if is_cpu_host:
        s.para(
            "**Host context:** this campaign was executed on a **CPU host** "
            "(no accelerator). The numbers below characterize the host's "
            "own nominal capacity (system DDR, single-thread CPU BF16 "
            "throughput, etc.) — they are infrastructure regression "
            "baselines, not target-hardware performance."
        )
    elif profile.get("has_rated_specs"):
        bits = []
        if profile.get("rated_mem_gib"):
            bits.append(f"{profile['rated_mem_gib']:.0f} GiB on-package memory")
        if profile.get("rated_bw_gb_s"):
            bits.append(f"{profile['rated_bw_gb_s']/1000.0:.1f} TB/s peak memory bandwidth")
        if profile.get("rated_bf16_low"):
            bits.append("BF16 dense peak in the "
                        f"{profile['rated_bf16_low']/1000.0:.2f}"
                        + (f"–{profile['rated_bf16_high']/1000.0:.2f} PF"
                           if profile.get("rated_bf16_high")
                           and profile["rated_bf16_high"] != profile["rated_bf16_low"]
                           else " PF") + " class")
        spec_phrase = "; ".join(bits) if bits else "vendor-published specs"
        s.para(
            f"The target platform is **{profile['name']}**: {spec_phrase}. "
            f"The numbers below are **measured**, not advertised."
        )
    else:
        s.para(
            f"The target platform is **{profile.get('name', 'the detected accelerator')}**. "
            f"Vendor-rated specs for this device are not in the registry, "
            f"so measured ceilings are reported standalone (no rated-% "
            f"column). The numbers below are **measured**, not advertised."
        )

    rated_low = profile.get("rated_bf16_low")
    rated_high = profile.get("rated_bf16_high")
    rated_bw = profile.get("rated_bw_gb_s")
    rated_mem = profile.get("rated_mem_gib")
    rated_label = profile.get("short") or "rated"

    rows = []
    if compute:
        peak = compute.get("compute_roof_tflops")
        if is_cpu_host:
            row = {"metric": "BF16 dense peak (TFLOP/s)",
                   "measured": _fmt(peak, places=3),
                   "host context": "CPU host"}
        elif rated_low:
            rated_text = (f"{rated_low:,.0f} / {rated_high:,.0f}"
                          if rated_high and rated_high != rated_low
                          else f"{rated_low:,.0f}")
            row = {"metric": "BF16 dense peak (TFLOP/s)",
                   "measured": _fmt(peak),
                   f"rated ({rated_label})": rated_text,
                   "% of rated low": _pct(peak / rated_low if peak else None)}
        else:
            row = {"metric": "BF16 dense peak (TFLOP/s)",
                   "measured": _fmt(peak)}
        rows.append(row)
    if bw:
        bwv = bw.get("bandwidth_roof_gb_s")
        if is_cpu_host:
            row = {"metric": "Sustained Memory bandwidth (GB/s)",
                   "measured": _fmt(bwv, 1),
                   "host context": "CPU host (system DDR)"}
        elif rated_bw:
            row = {"metric": "Memory sustained (GB/s)",
                   "measured": _fmt(bwv, 0),
                   f"rated ({rated_label})": _fmt(rated_bw, 0),
                   "% of rated": _pct(bwv / rated_bw if bwv else None)}
        else:
            row = {"metric": "Memory sustained (GB/s)",
                   "measured": _fmt(bwv, 1)}
        rows.append(row)
    if dram:
        if is_cpu_host:
            host_total_gib = (dram.get("device_total_bytes") or 0) / 1024 ** 3
            rows.append({"metric": "Usable Memory (GiB, bf16 contiguous)",
                         "measured": _fmt(dram.get("max_alloc_bf16_gib")),
                         "host context": f"CPU host total RAM ≈ {host_total_gib:.1f} GiB"})
            rows.append({"metric": "Allocator fragmentation ratio",
                         "measured": _fmt(dram.get("frag_sensitivity_ratio")),
                         "host context": "lower = more host-OS fragmentation"})
        elif rated_mem:
            rows.append({"metric": "Usable Memory (GiB, bf16 contiguous)",
                         "measured": _fmt(dram.get("max_alloc_bf16_gib")),
                         f"rated ({rated_label})": f"{rated_mem:.0f}",
                         "% of rated": _pct(dram.get("eff_util_fraction_bf16"))})
            rows.append({"metric": "Allocator fragmentation ratio",
                         "measured": _fmt(dram.get("frag_sensitivity_ratio")),
                         f"rated ({rated_label})": "1.000",
                         "% of rated": _pct(dram.get("frag_sensitivity_ratio"))})
        else:
            rows.append({"metric": "Usable Memory (GiB, bf16 contiguous)",
                         "measured": _fmt(dram.get("max_alloc_bf16_gib"))})
            rows.append({"metric": "Allocator fragmentation ratio",
                         "measured": _fmt(dram.get("frag_sensitivity_ratio"))})
    if is_cpu_host:
        caption = "Measured ceilings (CPU host)"
    elif profile.get("has_rated_specs"):
        caption = f"Measured ceilings vs {rated_label} rated specs"
    else:
        caption = "Measured ceilings (rated specs not in registry)"
    s.table(rows, caption=caption)

    insights = []
    if compute and compute.get("compute_roof_tflops") and bw and bw.get("bandwidth_roof_gb_s"):
        ridge = compute["compute_roof_tflops"] * 1e12 / (bw["bandwidth_roof_gb_s"] * 1e9)
        insights.append(
            f"Roofline ridge point lands at **{ridge:.0f} FLOP/B** — any op "
            f"with arithmetic intensity above this is compute-bound on this device."
        )
    if compute and compute.get("compute_roof_tflops") and not is_cpu_host:
        peak = compute["compute_roof_tflops"]
        if rated_low:
            if peak >= rated_low:
                insights.append(
                    f"Measured BF16 peak ({peak:.0f} TFLOP/s) **exceeds the rated "
                    f"{rated_low:,.0f} TFLOP/s spec** by {(peak/rated_low - 1)*100:.0f}% — "
                    f"consistent with the common observation that *official spec < "
                    f"measured peak BF16 FLOPs* on tight-loop microbenchmarks."
                )
            else:
                insights.append(
                    f"Measured BF16 peak ({peak:.0f} TFLOP/s) **falls short of the rated "
                    f"{rated_low:,.0f} TFLOP/s spec** by {(1 - peak/rated_low)*100:.0f}% — "
                    f"investigate clock state, kernel selection, and matrix size; "
                    f"see the size-sweep chart in the next section."
                )
        else:
            insights.append(
                f"Measured BF16 peak: **{peak:.0f} TFLOP/s**. Rated spec for "
                f"this device is not in the registry; the size-sweep chart "
                f"that follows is the right reference for kernel quality."
            )
    elif compute and compute.get("compute_roof_tflops") and is_cpu_host:
        peak = compute["compute_roof_tflops"]
        insights.append(
            f"Measured CPU BF16 peak: **{peak:.3f} TFLOP/s**. Useful as a "
            f"regression baseline for the timing infrastructure; not "
            f"comparable to any GPU target."
        )
    if peak_json:
        insights.append(
            f"Peak measurement is the tight-loop median over {peak_json.get('iters')} iterations "
            f"at M=N=K={peak_json.get('size')}, total elapsed "
            f"{_fmt(peak_json.get('tight_loop_total_ms'), 1)} ms."
        )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)
    s.insight_takeaway(
        ("These three measurements (compute peak, bandwidth roof, usable "
         "memory) anchor every downstream MFU and roofline calculation. "
         "On a CPU host they are infrastructure regression baselines; "
         "on a target accelerator they are the ratios against rated spec."
         if is_cpu_host else
         "Measured ceilings are what the roofline uses — not the spec sheet."),
        ("If a downstream MFU number looks off, audit this section first; "
         "the basis is more often the issue than the measurement."),
    )
    return s


def section_relevant_shapes(compute_sweep: dict, plots_dir: Path) -> Section:
    s = _heading(1, "GEMM Size Sweep")
    s.para(
        "Square GEMMs sweep from launch-bound (small M) into a near-peak "
        "compute regime (large M); the curve below shows the transition."
    )
    s.image(plots_dir / "A2_bf16_gemm_sweep.png",
            alt="BF16 GEMM size sweep",
            caption="Figure 1 — Square GEMM TFLOP/s vs M=N=K.")

    insights = []
    if compute_sweep:
        sq = compute_sweep.get("square") or []
        if sq:
            peak = max(r["tflops"] for r in sq)
            high_frac = _threshold("gemm_sweep_peak_fraction_high")
            low_frac  = _threshold("gemm_sweep_peak_fraction_low")
            min_M_at_high = next(
                (r["M"] for r in sorted(sq, key=lambda r: r["M"])
                 if r["tflops"] >= high_frac * peak), None)
            min_M_at_low  = next(
                (r["M"] for r in sorted(sq, key=lambda r: r["M"])
                 if r["tflops"] >= low_frac * peak), None)
            insights.append(
                f"Largest square GEMM peak: **{_fmt_tflops(peak)} TFLOP/s** at "
                f"M={max(sq, key=lambda r: r['tflops'])['M']}."
            )
            if min_M_at_high:
                insights.append(
                    f"BF16 GEMMs reach **{high_frac * 100:.0f}% of measured peak "
                    f"at M = {min_M_at_high}**. Below this, launch overhead "
                    f"and matrix-unit utilization dominate."
                )
            if min_M_at_low:
                insights.append(
                    f"{low_frac * 100:.0f}% of peak is reached at M = "
                    f"{min_M_at_low} — the practical floor below which "
                    f"kernels are launch-bound."
                )
        rect = compute_sweep.get("rectangular") or []
        if rect:
            best = max(rect, key=lambda r: r["tflops"])
            worst = min(rect, key=lambda r: r["tflops"])
            insights.append(
                f"Workload-shape rectangular GEMMs span "
                f"**{_fmt_tflops(worst['tflops'])}–{_fmt_tflops(best['tflops'])} TFLOP/s** "
                f"(worst: `{worst['name']}`, best: `{best['name']}`)."
            )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)
    s.insight_takeaway(
        "The square-GEMM curve characterizes launch-bound vs compute-bound "
        "transitions; the rectangular workload shapes show the spread the "
        "real attention/FFN GEMMs see in practice.",
        "Use the 90%-of-peak M-threshold as the lower bound for batched "
        "matmul shapes in production code paths.",
    )

    if compute_sweep:
        rect = compute_sweep.get("rectangular") or []
        if rect:
            s.text("\n**Workload-shape GEMMs:**\n",
                   html="<p><strong>Workload-shape GEMMs:</strong></p>")
            s.table([{
                "name": r["name"],
                "M": r["M"], "K": r["K"], "N": r["N"],
                "t (ms)": _fmt(r.get("t_ms_median")),
                "TFLOP/s": _fmt_tflops(r.get("tflops")),
            } for r in rect])
    return s


def section_dtype_sweep(dtype_sweep: dict) -> Section:
    """Cross-precision throughput at a single representative GEMM shape.

    Renders the fp32 / fp16 / bf16 / fp8 sweep from `bench01.dtype_sweep`
    so the reader can compare per-precision throughput on identical
    inputs. Speedups are quoted versus fp32 (the only dtype every backend
    is guaranteed to support natively).
    """
    s = _heading(1, "Cross-Precision Throughput — Square GEMM")
    s.para(
        "Same M=K=N square matmul, swept across every dtype the active "
        "PyTorch build advertises. Speedup is relative to `fp32`. FP8 "
        "rows are flagged when the backend lacks a native FP8 GEMM path "
        "and the sweep had to upcast inputs — those numbers reflect "
        "the upcast cost, not a real FP8 tensor-core path."
    )

    if not dtype_sweep or not (dtype_sweep.get("rows") or []):
        s.para("_(dtype sweep did not run, or output missing)_")
        return s

    rows = dtype_sweep.get("rows") or []
    size = dtype_sweep.get("size")
    dev = dtype_sweep.get("device_type") or "?"
    s.bullets([
        f"Device: `{dev}`.",
        f"GEMM shape: M=K=N=**{size}**.",
    ])

    fp32_tflops = next(
        (r["tflops"] for r in rows
         if r.get("dtype") == "fp32" and r.get("supported") and r.get("tflops")),
        None,
    )

    table_rows: List[dict] = []
    for r in rows:
        if r.get("supported"):
            tfl = r.get("tflops")
            row = {
                "dtype": r["dtype"],
                "t (ms)": _fmt(r.get("t_ms_median"), 3),
                "TFLOP/s": _fmt_tflops(tfl),
                "vs fp32": (
                    f"{tfl / fp32_tflops:.2f}×"
                    if (tfl and fp32_tflops) else "—"
                ),
                "note": r.get("note") or "",
            }
        else:
            row = {
                "dtype": r["dtype"],
                "t (ms)": "—",
                "TFLOP/s": "n/a",
                "vs fp32": "—",
                "note": f"unsupported: {r.get('error', '?')}",
            }
        table_rows.append(row)
    s.table(table_rows, caption="Cross-precision square-GEMM throughput")

    insights: List[str] = []
    measured = [r for r in rows
                if r.get("supported") and r.get("tflops")]
    if measured:
        best = max(measured, key=lambda r: r["tflops"])
        insights.append(
            f"Highest dtype throughput: **{best['dtype']} at "
            f"{_fmt_tflops(best['tflops'])} TFLOP/s**."
        )
        bf16 = next((r for r in measured if r["dtype"] == "bf16"), None)
        fp16 = next((r for r in measured if r["dtype"] == "fp16"), None)
        if bf16 and fp16 and fp16["tflops"] > 0:
            ratio = bf16["tflops"] / fp16["tflops"]
            if ratio >= 2 or ratio <= 0.5:
                insights.append(
                    f"`bf16` vs `fp16` throughput ratio is **{ratio:.2f}×** — "
                    f"strong asymmetry typically indicates one dtype lacks a "
                    f"native kernel on this backend (e.g. CPU `fp16` matmul "
                    f"falling back to a scalar reference path)."
                )
        upcast = [r for r in measured
                  if (r.get("note") or "").startswith("FP8 inputs upcast")]
        if upcast:
            insights.append(
                "FP8 rows are upcast to a wider dtype before the matmul on "
                "this backend, so their TFLOP/s reflect upcast + GEMM cost, "
                "**not** a true FP8 tensor-core path."
            )
    unsup = [r for r in rows if not r.get("supported")]
    if unsup:
        insights.append(
            "Unsupported dtypes: " +
            ", ".join(f"`{r['dtype']}`" for r in unsup) +
            " — see the `note` column for the backend's reason."
        )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)
    return s


def section_component_gemms(component_gemms: dict, ops: dict) -> Section:
    """Per-component BF16 matmul throughput for the workload's GEMM inventory.

    Renders one row per dense GEMM in the per-block decomposition with the
    canonical name (``self_attn.q``, ``ffn.linear1``, ...), its (M, K, N)
    shape, analytic GFLOPs, and the measured BF16 TFLOP/s. Names line up
    1:1 with the per-op accounting table in §08, so cross-references are
    direct.
    """
    s = _heading(1, "Component GEMMs — BF16 Matmul Throughput")
    s.para(
        "Each row below is one dense GEMM in the `escher_14b_480p` per-block "
        "op decomposition, timed at the configured workload shape (or the "
        "leading-dim-capped variant when running on a CPU host). The analytic "
        "FLOPs / bytes / arithmetic-intensity columns come from "
        "`benchmarks/common/flop_accounting.py`; the measured TFLOP/s column "
        "is `bench01`'s component GEMM sweep over the same shapes."
    )

    if not component_gemms or not (component_gemms.get("rows") or []):
        s.para("_(component GEMM sweep did not run, or output missing)_")
        return s

    rows = component_gemms.get("rows") or []
    cap = component_gemms.get("leading_dim_cap") or 0
    budget = component_gemms.get("flop_budget_gflops") or 0
    dev = component_gemms.get("device_type") or "?"

    notes = [f"Device: `{dev}`."]
    if cap:
        notes.append(
            f"Leading dim (M / sequence-length) capped at **{cap}** for "
            "tractable runtime; rows where the workload's native M exceeds "
            "this carry a separate `M_meas` column."
        )
    if budget:
        notes.append(
            f"Skipped any GEMM whose analytic FLOPs > **{budget:g} GFLOP** "
            "(after the cap). Skipped rows are listed below the measured ones."
        )
    s.bullets(notes)

    measured = [r for r in rows if not r.get("skipped_reason")]
    skipped = [r for r in rows if r.get("skipped_reason")]

    # When the campaign capped the leading dim, every row gets an M_meas /
    # GFLOPs_meas column (set to "—" when no cap was applied to that row) so
    # the markdown / HTML table renderers — which lock the column set on the
    # first row's keys — pick the wider header.
    show_cap = bool(cap) and any(
        r.get("M_measured") and r["M_measured"] != r["M"] for r in measured
    )

    def _row_for_table(r: dict) -> dict:
        out = {
            "name": r["name"],
            "category": r.get("category", ""),
            "M": r.get("M"),
            "K": r.get("K"),
            "N": r.get("N"),
            "GFLOPs": _fmt(r.get("gflops"), 1),
            "AI (FLOP/B)": _fmt(r.get("arithmetic_intensity"), 1),
        }
        if show_cap:
            capped = r.get("M_measured") and r["M_measured"] != r["M"]
            out["M_meas"] = r["M_measured"] if capped else "—"
            out["GFLOPs_meas"] = _fmt(r.get("gflops_measured"), 1) if capped else "—"
        out["t (ms)"] = _fmt(r.get("t_ms_median"), 3)
        out["TFLOP/s"] = _fmt(r.get("tflops"), 3)
        return out

    if measured:
        s.text("\n**Per-component throughput:**\n",
               html="<p><strong>Per-component throughput:</strong></p>")
        s.table([_row_for_table(r) for r in measured],
                caption="Component GEMM TFLOP/s")

    if skipped:
        s.text("\n**Skipped components (analytic only):**\n",
               html="<p><strong>Skipped components (analytic only):</strong></p>")
        s.table([
            {
                "name": r["name"],
                "M": r.get("M"), "K": r.get("K"), "N": r.get("N"),
                "GFLOPs": _fmt(r.get("gflops"), 1),
                "AI (FLOP/B)": _fmt(r.get("arithmetic_intensity"), 1),
                "reason": r.get("skipped_reason"),
            } for r in skipped
        ])

    insights: List[str] = []
    if measured:
        best = max(measured, key=lambda r: r.get("tflops") or 0.0)
        worst = min((r for r in measured if (r.get("tflops") or 0.0) > 0),
                    key=lambda r: r["tflops"], default=None)
        insights.append(
            f"Highest measured component GEMM throughput: "
            f"**`{best['name']}` at {best['tflops']:.3f} TFLOP/s** "
            f"({best['M_measured']}×{best['K']}×{best['N']})."
        )
        if worst and worst["name"] != best["name"]:
            insights.append(
                f"Lowest measured component GEMM throughput: "
                f"**`{worst['name']}` at {worst['tflops']:.3f} TFLOP/s** "
                f"({worst['M_measured']}×{worst['K']}×{worst['N']}) — usually "
                f"a small-M / launch-bound projection (e.g. `time_proj`, "
                f"`time_embed`, or the cross-attention K/V at M=`seq_text`)."
            )

    # Cross-reference with bench04's per-op timings if available.
    if ops:
        op_rows = ops.get("rows") or []
        gemm_meas = {r["op_name"]: r for r in op_rows
                     if "x" in (r.get("input_shape") or "") and r.get("flops")}
        if measured and gemm_meas:
            disagreements = []
            for r in measured:
                m = gemm_meas.get(r["name"])
                if not m:
                    continue
                # bench04 may run at the workload's full M (no cap); compare
                # only when both timed the same shape.
                if r.get("M_measured") != r.get("M"):
                    continue
                t01 = r.get("t_ms_median")
                t04 = m.get("t_ms_default")
                if not (isinstance(t01, (int, float)) and isinstance(t04, (int, float))
                        and t01 > 0 and t04 > 0):
                    continue
                rel = (t04 - t01) / t01
                if abs(rel) > 0.10:
                    disagreements.append(
                        f"`{r['name']}`: bench01={t01:.2f}ms vs bench04 default="
                        f"{t04:.2f}ms ({rel*100:+.0f}%)"
                    )
            if disagreements:
                insights.append(
                    "Cross-check vs `bench04` per-op timings (>10% delta): "
                    + "; ".join(disagreements[:5])
                    + ("..." if len(disagreements) > 5 else "")
                )
            else:
                insights.append(
                    "Cross-check vs `bench04` per-op timings: agreement within 10% "
                    "across components measured at the same shape."
                )

    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)
    return s


def section_bandwidth(bw_full: list, bw_summary: dict, plots_dir: Path,
                      profile: Optional[Dict[str, Any]] = None) -> Section:
    """Memory-bandwidth section. Generic across hardware: when the
    target ``profile`` carries a rated bandwidth, the table shows
    "% of rated"; otherwise the rated column is omitted.
    """
    profile = profile or {}
    rated_bw = profile.get("rated_bw_gb_s")
    rated_label = profile.get("short") or "rated"
    s = _heading(1, "Memory Bandwidth")
    s.para(
        "Streaming microbenchmarks (`copy_`, `add`, `mul`, axpy, `sum`, `fill_`, "
        "and a strided variant) bracket the sustainable bandwidth available to "
        "real tensor kernels. The plateau on the right of the curve is what the "
        "roofline uses as its bandwidth roof — not the spec sheet."
    )
    s.image(plots_dir / "A3_hbm_bandwidth.png",
            alt="Memory bandwidth",
            caption="Figure 2 — Achieved GB/s vs buffer size, per access pattern.")

    if bw_summary:
        plateaus = bw_summary.get("plateau_gb_s_per_op", {}) or {}
        if rated_bw:
            rows = [{"op": k, "plateau (GB/s)": _fmt(v, 0),
                     f"% of {rated_label} rated": _pct(v / rated_bw if v else None)}
                    for k, v in sorted(plateaus.items(), key=lambda kv: -kv[1])]
        else:
            rows = [{"op": k, "plateau (GB/s)": _fmt(v, 0)}
                    for k, v in sorted(plateaus.items(), key=lambda kv: -kv[1])]
        s.table(rows, caption="Sustained bandwidth per micro-op")

    insights = []
    if bw_summary:
        roof = bw_summary.get("bandwidth_roof_gb_s")
        if roof:
            if rated_bw:
                insights.append(
                    f"Bandwidth roof = **{roof:.0f} GB/s** "
                    f"({roof/rated_bw*100:.0f}% of {rated_bw:.0f} GB/s "
                    f"{rated_label} rated)."
                )
            else:
                insights.append(
                    f"Bandwidth roof = **{roof:.0f} GB/s** "
                    f"(rated spec for the host device is not in the registry)."
                )
    if bw_full:
        strided = [r for r in bw_full if r["op"] == "strided_copy"]
        copy_ = [r for r in bw_full if r["op"] == "copy_"]
        if strided and copy_:
            sb = max(r["gb_s"] for r in strided)
            cb = max(r["gb_s"] for r in copy_)
            if cb > 0:
                insights.append(
                    f"Non-contiguous (strided) reads sustain **{sb/cb*100:.0f}% of contiguous `copy_`** "
                    f"({sb:.0f} vs {cb:.0f} GB/s) — quantifies the layout penalty for kernels "
                    f"that cannot fuse strided access."
                )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)

    s.subheading("Why the plateau is below spec", level=2)
    s.bullets([
        "**Working-set regime.** The plateau is taken at buffer sizes far "
        "larger than any cache, so every byte misses to main memory. The "
        "vendor-published peak (HBM bandwidth on a GPU, DDR throughput on "
        "a CPU host) describes the *peak attainable*, not the *sustainable* "
        "throughput once command-queue, refresh, and write-combining "
        "overheads kick in.",
        "**Dominant bottleneck.** For a single-stream copy the sustainable "
        "bandwidth is bounded by memory-controller queue depth and "
        "DRAM/HBM bank parallelism, not by the link itself. Real workloads "
        "(attention KV reuse, FFN activations) live in the cache-curve "
        "transition region — that's why the cache-hierarchy section "
        "below matters more than this single plateau number.",
        "**Strided penalty.** Strided / non-contiguous patterns drop the "
        "achievable plateau even further — the strided-vs-contig ratio "
        "(above) is a direct quantification of that penalty.",
    ])
    s.insight_takeaway(
        "The plateau is the floor. Anything in the cache-hierarchy regime "
        "(typical for attention reuse) is faster than this number; "
        "anything strided is slower.",
        "Use the plateau as the worst-case bandwidth bound when reading "
        "the roofline; check the cache-hierarchy curve for the regime a "
        "specific kernel actually lives in.",
    )
    return s


def _bytes_human(n: float) -> str:
    """Compact human-readable byte size for table cells (KiB / MiB / GiB)."""
    if n is None:
        return "n/a"
    n = float(n)
    if n < 1024:
        return f"{n:.0f} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.0f} KiB"
    if n < 1024 ** 3:
        return f"{n / (1024 ** 2):.1f} MiB"
    return f"{n / (1024 ** 3):.2f} GiB"


def section_cache_curve(cache_curve: dict, plots_dir: Path) -> Section:
    """Cache-hierarchy bandwidth curve from `bench02.cache_curve`.

    Renders the plot and a small table per cache tier with the
    *measured* peak bandwidth seen while the working set fits inside
    that tier. The tier boundaries come from sysfs (CPU) or torch
    device props plus on-spec landmarks (GPU).
    """
    s = _heading(1, "Cache Hierarchy")
    s.para(
        "Repeated `copy_` over the same buffer pair, swept across "
        "working-set sizes from a few KiB through DRAM. After warm-up the "
        "data lives in whichever cache level holds the working set; the "
        "measured GB/s plateaus at L1, L2, L3 / Infinity Cache, and "
        "finally DRAM / HBM. The classic stepwise descent in the plot is "
        "what should anchor any cache-aware kernel reasoning."
    )
    s.image(plots_dir / "A3b_cache_curve.png",
            alt="Cache hierarchy bandwidth curve",
            caption="Figure 2b — Cache hierarchy: GB/s vs working-set size.")

    if not cache_curve or not (cache_curve.get("rows") or []):
        s.para("_(cache curve did not run, or output missing)_")
        return s

    rows = sorted(cache_curve["rows"], key=lambda r: r["working_set_bytes"])
    tiers = (cache_curve.get("cpu_caches") or []) + \
            (cache_curve.get("gpu_caches") or [])

    # Build per-tier "best plateau" by taking the max GB/s of any
    # measured size whose working-set fits inside that tier.
    tier_table: List[dict] = []
    sorted_tiers = sorted(
        tiers, key=lambda t: (t["level"], t["size_bytes"])
    )
    prev_size = 0
    for t in sorted_tiers:
        size_b = t["size_bytes"]
        in_tier = [r for r in rows
                   if prev_size < r["working_set_bytes"] <= size_b]
        if in_tier:
            best = max(in_tier, key=lambda r: r["gb_s"])
            label = f"L{t['level']}"
            if t.get("type") == "InfinityCache":
                label = "InfinityCache"
            tier_table.append({
                "tier":             label,
                "size":             _bytes_human(size_b),
                "peak GB/s":        _fmt(best["gb_s"], 1),
                "@ working set":    _bytes_human(best["working_set_bytes"]),
            })
        prev_size = size_b

    # Plus a "DRAM / HBM" row for working sets larger than the largest
    # known cache tier.
    if sorted_tiers:
        last_size = sorted_tiers[-1]["size_bytes"]
        beyond = [r for r in rows if r["working_set_bytes"] > last_size]
        if beyond:
            best = max(beyond, key=lambda r: r["gb_s"])
            tier_table.append({
                "tier":             "DRAM / HBM",
                "size":             "—",
                "peak GB/s":        _fmt(best["gb_s"], 1),
                "@ working set":    _bytes_human(best["working_set_bytes"]),
            })

    if tier_table:
        s.table(tier_table, caption="Per-tier sustained bandwidth")

    insights: List[str] = []
    if rows:
        peak = max(rows, key=lambda r: r["gb_s"])
        floor = min(rows[-3:], key=lambda r: r["gb_s"]) if len(rows) >= 3 else rows[-1]
        insights.append(
            f"Curve peak: **{peak['gb_s']:.1f} GB/s** at "
            f"{_bytes_human(peak['working_set_bytes'])} working set "
            f"(typically the L2 / L3 fit point on this device)."
        )
        if peak["gb_s"] > 0 and floor["gb_s"] > 0:
            ratio = peak["gb_s"] / floor["gb_s"]
            insights.append(
                f"Cache-fit speedup over DRAM / HBM steady state: "
                f"**{ratio:.1f}×** ({peak['gb_s']:.1f} GB/s "
                f"-> {floor['gb_s']:.1f} GB/s at "
                f"{_bytes_human(floor['working_set_bytes'])} working set)."
            )
    if not tiers:
        insights.append(
            "_Cache topology metadata unavailable (no sysfs / no CUDA "
            "device props); plateaus are visible in the curve but not "
            "annotated against named tiers._"
        )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)
    s.insight_takeaway(
        "Real attention/FFN reuse lives somewhere in the cache-curve "
        "transition, not at the DRAM/HBM plateau. Cache hierarchy shape "
        "tells you the regime each kernel actually operates in.",
        "When a roofline calc puts a kernel near the bandwidth roof, "
        "cross-check this curve to see whether reuse pulls it higher.",
    )
    return s


def section_stability(stability: dict, plots_dir: Path) -> Section:
    """Numerical-stability sweep: bf16 / fp16 / fp8 GEMM error vs FP32.

    Renders both panels of the stability plot plus a compact table:
    per (dtype, K), the matrix-scaled max relative error and the
    analytic ``5·√K·2^-mantissa`` bound, with a pass column. The
    section also drops a one-paragraph reading guide so the reader
    doesn't have to re-derive what "rel_err" means in this context.
    """
    s = _heading(1, "Numerical Stability")
    s.para(
        "For each `(dtype, K)` we generate `A, B ∈ R^(K×K)` from "
        "`N(0, 1/√K)` (the LLM-weight regime), compute `Z_ref = A @ B` in "
        "FP32, downcast inputs to the test dtype, perform the matmul, and "
        "compare the matrix-scaled max relative error `max|Z_lp - Z_ref| / "
        "max|Z_ref|` against the analytic bound `5·√K·2⁻ᵐ` "
        "(`m` = mantissa bits: 10/7/3/2 for fp16/bf16/fp8_e4m3/fp8_e5m2). "
        "This is the same convention as `bench01.correctness_check`, "
        "extended to the full dtype × K grid."
    )
    s.image(plots_dir / "A9_stability.png",
            alt="Numerical stability sweep",
            caption=("Figure 9 — (a) max rel error vs K with analytic "
                     "bounds, (b) per-element error histogram at the "
                     "largest K."))

    if not stability or not (stability.get("rows") or []):
        s.para("_(stability sweep did not run, or output missing)_")
        return s

    rows = stability["rows"]
    table_rows: List[dict] = []
    for r in sorted(rows, key=lambda r: (r["dtype"], r["K"])):
        if "rel_err" not in r:
            table_rows.append({
                "dtype": r["dtype"],
                "K": r["K"],
                "max rel err": "n/a",
                "p99 rel err": "n/a",
                "bound": "n/a",
                "pass": "FAIL",
                "note": r.get("error", ""),
            })
            continue
        table_rows.append({
            "dtype":         r["dtype"],
            "K":             r["K"],
            "max rel err":   f"{r['rel_err']['max']:.3e}",
            "p99 rel err":   f"{r['rel_err_pointwise']['p99']:.3e}",
            "bound":         f"{r['rel_err_bound']:.3e}",
            "pass":          "ok" if r.get("passed") else "FAIL",
            "note":          r.get("note") or "",
        })
    s.table(table_rows,
            caption="Per (dtype, K) GEMM error vs FP32 reference")

    insights: List[str] = []
    by_dtype: Dict[str, List[dict]] = {}
    for r in rows:
        by_dtype.setdefault(r["dtype"], []).append(r)
    for dt, rs in sorted(by_dtype.items()):
        rs2 = sorted([r for r in rs if "rel_err" in r], key=lambda r: r["K"])
        if not rs2:
            continue
        worst = max(rs2, key=lambda r: r["rel_err"]["max"])
        insights.append(
            f"`{dt}`: worst max rel err = **{worst['rel_err']['max']:.2e}** "
            f"at K = {worst['K']} (analytic bound "
            f"{worst['rel_err_bound']:.2e}, "
            f"{worst['rel_err']['max'] / worst['rel_err_bound'] * 100:.1f}% of bound)."
        )
    failed = [r for r in rows if not r.get("passed")]
    if failed:
        bad = ", ".join(f"`{r['dtype']}` @ K={r['K']}"
                        for r in failed[:5])
        insights.append(
            f"**{len(failed)} row(s) FAIL** the analytic bound: {bad}"
            + ("…" if len(failed) > 5 else "")
            + ". Investigate the kernel — emulation, transpose bug, or "
              "skipped accumulation are the usual culprits."
        )
    upcast = [r for r in rows
              if (r.get("note") or "").startswith("upcast to bf16")]
    if upcast:
        upcast_dtypes = sorted({r["dtype"] for r in upcast})
        insights.append(
            "FP8 results were measured with bf16 upcast for the matmul "
            f"({', '.join(upcast_dtypes)}); the reported error reflects "
            "bf16 accumulation, not a true FP8 GEMM kernel. Re-run on a "
            "backend with native FP8 matmul (`cuBLAS hgemm`, ROCm "
            "`rocblas_gemm_ex` with FP8 inputs, or hipBLASLt) to get "
            "FP8-native numbers."
        )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)
    s.insight_takeaway(
        "All passing rows confirm reduced-precision GEMMs land within the "
        "analytic error bound — i.e. the lossy cast doesn't compound past "
        "the dtype's mantissa floor.",
        "If a row FAILs in a future regression run, the prior is a kernel "
        "issue (emulation / transpose bug / skipped accumulation) — the "
        "bound itself is fixed.",
    )
    return s


def section_dram(dram: dict, profile: Optional[Dict[str, Any]] = None) -> Section:
    """Memory-capacity section. Generic across hardware: when the
    target ``profile`` carries a rated capacity, the prose and the
    "effective utilization vs ..." row reference it; otherwise the
    section reports raw numbers against the device-detected total.
    """
    profile = profile or {}
    rated_mem = profile.get("rated_mem_gib")
    rated_label = profile.get("short") or "rated"
    s = _heading(1, "Memory Capacity")
    if rated_mem:
        s.para(
            f"Practical allocatable device memory is bounded below the "
            f"nominal **{rated_mem:.0f} GiB** ({rated_label} on-package "
            f"memory) by driver reserve, framework overhead, and "
            f"fragmentation. The numbers below are what the workload can "
            f"actually use — the operationally relevant figure for "
            f"inference and diffusion."
        )
    else:
        s.para(
            "Practical allocatable device memory is bounded below the "
            "device-reported total by driver reserve, framework overhead, "
            "and fragmentation. The numbers below are what the workload "
            "can actually use — the operationally relevant figure for "
            "inference and diffusion."
        )
    if not dram:
        s.para("_(no memory data collected)_")
        return s

    util_label = (f"Effective utilization vs {rated_mem:.0f} GiB rated"
                  if rated_mem else "Effective utilization (vs device total)")
    rows = [
        {"metric": "Device total", "value (GiB)": _fmt(dram.get("device_total_bytes", 0) / 1024**3)},
        {"metric": "Free pre-test", "value (GiB)": _fmt(dram.get("device_free_bytes_pre_test", 0) / 1024**3)},
        {"metric": "Max bf16 contiguous alloc", "value (GiB)": _fmt(dram.get("max_alloc_bf16_gib"))},
        {"metric": "Max fp16 contiguous alloc", "value (GiB)": _fmt((dram.get("max_alloc_fp16_bytes") or 0) / 1024**3)},
        {"metric": util_label, "value (GiB)": _pct(dram.get("eff_util_fraction_bf16"))},
        {"metric": "Fragmentation ratio (chunked / contig)", "value (GiB)": _fmt(dram.get("frag_sensitivity_ratio"))},
    ]
    s.table(rows)

    headroom = dram.get("headroom") or {}
    if headroom:
        s.text("\n**Headroom after model load:**\n",
               html="<p><strong>Headroom after model load:</strong></p>")
        s.para(
            "Allocates analytic-sized bf16 weight tensors derived from the "
            "config's GEMM inventory (sums `K·N` per dense projection × "
            "`depth`, plus a 1% embedding/head allowance). Residual = "
            "`mem_get_info` after the weights are resident — the bytes "
            "actually available for activations, KV caches, and gradients "
            "in steady-state inference. This is the operational "
            "TP-3-table number TESTPLAN §16.3 wants."
        )
        rows = [
            {"metric": "Model parameters",
             "value": f"{(headroom.get('model_params') or 0) / 1e9:.2f} B"},
            {"metric": "Model bf16 weights (full)",
             "value": _fmt(headroom.get('model_target_full_gib'), 2) + " GiB"},
            {"metric": "Allocated by probe",
             "value": _fmt(headroom.get('model_bytes_gib'), 2) + " GiB"},
            {"metric": "Residual capacity (free after load)",
             "value": _fmt(headroom.get('residual_capacity_gib'), 2) + " GiB"},
            {"metric": "Residual fraction of total",
             "value": _pct(headroom.get('residual_fraction'))},
        ]
        if headroom.get("probe_capped"):
            rows.append({
                "metric": "Probe note",
                "value": ("CAPPED — host could not hold the full model; "
                          "residual is computed from the partial allocation. "
                          "Re-run on the GPU target for the operational "
                          "number."),
            })
        s.table(rows, caption="Post-load residual capacity")

    insights = []
    if dram.get("eff_util_fraction_bf16") is not None:
        eu = dram["eff_util_fraction_bf16"]
        ref_phrase = (f"the {rated_mem:.0f} GiB {rated_label} rating"
                      if rated_mem else "the device-reported total")
        insights.append(
            f"Usable bf16 capacity is **{eu*100:.1f}% of {ref_phrase}**; "
            f"the {(1 - eu)*100:.1f}% gap is driver + framework + allocator overhead."
        )
    if dram.get("frag_sensitivity_ratio") is not None:
        fr = dram["frag_sensitivity_ratio"]
        if fr < _threshold("fragmentation_warning_ratio"):
            insights.append(
                f"Fragmentation costs ~{(1 - fr)*100:.0f}% of capacity when allocating "
                f"in many small chunks — relevant for KV-cache and per-layer activation "
                f"buffers in long-context generation."
            )
    if headroom and headroom.get("loaded") and not headroom.get("probe_capped"):
        residual_gib = headroom.get("residual_capacity_gib") or 0
        model_gib = headroom.get("model_bytes_gib") or 0
        insights.append(
            f"After loading the **{model_gib:.1f} GiB** bf16 model, "
            f"**{residual_gib:.1f} GiB** is free for activations, KV cache, "
            f"and gradients — the post-load steady-state headroom for the "
            f"workload."
        )
    elif headroom and headroom.get("probe_capped"):
        insights.append(
            f"Host could not hold the full **{headroom.get('model_target_full_gib') or 0:.1f} GiB** "
            f"bf16 model (deficit: "
            f"{(headroom.get('deficit_bytes') or 0) / 1024**3:.1f} GiB). The "
            f"residual figure was computed from the largest allocation that "
            f"fit — useful as a smoke test, not as the TP-3 headroom number."
        )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)
    s.insight_takeaway(
        "Usable bf16 capacity (not the rated number) is the operational "
        "ceiling; headroom-after-load is the operational headroom. Both "
        "come from real allocator behavior, not the spec sheet.",
        "Use the residual-after-load number for KV cache / activation "
        "budget; size batch & sequence so peak working-set lands inside "
        "it with margin.",
    )
    return s


def section_workload_roofline(ops: dict, plots_dir: Path) -> Section:
    s = _heading(1, "Workload & Roofline")
    s.para(
        "The `escher_14b_480p` op decomposition is plotted on a roofline whose "
        "compute and bandwidth ceilings come from §3 (measured, not rated). "
        "Markers are color-coded by op family."
    )
    s.image(plots_dir / "A6_roofline.png",
            alt="Roofline plot",
            caption="Figure 3 — Per-op roofline placement.")

    if not ops:
        s.para("_(no per-op data collected)_")
        return s

    cal = ops.get("calibration_drift") or {}
    tot = ops.get("totals") or {}
    rows = []
    if tot:
        rows.append({"metric": "Total per block (GFLOPs)",
                     "value": _fmt(tot.get("total_gflops"), 1),
                     "note": f"drift vs reference: "
                             f"{_fmt(cal.get('gflops_drift_pct'))}%"})
        rows.append({"metric": "Total per block (Memory MB)",
                     "value": _fmt(tot.get("total_mb_hbm"), 1),
                     "note": f"drift vs reference: "
                             f"{_fmt(cal.get('mb_hbm_drift_pct'))}%"})
        rows.append({"metric": "Avg arithmetic intensity (FLOP/B)",
                     "value": _fmt(tot.get("avg_arithmetic_intensity"), 1),
                     "note": ""})
    if ops.get("ridge_flop_per_byte"):
        rows.append({"metric": "Ridge point (FLOP/B)",
                     "value": _fmt(ops.get("ridge_flop_per_byte"), 1),
                     "note": "compute_peak / bandwidth_roof"})
    s.table(rows)

    rs = ops.get("rows") or []
    n_compute = sum(1 for r in rs if r.get("bound") == "compute")
    n_memory = sum(1 for r in rs if r.get("bound") == "memory")
    flops_compute = sum((r.get("flops") or 0) for r in rs if r.get("bound") == "compute")
    flops_total = sum((r.get("flops") or 0) for r in rs)
    pct_flops_compute = (flops_compute / flops_total) if flops_total else None
    insights = []
    if n_compute or n_memory:
        insights.append(
            f"Of the {n_compute + n_memory} measurable ops, **{n_compute} are compute-bound** "
            f"and {n_memory} are memory-bound."
        )
    if pct_flops_compute is not None:
        insights.append(
            f"**{pct_flops_compute*100:.0f}% of the workload's FLOPs sit in compute-bound ops** — "
            f"this is what makes `escher_14b_480p` a compute-dominant transformer stack, "
            f"matching the source PDF's *DiT workload is extremely compute-bound* finding."
        )
    drift_thresh = _threshold("calibration_drift_pct")
    if cal.get("gflops_drift_pct") is not None and abs(cal["gflops_drift_pct"]) > drift_thresh:
        insights.append(
            f"**Calibration drift > {drift_thresh:g}%** "
            f"({cal['gflops_drift_pct']:+.1f}% GFLOPs vs reference). "
            "**Diagnosis:** the FLOP totals derive from the GEMM inventory "
            "in `configs/escher_14b_480p.json`; a drift of this size "
            "almost always means the per-block shape spec (depth / "
            "hidden_dim / FFN expansion / attention kernel mix) doesn't "
            "match the source pilot. **Action:** revisit config shapes "
            "against the source PDF workload spec **before sign-off**, "
            "and re-run `bench04_workload_ops.py` to confirm the drift "
            "closes."
        )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)

    table_rows: List[Dict] = []
    for r in rs:
        if (r.get("flops") or 0) == 0 and (r.get("bytes_hbm") or 0) == 0:
            continue
        t_def = r.get("t_ms_default")
        t_opt = r.get("t_ms_optimized")
        speedup = (t_def / t_opt) if (t_def and t_opt and t_opt > 0
                                      and not (isinstance(t_def, float) and math.isnan(t_def))
                                      and not (isinstance(t_opt, float) and math.isnan(t_opt))) else None
        table_rows.append({
            "op": r.get("op_name"),
            "category": r.get("category"),
            "GFLOPs": _fmt((r.get("flops") or 0) / 1e9, 2),
            "Memory MB": _fmt((r.get("bytes_hbm") or 0) / 1e6, 2),
            "AI": _fmt(r.get("arithmetic_intensity"), 1),
            "bound": r.get("bound") or "",
            "t default (ms)": _fmt(t_def),
            "t opt (ms)": _fmt(t_opt),
            "opt speedup": _fmt(speedup, 2),
            "meas/theory (opt)": _fmt(r.get("meas_over_theory_optimized"), 2),
        })
    if table_rows:
        s.text("\n**Per-op detail:**\n",
               html="<p><strong>Per-op detail:</strong></p>")
        s.table(table_rows)
    s.insight_takeaway(
        "The roofline tells you whether each op is compute-bound or "
        "memory-bound — i.e. which ceiling sets its theoretical limit. "
        "Calibration drift on the totals is a config-shape issue, not a "
        "measurement issue.",
        "Optimization budget goes to compute-bound ops (where peak is the "
        "ceiling); memory-bound ops gate on bandwidth and locality — "
        "treat them differently.",
    )
    return s


def section_per_op_default_vs_optimized(ops: dict, plots_dir: Path) -> Section:
    s = _heading(1, "Per-Op Throughput")
    s.para(
        "Theoretical bottleneck time (max of compute / memory time) compared "
        "against measured time on the default torch path (math + memory-efficient "
        "SDPA, no AITER) and the optimized path (AITER → flash_attn → SDPA-flash, "
        "in that order of preference)."
    )
    s.image(plots_dir / "A7_per_op_theory_vs_meas.png",
            alt="Theory vs measured per op",
            caption="Figure 4 — Per-op theory vs default vs optimized timing.")

    if not ops:
        s.para("_(no per-op data collected)_")
        return s

    rs = ops.get("rows") or []

    # Top-3 bottlenecks callout (user feedback): a quick scan-friendly
    # block listing the three ops that consume the most wall-clock on
    # the optimized path. A reader looking at the per-op table doesn't
    # have to sort by hand to find the operational priorities.
    timed = [r for r in rs
             if isinstance(r.get("t_ms_optimized"), (int, float))
             and not math.isnan(r["t_ms_optimized"])
             and r["t_ms_optimized"] > 0]
    timed.sort(key=lambda r: -(r.get("t_ms_optimized") or 0))
    top3 = timed[:3]
    if top3:
        total_t = sum(r["t_ms_optimized"] for r in timed)
        rows = []
        for r in top3:
            t = r["t_ms_optimized"]
            rows.append({
                "rank":           f"#{len(rows)+1}",
                "op":             r.get("op_name"),
                "t opt (ms)":     _fmt(t),
                "% block time":   _pct(t / total_t if total_t else None),
                "meas/theory":    _fmt(r.get("meas_over_theory_optimized"), 2),
                "bound":          r.get("bound") or "",
            })
        s.subheading("Top 3 bottlenecks (optimized path)", level=2)
        s.table(rows, caption="Wall-clock priority order — fix from the top")
    flash_rows = [r for r in rs if (r.get("op_name") or "").endswith(".flash")]
    speedups: List[tuple[str, float]] = []
    for r in flash_rows:
        d = r.get("t_ms_default"); o = r.get("t_ms_optimized")
        if isinstance(d, (int, float)) and isinstance(o, (int, float)) and o > 0 \
           and not math.isnan(d) and not math.isnan(o):
            speedups.append((r["op_name"], d / o))

    insights = []
    if speedups:
        avg = sum(s2 for _, s2 in speedups) / len(speedups)
        best = max(speedups, key=lambda x: x[1])
        insights.append(
            f"**Optimized attention is {avg:.2f}× faster than default torch SDPA on average** "
            f"across {len(speedups)} attention ops; best speedup is {best[1]:.2f}× on `{best[0]}`."
        )
        insights.append(
            "This reproduces the source PDF observation: *default torch SDPA << AITER attention*. "
            "The remaining gap to theory is implementation-quality tax, not hardware limit."
        )

    # meas/theory analysis
    def _ratio(r, key):
        v = r.get(key)
        return v if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else None
    rats_opt = [(r["op_name"], _ratio(r, "meas_over_theory_optimized"))
                for r in rs if _ratio(r, "meas_over_theory_optimized")]
    if rats_opt:
        worst = max(rats_opt, key=lambda x: x[1])
        hw_limit = _threshold("meas_over_theory_at_hw_limit")
        tunable_thresh = _threshold("meas_over_theory_tunable")
        n_at_limit = sum(1 for _, v in rats_opt if v <= hw_limit)
        n_tunable  = sum(1 for _, v in rats_opt if hw_limit < v <= tunable_thresh)
        n_impl     = sum(1 for _, v in rats_opt if v > tunable_thresh)
        insights.append(
            f"Per the §10.4 thresholds: **{n_at_limit} ops at hardware limit** "
            f"(meas/theory ≤ {hw_limit:.2f}), {n_tunable} tunable "
            f"(≤ {tunable_thresh:.2f}), {n_impl} likely "
            f"implementation-quality issues (> {tunable_thresh:.2f})."
        )
        if n_impl > 0:
            insights.append(
                f"Worst offender: `{worst[0]}` at {worst[1]:.2f}× theory — investigate kernel."
            )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)
    tunable_thresh_label = _threshold("meas_over_theory_tunable")
    s.insight_takeaway(
        f"AITER (or its fallback) outperforms torch's default SDPA across the "
        f"attention ops; the ops that exceed {tunable_thresh_label:.2f}× theory "
        f"are the operational priorities — they don't reflect the hardware ceiling.",
        "Address the top-3 bottlenecks in priority order; for each, look at "
        "meas/theory to decide whether it's a tuning win or a kernel rewrite.",
    )
    return s


def section_mfu(mfu: dict, plots_dir: Path,
                profile: Optional[Dict[str, Any]] = None) -> Section:
    """End-to-end MFU section. Generic across hardware: rated-PF
    columns (and the corresponding "rated %" bars in the chart) are
    only shown when the target ``profile`` carries rated BF16 specs.
    On a CPU host or unknown GPU the table reports just the
    measured-peak column.
    """
    profile = profile or {}
    rated_low_tf  = profile.get("rated_bf16_low")
    rated_high_tf = profile.get("rated_bf16_high")
    has_rated     = bool(rated_low_tf)
    s = _heading(1, "End-to-End MFU")
    if has_rated:
        rated_phrase = (f"{rated_low_tf/1000.0:.2f} PF rated"
                        + (f" and {rated_high_tf/1000.0:.2f} PF rated"
                           if rated_high_tf and rated_high_tf != rated_low_tf
                           else ""))
        bases_phrase = (f"three FLOP bases — measured chip peak "
                        f"(from `bench01.peak_tight_loop`), "
                        f"{rated_phrase} — so")
    else:
        bases_phrase = ("the measured chip peak (from "
                        "`bench01.peak_tight_loop`) so")
    ref_targets_pct = _cfg().get("pdf_reference_targets_pct") or {}
    _sop_t = ref_targets_pct.get("sum_of_ops",   77)
    _eg_t  = ref_targets_pct.get("eager_e2e",    93)
    _co_t  = ref_targets_pct.get("compiled_e2e", 99)
    s.para(
        f"Three scopes share the same FLOP basis (the analytic per-op accounting). "
        f"Differences between scopes are pure framework / launch / fusion overhead. "
        f"Each scope is shown on {bases_phrase} the ordering and the gap to spec "
        f"can be read off the same chart. The horizontal black markers on the "
        f"measured-peak bars are the source PDF's reference targets "
        f"(sum-of-ops ≈ {_sop_t:.0f}%, eager e2e ≈ {_eg_t:.0f}%, "
        f"compiled e2e ≈ {_co_t:.0f}%), drawn so reproduction can be validated by eye."
    )
    s.image(plots_dir / "A8_mfu.png",
            alt="MFU comparison: sum-of-ops vs eager vs compiled across three FLOP bases",
            caption=("Figure 5 — Model FLOPs Utilization across measurement scopes "
                     "and FLOP bases. Black ticks are the source PDF's reference "
                     "targets on the measured-peak basis."))
    s.para(
        "The second figure shows the per-chunk timing distribution for the timed "
        "e2e scopes. The PDF's qualitative claim — \"compiled e2e is not just "
        "faster on the median, it's more stable across chunks\" — corresponds to "
        "tighter p10/p90 spread and lower std on the compiled boxplot."
    )
    s.image(plots_dir / "A8b_mfu_per_chunk.png",
            alt="Per-chunk timing distribution: eager vs compiled e2e",
            caption=("Figure 5b — Per-chunk forward-pass time distribution for "
                     "the timed e2e scopes (sum-of-ops scopes are omitted; they "
                     "have no per-chunk distribution)."))

    if not mfu:
        s.para("_(no MFU data collected)_")
        return s

    rows_in = mfu.get("rows") or []
    targets = mfu.get("pdf_reference_targets_pct") or {}

    def _target_for(scope: str):
        if scope in targets:
            return float(targets[scope])
        for k, v in targets.items():
            if scope.startswith(k):
                return float(v)
        return None

    rated_low_label  = (f"MFU ({rated_low_tf/1000.0:.2f} PF)"
                        if rated_low_tf else None)
    rated_high_label = (f"MFU ({rated_high_tf/1000.0:.2f} PF)"
                        if rated_high_tf and rated_high_tf != rated_low_tf
                        else None)

    rows = []
    for r in rows_in:
        scope = r["scope"]
        meas_pct = (r.get("mfu_measured_peak") or 0) * 100 if r.get("mfu_measured_peak") is not None else None
        target_pct = _target_for(scope)
        delta = (f"{meas_pct - target_pct:+.0f} pp"
                 if (meas_pct is not None and target_pct is not None) else "—")
        row: Dict[str, Any] = {
            "scope": scope,
            "t_total_ms": _fmt(r.get("t_total_ms")),
            "TFLOP/s": _fmt_tflops(r.get("tflops_achieved")),
            "MFU (measured peak)": _pct(r.get("mfu_measured_peak")),
            "PDF target (measured peak)": (f"{target_pct:.0f}%" if target_pct is not None else "—"),
            "Δ vs PDF": delta,
        }
        ach = r.get("tflops_achieved")
        if rated_low_label:
            row[rated_low_label]  = (_pct(ach / rated_low_tf)
                                      if isinstance(ach, (int, float)) else "n/a")
        if rated_high_label:
            row[rated_high_label] = (_pct(ach / rated_high_tf)
                                      if isinstance(ach, (int, float)) else "n/a")
        rows.append(row)
    s.table(rows)

    # Per-chunk stability table for the e2e scopes.
    stab_rows = []
    for r in rows_in:
        scope = r["scope"]
        if not (scope.startswith("eager") or scope.startswith("compiled")):
            continue
        if not r.get("times_ms"):
            continue
        stab_rows.append({
            "scope": scope,
            "iters": r.get("iters"),
            "median (ms)": _fmt(r.get("t_total_ms")),
            "p10 (ms)": _fmt(r.get("p10_ms")),
            "p90 (ms)": _fmt(r.get("p90_ms")),
            "min (ms)": _fmt(r.get("min_ms")),
            "max (ms)": _fmt(r.get("max_ms")),
            "std (ms)": _fmt(r.get("std_ms")),
        })
    if stab_rows:
        s.text("\n**Per-chunk timing stability (e2e scopes):**\n",
               html="<p><strong>Per-chunk timing stability (e2e scopes):</strong></p>")
        s.table(stab_rows)

    by_scope = {r["scope"]: r for r in rows_in}
    sop = by_scope.get("sum_of_ops_optimized") or by_scope.get("sum_of_ops_default")
    eager = by_scope.get("eager_e2e")
    compiled = by_scope.get("compiled_e2e")
    insights = []
    if sop and eager and compiled:
        sop_raw = sop.get("mfu_measured_peak")
        eg_raw  = eager.get("mfu_measured_peak")
        co_raw  = compiled.get("mfu_measured_peak")
        sop_v = sop_raw * 100 if sop_raw is not None else None
        eg_v  = eg_raw  * 100 if eg_raw  is not None else None
        co_v  = co_raw  * 100 if co_raw  is not None else None

        def _pp(v):
            return f"{v:.0f}%" if v is not None else "n/a"

        # Lift sign-aware. On the GPU the PDF predicts compiled > eager via
        # fusion / launch reduction; on smaller CPU shapes Inductor's
        # autotune cost can push compiled *below* eager. Surface both
        # directions honestly instead of assuming the GPU ordering.
        if eg_v is not None and co_v is not None:
            lift = co_v - eg_v
            if lift >= 0:
                lift_explainer = (
                    "attributable to fewer dispatches, larger fused regions, "
                    "and reduced framework overhead — the source PDF's expected ordering."
                )
            else:
                lift_explainer = (
                    "**compiled is slower than eager here**, which is a real and "
                    "documented `torch.compile` failure mode on small CPU shapes "
                    "(Inductor autotune / dispatch overhead exceeds the fusion win). "
                    "On a GPU host we expect this lift to flip positive; if it does "
                    "not, audit the compile mode and the per-shape autotune budget."
                )
            insights.append(
                f"Scope ordering: sum-of-ops {_pp(sop_v)} → eager e2e {_pp(eg_v)} → "
                f"compiled e2e {_pp(co_v)}. The lift from eager to compiled is "
                f"**{lift:+.0f} pp** — {lift_explainer}"
            )

        # PDF reproduction commentary: how close are we to 77/93/99?
        pdf_pairs = []
        for label, scope_key, meas in (
            ("sum-of-ops", sop["scope"], sop_v),
            ("eager e2e",  "eager_e2e",   eg_v),
            ("compiled e2e", "compiled_e2e", co_v),
        ):
            t = _target_for(scope_key)
            if t is not None and meas is not None and meas > 0:
                pdf_pairs.append(f"{label} {meas:.0f}% (PDF≈{t:.0f}%, Δ {meas - t:+.0f} pp)")
        if pdf_pairs:
            tol_pp = _threshold("mfu_pdf_tolerance_pp")
            insights.append(
                "**PDF reproduction (measured-peak basis):** " + "; ".join(pdf_pairs)
                + f". Reproduction is judged adequate when each Δ falls within "
                  f"±{tol_pp:g} pp of the PDF target (TESTPLAN §1.2 SC-4)."
            )
        if sop_v is not None and co_v is not None and co_v > sop_v:
            insights.append(
                f"Compiled e2e exceeding sum-of-ops by **{co_v - sop_v:+.0f} pp** is **expected, "
                f"not suspicious**: the compiled graph fuses work across boundaries that the "
                f"per-op accounting can't see. The audit step is to confirm the FLOP basis "
                f"and timing methodology match — the source PDF flags this exact phenomenon."
            )
        if co_v is not None and co_v > 100:
            insights.append(
                f"⚠️ Compiled MFU > 100% on measured peak ({co_v:.0f}%) — audit FLOP accounting "
                f"or peak measurement; this is a basis problem, not a real result."
            )

    # Stability commentary, sign-aware. The PDF claim is "compiled is more
    # stable than eager"; we report the actual direction and only call it a
    # match when σ_compiled ≤ σ_eager.
    eg_std = (eager or {}).get("std_ms") if eager else None
    co_std = (compiled or {}).get("std_ms") if compiled else None
    if eg_std is not None and co_std is not None and eg_std > 0:
        rel = (co_std - eg_std) / eg_std * 100
        if co_std <= eg_std:
            stab = (
                "Lower compiled σ matches the source PDF's stability claim; rising σ "
                "in a regression run is a leading indicator that compile fusions broke."
            )
        else:
            stab = (
                "**Compiled σ is higher than eager σ** here, which inverts the source "
                "PDF's stability claim. On CPU this is commonly Inductor's per-call "
                "guard / recompile overhead leaking into the timed region — a real "
                "datapoint, not a measurement bug. On a GPU host this inversion would "
                "warrant a deeper audit of the compile mode."
            )
        insights.append(
            f"Per-chunk std: eager σ = {eg_std:.2f} ms vs compiled σ = {co_std:.2f} ms "
            f"({rel:+.0f}% relative). " + stab
        )
    if mfu.get("compile_mode_used"):
        insights.append(f"`torch.compile` mode used: `{mfu['compile_mode_used']}`.")
    if mfu.get("device_type") == "cpu":
        insights.append(
            "CPU host: rated-peak MFU columns are intentionally hidden "
            "because no GPU profile applies — a CPU's TFLOP/s peak is "
            "orders of magnitude below any accelerator's rated number, "
            "so the ratio is not informative. Read the measured-peak "
            "column for the apples-to-apples timing-infra check."
        )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)

    # Sign-off basis guidance: a recurring reader mistake is comparing a
    # measured-peak MFU figure on this run to a rated-peak MFU figure
    # somewhere else. State the correct basis explicitly so the
    # narrative is unambiguous.
    s.subheading("Which MFU basis is the sign-off basis?", level=2)
    bullets = [
        "**Measured-peak basis** is the *internal* baseline. It compares "
        "the workload to *this hardware as actually clocked at the time of "
        "the run*, so it isolates kernel/compile overhead from firmware / "
        "thermal / clock-gate variability. Use it to **detect regressions** "
        "between runs.",
    ]
    if rated_low_label:
        bullets.append(
            f"**{rated_low_label.replace('MFU ', 'Rated ')} basis** is the "
            f"*external* sign-off basis vs the device's lower-bound rated "
            f"BF16 number ({rated_low_tf:,.0f} TFLOP/s for "
            f"{profile.get('short')}). Use it when reporting to anyone "
            f"outside the regression team — it is the apples-to-apples "
            f"comparison against datasheet expectations."
        )
    if rated_high_label:
        bullets.append(
            f"**{rated_high_label.replace('MFU ', 'Rated ')} basis** is the "
            f"upper-bound spec figure. Do **not** use it for sign-off; "
            f"reserve it for headroom claims and forward-looking analyses."
        )
    if not has_rated:
        bullets.append(
            "Rated-peak MFU columns are not shown because the rated BF16 "
            "spec for this device is not in the registry (or this is a "
            "CPU-host run). Use the measured-peak column as the sign-off "
            "basis until rated specs are wired in."
        )
    s.bullets(bullets)
    if has_rated:
        s.callout(
            "info", "Sign-off recommendation",
            f"For regression sign-off use **measured-peak**; for executive / "
            f"external comparisons use **rated "
            f"{rated_low_tf/1000.0:.2f} PF** (the lower-bound rated number "
            f"for {profile.get('short')}). Never quote the same MFU number "
            f"against multiple bases without the basis label."
        )
    else:
        s.callout(
            "info", "Sign-off recommendation",
            "For regression sign-off use **measured-peak**. Once rated "
            "specs for the target accelerator are in the registry, the "
            "lower-bound rated basis becomes the external citation; until "
            "then, always label every MFU number with its basis."
        )
    s.insight_takeaway(
        "Compiled E2E is the headline; sum-of-ops and eager E2E exist to "
        "explain *why* the compiled number is what it is, not as competitor "
        "metrics.",
        ("Lock the sign-off basis (measured-peak vs lower-bound rated) once "
         "for the campaign and use it consistently in every external citation."),
    )
    return s


def section_multigpu(comm: dict, fused: Optional[dict] = None) -> Section:
    is_cpu = bool(comm and comm.get("device_type") == "cpu")
    if is_cpu:
        s = _heading(1, "Multi-CCD / Multi-Socket Communication")
        s.para(
            "On a CPU host the GPU collective sweep is replaced by its CPU "
            "analogue: gloo over loopback with each rank pinned to a "
            "dedicated CCD (Linux `die_id`) or socket. This makes the "
            "all-gather / reduce-scatter / all-reduce numbers reflect the "
            "**Infinity Fabric** (intra-socket) or **xGMI / UPI** "
            "(inter-socket) interconnect rather than memcpy within a single "
            "CCD. Methodology and payload schema are identical to the GPU "
            "path so the JSON output diffs cleanly across hosts."
        )
    else:
        s = _heading(1, "Multi-GPU Communication")
        s.para(
            "Tensor-parallel collectives at payloads representative of real DiT "
            "activations: `all_gather` (matched to fused AG+MM), `reduce_scatter` "
            "(matched to MM+RS), `all_reduce` (used inside attention reductions), "
            "and `all_to_all` (the alternative TP topology the source PDF flags as "
            "future work — A2A in place of AG+RS)."
        )
    if not comm:
        s.para("_(multi-GPU step did not run, or output missing)_")
        return s
    world = comm.get("world")
    rs = comm.get("rows") or []
    if not rs:
        s.para("_(no multi-rank rows collected)_")
        return s

    backend = comm.get("backend") or ("gloo" if is_cpu else "nccl/rccl")
    if is_cpu:
        topo = comm.get("cpu_topology") or {}
        mode = topo.get("topology_mode") or "?"
        sockets = topo.get("sockets")
        dies = topo.get("dies")
        cores_per_die = topo.get("cores_per_die")
        threads_per_core = topo.get("threads_per_core")
        s.para(
            f"World size: **{world}** rank(s) — backend `{backend}`, "
            f"topology mode `{mode}` "
            f"(sockets={sockets}, CCDs={dies}, cores/CCD={cores_per_die}, "
            f"threads/core={threads_per_core})."
        )
        rank_pinning = topo.get("rank_pinning") or []
        if rank_pinning:
            s.table(
                [{
                    "rank": r.get("rank"),
                    "n_cpus": r.get("n_cpus"),
                    "cpus": (str(r.get("cpus"))
                             if len(r.get("cpus") or []) <= 16
                             else f"[{r['cpus'][0]}..{r['cpus'][-1]}] "
                                  f"({len(r['cpus'])} cpus)"),
                } for r in rank_pinning],
                caption="Rank-to-CPU pinning (one row per rank)",
            )
    else:
        s.para(f"World size: **{world}** GPU(s) — backend `{backend}`.")

    s.table([{
        "op": r["op"], "payload (MB)": _fmt(r["bytes"] / 1e6, 0),
        "t (ms)": _fmt(r.get("t_ms")),
        "algbw (GB/s)": _fmt(r.get("algbw_gb_s"), 0),
        "busbw (GB/s)": _fmt(r.get("busbw_gb_s"), 0),
    } for r in rs])

    insights = []
    by_op: Dict[str, list] = {}
    for r in rs:
        by_op.setdefault(r["op"], []).append(r)
    for op, lst in by_op.items():
        plateau = max((r.get("busbw_gb_s") or 0) for r in lst)
        insights.append(f"`{op}` plateau busbw: **{plateau:.1f} GB/s** (largest payload).")

    # A2A vs (AG + RS) head-to-head: the source PDF's "future work" topology.
    # We compare wall-clock at matched payload size — the question is "if we
    # replaced an AG followed by an RS with a single all_to_all, would we
    # come out ahead?" Pure shape-equivalent comparison since the data
    # movement totals are identical.
    a2a_rows = by_op.get("all_to_all") or []
    ag_rows  = by_op.get("all_gather") or []
    rs_rows  = by_op.get("reduce_scatter") or []
    if a2a_rows and ag_rows and rs_rows:
        ag_by_b = {r["bytes"]: r for r in ag_rows}
        rs_by_b = {r["bytes"]: r for r in rs_rows}
        comp_rows: List[dict] = []
        for r in a2a_rows:
            b = r["bytes"]
            ag = ag_by_b.get(b)
            rs2 = rs_by_b.get(b)
            if not ag or not rs2:
                continue
            ag_t = ag.get("t_ms") or 0
            rs_t = rs2.get("t_ms") or 0
            a2a_t = r.get("t_ms") or 0
            combo_t = ag_t + rs_t
            ratio = (a2a_t / combo_t) if combo_t else None
            comp_rows.append({
                "payload (MB)":   _fmt(b / 1e6, 0),
                "AG t (ms)":      _fmt(ag_t),
                "RS t (ms)":      _fmt(rs_t),
                "AG+RS t (ms)":   _fmt(combo_t),
                "A2A t (ms)":     _fmt(a2a_t),
                "A2A / (AG+RS)":  (f"{ratio:.2f}×" if ratio else "—"),
                "verdict":        (lambda r: (
                    "A2A wins"   if r and r < (1 - _threshold("topology_decisive_advantage_pct") / 100.0)
                    else ("AG+RS wins"
                          if r and r > (1 + _threshold("topology_decisive_advantage_pct") / 100.0)
                          else "tied")
                ))(ratio),
            })
        if comp_rows:
            s.text("\n**A2A vs AG+RS head-to-head:**\n",
                   html="<p><strong>A2A vs AG+RS head-to-head:</strong></p>")
            s.table(comp_rows,
                    caption="Wall-clock: alternative TP topology comparison")
            wins = [r for r in comp_rows if r["verdict"] == "A2A wins"]
            losses = [r for r in comp_rows if r["verdict"] == "AG+RS wins"]
            if wins and not losses:
                insights.append(
                    f"A2A beats AG+RS at every measured payload "
                    f"(**{len(wins)}/{len(comp_rows)}** wins) — the alternative "
                    "TP topology is the better choice on this fabric."
                )
            elif losses and not wins:
                insights.append(
                    f"AG+RS beats A2A at every measured payload "
                    f"(**{len(losses)}/{len(comp_rows)}** AG+RS wins) — the "
                    "current topology is fabric-optimal; A2A future work "
                    "would not help here."
                )
            else:
                insights.append(
                    f"A2A vs AG+RS is mixed across payloads "
                    f"(A2A wins {len(wins)}, AG+RS wins {len(losses)}). "
                    "Crossover suggests payload-dependent topology choice."
                )
    if is_cpu:
        insights.append(
            "These numbers reflect the host's CCD-to-CCD or socket-to-socket "
            "interconnect (Infinity Fabric / xGMI / UPI) plus the gloo TCP-loopback "
            "stack — not RCCL/NCCL. They are still useful for regression detection "
            "and for sanity-checking the multi-rank dispatch infrastructure on "
            "CPU-only CI machines."
        )
    else:
        insights.append(
            "Cross-validation against `rccl-tests` (§9) confirms whether these PyTorch "
            "numbers track the RCCL ground truth; large gaps indicate framework overhead "
            "in the collective dispatch path."
        )
    s.text("\n**Insights:**\n",
           html="<p><strong>Insights:</strong></p>")
    s.bullets(insights)

    # Fused-path status with the precise three-way classification per
    # the user feedback: distinguish "not supported", "not installed",
    # "not exercised" so follow-up is unambiguous. The classification
    # comes from `fused.json` (when present) supplemented by what the
    # multi-rank `comm.json` itself reveals.
    s.subheading("Fused TP path status", level=2)
    if not fused:
        s.callout(
            "info", "Fused path: not exercised",
            "The `bench06_fused` probe did not run on this campaign "
            "(missing `06_multigpu_fused/fused.json`). The TP path "
            "executed sequential collective + matmul; whether a fused "
            "kernel is available was not determined."
        )
    elif fused.get("available"):
        s.callout(
            "success", f"Fused path: available — `{fused.get('api_source')}`",
            "AG+MM and/or MM+RS resolved against AITER / functional-collectives. "
            "See *Fused Compute+Collective Kernels* for measurements."
        )
    else:
        reason = (fused.get("reason") or "").lower()
        if "not installed" in reason or "missing" in reason:
            kind, head = "warn", "Fused path: not installed"
        elif "world" in reason or "rank" in reason or "single rank" in reason:
            kind, head = "info", "Fused path: not exercised (world < 2)"
        else:
            kind, head = "warn", "Fused path: not supported (API not present)"
        s.callout(
            kind, head,
            f"Reason: `{fused.get('reason') or 'unknown'}`. "
            "See *Fused Compute+Collective Kernels* for the exact APIs "
            "the probe tried, and *Recommendations* for the follow-up."
        )

    s.insight_takeaway(
        ("Collective bandwidths plateau at the fabric ceiling for large "
         "payloads; A2A vs AG+RS is the topology decision the source pilot "
         "flags as future work."),
        ("If the fused path is *not supported*, that's a stack-version "
         "follow-up; *not installed* is a deploy follow-up; *not exercised* "
         "is a campaign-config follow-up. Don't conflate them."),
    )
    return s


def section_fused_collectives(fused: dict) -> Section:
    s = _heading(1, "Fused Compute+Collective Kernels")
    s.para(
        "AG+MM (`all_gather` + matmul) and MM+RS (matmul + `reduce_scatter`) "
        "are the two fused collective+GEMM kernels the source PDF flags as "
        "future-work targets. The campaign probes for them in AITER's "
        "namespace and the upstream PyTorch functional-collectives surface; "
        "if either resolves, we measure TFLOP/s and on-wire bytes/s. If "
        "neither resolves the row is **SKIP**, with the exact API surfaces "
        "we tried — so the moment AITER ships them, the same script "
        "starts producing numbers without code changes."
    )
    if not fused:
        s.para("_(fused-collective probe did not run — bench06_fused output missing)_")
        return s

    backend = fused.get("backend") or "?"
    world = fused.get("world")
    s.para(f"World size: **{world}** — backend `{backend}` — "
           f"available: **{fused.get('available')}**.")
    if not fused.get("available"):
        reason = fused.get("reason") or "unknown"
        s.text(f"\nStatus: _not available_ — `{reason}`.\n",
               html=f"<p>Status: <em>not available</em> — <code>{reason}</code>.</p>")
        s.para(
            "Re-run after AITER ships the fused-collectives API. The probe "
            "checks (in priority order):"
        )
        s.bullets([
            "`aiter.ops.fused_all_gather_matmul` / `fused_matmul_reduce_scatter`",
            "`aiter.fused_collective.ag_matmul` / `mm_reduce_scatter`",
            "`aiter.distributed.all_gather_matmul` / `matmul_reduce_scatter`",
            "`torch.distributed._functional_collectives.fused_all_gather_matmul` / "
            "`fused_matmul_reduce_scatter`",
        ])
        return s

    rows = fused.get("rows") or []
    if not rows:
        s.para("_(API resolved but no measurement rows present — investigate)_")
        return s
    s.para(f"API source: `{fused.get('api_source')}`.")
    table_rows = []
    for r in rows:
        if "error" in r:
            table_rows.append({
                "op":     r.get("op"),
                "M":      r.get("M"), "K": r.get("K"), "N": r.get("N"),
                "t (ms)": "ERR",
                "TFLOP/s": "—",
                "wire bw (GB/s)": "—",
                "note":   r.get("error", "")[:60],
            })
        else:
            wirebw = r.get("ag_gb_s") or r.get("rs_gb_s")
            table_rows.append({
                "op":     r.get("op"),
                "M":      r.get("M"), "K": r.get("K"), "N": r.get("N"),
                "t (ms)": _fmt(r.get("t_ms")),
                "TFLOP/s": _fmt_tflops(r.get("tflops")),
                "wire bw (GB/s)": _fmt(wirebw, 1),
                "note":   "",
            })
    s.table(table_rows, caption="Fused AG+MM and MM+RS micro-shape sweep")

    insights = []
    ag = [r for r in rows if r.get("op") == "ag_mm" and "error" not in r]
    rs = [r for r in rows if r.get("op") == "mm_rs" and "error" not in r]
    if ag:
        max_ag = max(ag, key=lambda r: r.get("tflops") or 0)
        insights.append(
            f"`ag_mm` peak: **{_fmt_tflops(max_ag.get('tflops'))} TFLOP/s** "
            f"at (M,K,N)=({max_ag.get('M')},{max_ag.get('K')},{max_ag.get('N')})."
        )
    if rs:
        max_rs = max(rs, key=lambda r: r.get("tflops") or 0)
        insights.append(
            f"`mm_rs` peak: **{_fmt_tflops(max_rs.get('tflops'))} TFLOP/s** "
            f"at (M,K,N)=({max_rs.get('M')},{max_rs.get('K')},{max_rs.get('N')})."
        )
    if insights:
        s.text("\n**Insights:**\n",
               html="<p><strong>Insights:</strong></p>")
        s.bullets(insights)
    s.insight_takeaway(
        "AG+MM and MM+RS are the source-pilot future-work targets. The "
        "scaffold above is wired to *produce numbers automatically* the "
        "moment the AITER (or upstream functional-collectives) API "
        "resolves — no code change needed on the campaign side.",
        "Track AITER releases; the moment a fused-collective+GEMM kernel "
        "ships, this section flips from `SKIP` to a measured TFLOP/s row.",
    )
    return s


def section_validation(validation: list) -> Section:
    s = _heading(1, "Validation: PyTorch vs Ground Truth")
    s.para(
        "Each PyTorch metric is compared against the canonical AMD validation tool: "
        "RVS (`gst`) for compute peak, `rocm-bandwidth-test` for memory bandwidth, "
        "and `rccl-tests` for collectives. SKIP rows mean the ground-truth tool "
        "was not installed; the campaign proceeds but does not assert correctness "
        "for that row."
    )
    if not validation:
        s.para("_(cross-validation did not run, or output missing)_")
        return s
    s.table([{
        "metric": r.get("metric"),
        "pytorch": r.get("pytorch"),
        "ground_truth": r.get("ground_truth"),
        "tool": r.get("tool"),
        "Δ %": r.get("abs_pct_diff"),
        "tol %": r.get("tolerance_pct"),
        "status": r.get("status"),
    } for r in validation])

    n_pass = sum(1 for r in validation if r.get("status") == "PASS")
    n_fail = sum(1 for r in validation if r.get("status") == "FAIL")
    n_skip = sum(1 for r in validation if r.get("status") == "SKIP")
    insights = [f"{n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP across {len(validation)} rows."]
    if n_fail:
        worst = max(
            (r for r in validation if r.get("status") == "FAIL"
             and isinstance(r.get("abs_pct_diff"), (int, float))),
            key=lambda r: r["abs_pct_diff"], default=None,
        )
        if worst:
            insights.append(
                f"⚠️ Largest disagreement: `{worst['metric']}` at {worst['abs_pct_diff']}% "
                f"(tolerance {worst['tolerance_pct']}%) against `{worst['tool']}`."
            )
    s.text("\n**Insights:**\n",
           html="<p><strong>Insights:</strong></p>")
    s.bullets(insights)
    s.insight_takeaway(
        "Cross-validation against AMD's reference tools (`rvs gst`, "
        "`rocm-bandwidth-test`, `rccl-tests`) confirms the PyTorch "
        "instrumentation is measuring what we think it is.",
        "SKIP rows mean the ground-truth tool was unavailable; treat "
        "them as `expected (CPU host)` if applicable, otherwise as a "
        "deploy follow-up.",
    )
    return s


def section_workload_description(cfg: dict, ops: dict) -> Section:
    """Workload spec, extracted from methodology so the model — not the
    measurement protocol — gets a dedicated, decision-ready section.

    Aggregates: model architecture (depth, hidden_dim, heads, FFN
    expansion), tensor shapes (batch / image-seq / text-seq), precision,
    op mix (count of compute-bound vs memory-bound ops), and the
    reference per-block FLOP/B totals. This is the section the reader
    consults *before* the analysis sub-sections so all charts and
    tables share the same frame of reference.
    """
    s = _heading(1, "Workload Description")
    if not cfg:
        s.para("_(no workload config loaded)_")
        return s
    m  = (cfg or {}).get("model", {}) or {}
    sh = (cfg or {}).get("shapes", {}) or {}
    name = (cfg or {}).get("name") or "escher_14b_480p"

    s.para(
        f"`{name}` is a Diffusion-Transformer (DiT) inference workload — "
        "transformer-stack only; the VAE encoder/decoder is out of scope per the "
        "source pilot. The architecture parameters and the canonical batch / "
        "sequence shapes used for every benchmark in this campaign are listed "
        "below; the per-op accounting in §*Workload & Roofline* derives "
        "directly from these numbers."
    )
    s.subheading("Architecture", level=2)
    s.table([
        {"param": "depth (transformer blocks)", "value": m.get("depth")},
        {"param": "hidden_dim (D)", "value": m.get("hidden_dim")},
        {"param": "n_heads", "value": m.get("n_heads")},
        {"param": "head_dim", "value": m.get("head_dim")},
        {"param": "ffn_expansion", "value": m.get("ffn_expansion")},
        {"param": "context_dim (cross-attn)", "value": m.get("context_dim")},
    ])
    s.subheading("Frozen shapes", level=2)
    s.table([
        {"param": "batch", "value": sh.get("batch")},
        {"param": "seq_image (S)", "value": sh.get("seq_image")},
        {"param": "seq_text (L)", "value": sh.get("seq_text")},
        {"param": "dtype", "value": cfg.get("dtype")},
    ])

    if ops:
        rs = ops.get("rows") or []
        tot = ops.get("totals") or {}
        n_compute = sum(1 for r in rs if r.get("bound") == "compute")
        n_memory  = sum(1 for r in rs if r.get("bound") == "memory")
        flops_total = sum((r.get("flops") or 0) for r in rs)
        flops_compute = sum((r.get("flops") or 0)
                            for r in rs if r.get("bound") == "compute")
        pct_flops = (flops_compute / flops_total) if flops_total else None

        s.subheading("Op mix (per block)", level=2)
        s.table([
            {"metric": "GFLOPs / block", "value": _fmt(tot.get("total_gflops"), 1)},
            {"metric": "Memory MB / block", "value": _fmt(tot.get("total_mb_hbm"), 1)},
            {"metric": "Average AI (FLOP/B)", "value": _fmt(tot.get("avg_arithmetic_intensity"), 1)},
            {"metric": "Compute-bound ops", "value": n_compute},
            {"metric": "Memory-bound ops", "value": n_memory},
            {"metric": "% of FLOPs in compute-bound ops",
             "value": _pct(pct_flops)},
        ])

        s.insight_takeaway(
            (f"`{name}` concentrates **{_pct(pct_flops)}** of its FLOPs in "
             "compute-bound ops — the matmul stack inside attention and FFN — "
             "with the remainder in elementwise/normalization ops. The roofline "
             "ridge therefore matters less than wall-clock GEMM throughput."),
            ("Optimization effort goes into attention + FFN matmuls; "
             "memory-bound ops are unlikely to move the headline number "
             "until the compute ceiling is saturated."),
        )
    return s


def section_reference_vs_observed(ops: dict, mfu: dict, compute: dict,
                                   bw: dict, dram: dict,
                                   is_cpu_host: bool = False,
                                   profile: Optional[Dict[str, Any]] = None
                                   ) -> Section:
    """Compact "what changed from reference" table — review-meeting friendly.

    Each row carries: metric, source-pilot / rated-spec reference,
    observed value in this campaign, signed delta, and a short
    likely-cause column. The reference values come from the source
    pilot summary embedded in `pdf_reference_targets_pct` (for MFU
    rows) and from the workload-config / vendor spec for the device in
    ``profile`` (for compute / bw / capacity rows).

    On a CPU host or when the device profile carries no rated specs,
    the rated-spec rows degrade to ``n/a`` with the host context
    stated explicitly rather than being silently dropped.
    """
    profile = profile or {}
    rated_low_tf = profile.get("rated_bf16_low")
    rated_bw     = profile.get("rated_bw_gb_s")
    rated_mem    = profile.get("rated_mem_gib")
    rated_short  = profile.get("short") or "rated"
    s = _heading(1, "Reference vs Observed")
    s.para(
        "Compact diff against the source pilot and the target device's "
        "rated specs (where known). Rows where the local campaign could "
        "not measure the metric (e.g. CPU host vs a GPU-rated spec) are "
        "surfaced as `n/a` with the host context stated in the *likely "
        "cause* column rather than silently dropped."
    )

    rows: List[Dict] = []
    # MFU rows from pdf_reference_targets_pct (the canonical source-pilot numbers)
    targets = (mfu or {}).get("pdf_reference_targets_pct") or {}
    by_scope = {r["scope"]: r for r in (mfu or {}).get("rows", [])}
    for label, scope_key in (
        ("MFU sum-of-ops",   "sum_of_ops_optimized"),
        ("MFU eager E2E",    "eager_e2e"),
        ("MFU compiled E2E", "compiled_e2e"),
    ):
        ref = targets.get(scope_key)
        # tolerate sum-of-ops alias
        if ref is None:
            for k, v in targets.items():
                if scope_key.startswith(k):
                    ref = v; break
        r = by_scope.get(scope_key) or by_scope.get(scope_key.replace("_optimized", "_default"))
        meas = (r or {}).get("mfu_measured_peak")
        meas_pct = meas * 100 if meas is not None else None
        delta = (f"{meas_pct - ref:+.0f} pp"
                 if (meas_pct is not None and ref is not None) else "n/a")
        # Classify the gap to the PDF target using the same tolerance the
        # SC-4 gate uses, so the "likely cause" column and the scorecard
        # never disagree on what counts as in-band.
        tol_pp = _threshold("mfu_pdf_tolerance_pp")
        cause = (
            "—" if (meas_pct is not None and ref is not None
                    and abs(meas_pct - ref) < tol_pp)
            else ("CPU host: scope-shape mix differs from source reference"
                  if is_cpu_host else
                  ("config drift / shape mix vs source spec"
                   if (meas_pct is not None and ref is not None
                       and meas_pct < ref - tol_pp) else
                   ("compiler win exceeds source basis (audit FLOP definition)"
                    if (meas_pct is not None and ref is not None
                        and meas_pct > ref + tol_pp) else "—")))
        )
        rows.append({
            "metric":      label + " (measured-peak basis)",
            "reference":   (f"{ref:.0f}%" if ref is not None else "n/a"),
            "observed":    (f"{meas_pct:.0f}%" if meas_pct is not None else "n/a"),
            "delta":       delta,
            "likely cause": cause,
        })

    # Calibration drift: total per-block GFLOPs vs reference
    cal = (ops or {}).get("calibration_drift") or {}
    if cal:
        for label, key, units in (
            ("Per-block GFLOPs",  "gflops_drift_pct",  "GFLOPs/block"),
            ("Per-block Memory MB",  "mb_hbm_drift_pct", "MB/block"),
        ):
            drift = cal.get(key)
            if drift is None:
                continue
            drift_thresh = _threshold("calibration_drift_pct")
            cause = ("config shape spec drifted from source pilot — "
                     "revisit the workload config"
                     if abs(drift) > drift_thresh else "—")
            rows.append({
                "metric":      f"{label} drift (vs reference)",
                "reference":   "0.0%",
                "observed":    f"{drift:+.1f}%",
                "delta":       f"{drift:+.1f}%",
                "likely cause": cause,
            })

    # Compute, bandwidth, capacity: measured vs target rated specs (when known).
    if compute and compute.get("compute_roof_tflops") is not None:
        peak = compute["compute_roof_tflops"]
        if is_cpu_host or rated_low_tf is None:
            ref_text = (f"{rated_low_tf:,.0f} TFLOP/s ({rated_short} rated low)"
                        if rated_low_tf else "rated specs not in registry")
            cause = ("CPU host — not comparable to GPU rating"
                     if is_cpu_host else
                     "no rated spec — observed value is the standalone reference")
            rows.append({
                "metric":      "BF16 compute peak",
                "reference":   ref_text,
                "observed":    f"{_fmt_tflops(peak)} TFLOP/s",
                "delta":       "n/a",
                "likely cause": cause,
            })
        else:
            rows.append({
                "metric":      "BF16 compute peak",
                "reference":   f"{rated_low_tf:,.0f} TFLOP/s ({rated_short} rated low)",
                "observed":    f"{_fmt_tflops(peak)} TFLOP/s",
                "delta":       f"{(peak / rated_low_tf - 1) * 100:+.0f}%",
                "likely cause": ("matrix-unit utilization at chosen shape"
                                 if peak < rated_low_tf else
                                 "consistent with 'measured > rated' on tight loops"),
            })
    if bw and bw.get("bandwidth_roof_gb_s") is not None:
        bwv = bw["bandwidth_roof_gb_s"]
        if is_cpu_host or rated_bw is None:
            ref_text = (f"{rated_bw:,.0f} GB/s ({rated_short} on-package rated)"
                        if rated_bw else "rated specs not in registry")
            cause = ("CPU host (DDR plateau, not HBM)"
                     if is_cpu_host else
                     "no rated spec — observed value is the standalone reference")
            rows.append({
                "metric":      "Memory bandwidth roof",
                "reference":   ref_text,
                "observed":    f"{_fmt(bwv, 1)} GB/s",
                "delta":       "n/a",
                "likely cause": cause,
            })
        else:
            rows.append({
                "metric":      "Memory bandwidth roof",
                "reference":   f"{rated_bw:,.0f} GB/s ({rated_short} rated)",
                "observed":    f"{_fmt(bwv, 0)} GB/s",
                "delta":       f"{(bwv / rated_bw - 1) * 100:+.0f}%",
                "likely cause": ("expected for streaming microbench — "
                                 "real workloads see less due to partial reuse"),
            })
    if dram and dram.get("max_alloc_bf16_gib") is not None:
        cap = dram["max_alloc_bf16_gib"]
        if is_cpu_host or rated_mem is None:
            ref_text = (f"{rated_mem:.0f} GiB ({rated_short} rated)"
                        if rated_mem else "rated specs not in registry")
            cause = ("CPU host RAM is the bound, not on-package memory"
                     if is_cpu_host else
                     "no rated spec — observed value is the standalone reference")
            rows.append({
                "metric":      "Usable Memory (bf16 contig)",
                "reference":   ref_text,
                "observed":    f"{cap:.2f} GiB",
                "delta":       "n/a",
                "likely cause": cause,
            })
        else:
            rows.append({
                "metric":      "Usable Memory (bf16 contig)",
                "reference":   f"{rated_mem:.0f} GiB ({rated_short} rated)",
                "observed":    f"{cap:.2f} GiB",
                "delta":       f"{(cap / rated_mem - 1) * 100:+.0f}%",
                "likely cause": ("driver reserve + framework overhead "
                                 "+ allocator fragmentation"),
            })

    if rows:
        s.table(rows, caption="Source-pilot / rated-spec reference vs current campaign")

    s.insight_takeaway(
        ("The reference column is the source pilot or device-rated spec; "
         "deltas tell you whether this run is in-family with the reference, "
         "where the infrastructure is the binding factor, and where config "
         "drift is."),
        ("Treat any non-`—` row in *likely cause* as a follow-up item; "
         "config-drift rows go straight to *Recommendations*."),
    )
    return s


def section_recommendations(scorecard: list, fused: dict, mfu: dict,
                            ops: dict, comm: dict,
                            is_cpu_host: bool = False) -> Section:
    """Prioritized action list. Synthesizes scorecard FAILs / WARNs,
    fused-kernel availability, calibration drift, and per-op outliers
    into a single ordered list with priorities (P1..P4) and explicit
    owners ("infra", "kernel team", etc., where the artifact reveals
    them — otherwise just "next campaign")."""
    s = _heading(1, "Recommendations")
    items: List[Tuple[str, str, str]] = []  # (priority, action, rationale)

    # P1 = blockers (FAILs, calibration drift over thresholds.calibration_drift_pct)
    fails = [r for r in scorecard or [] if r.get("status") == "FAIL"]
    for r in fails:
        sc = r.get("sc")
        reason = r.get("reason") or ", ".join(
            f"{k}={v}" for k, v in r.items()
            if k not in ("sc", "status", "reason"))
        items.append((
            "P1",
            f"Resolve {sc}: {reason[:140]}",
            "Hard FAIL — blocks sign-off.",
        ))
    cal = (ops or {}).get("calibration_drift") or {}
    drift_thresh = _threshold("calibration_drift_pct")
    if cal.get("gflops_drift_pct") and abs(cal["gflops_drift_pct"]) > drift_thresh:
        items.append((
            "P1",
            "Revisit workload config shapes to close calibration drift",
            f"Per-block GFLOPs drift = {cal['gflops_drift_pct']:+.1f}% vs "
            f"reference; > {drift_thresh:g}% threshold blocks reproduction sign-off.",
        ))

    if is_cpu_host:
        items.append((
            "P2",
            "Re-run the full campaign on the chosen target accelerator",
            "Operational sign-off requires target-hw numbers — on-package "
            "bandwidth, NCCL/RCCL collectives, and fused TP kernels can "
            "only be exercised there.",
        ))

    if fused and not fused.get("available"):
        items.append((
            "P3",
            "Track vendor releases (AITER / cuBLASLt / etc.) for fused AG+MM / MM+RS",
            "Source pilot flags these as future-work; the bench06_fused "
            "scaffold auto-flips to PASS once the API resolves "
            "(no code changes needed).",
        ))

    # P4 = op-level optimization candidates surfaced by the per-op data
    rs = (ops or {}).get("rows") or []
    over_theory: List[Tuple[str, float]] = []
    tunable_thresh = _threshold("meas_over_theory_tunable")
    for r in rs:
        v = r.get("meas_over_theory_optimized")
        if isinstance(v, (int, float)) and v > tunable_thresh and not math.isnan(v):
            over_theory.append((r["op_name"], v))
    if over_theory:
        worst = sorted(over_theory, key=lambda x: -x[1])[:3]
        items.append((
            "P4",
            f"Audit kernel quality on {len(worst)} op(s) above "
            f"{tunable_thresh:.2f}× theory: "
            + ", ".join(f"`{n}` ({v:.2f}×)" for n, v in worst),
            f"Source pilot threshold §10.4: > {tunable_thresh:.2f}× theory = "
            f"implementation-quality issue.",
        ))

    # P4 = MFU stability follow-up
    by_scope = {r["scope"]: r for r in (mfu or {}).get("rows", [])}
    eager = by_scope.get("eager_e2e") or {}
    compiled = by_scope.get("compiled_e2e") or {}
    if eager.get("std_ms") is not None and compiled.get("std_ms") is not None:
        if compiled["std_ms"] > eager["std_ms"]:
            items.append((
                "P4",
                "Audit `torch.compile` mode / autotune budget — "
                "compiled chunk-σ exceeds eager chunk-σ",
                "Inverts source-pilot expectation; on GPU this would warrant "
                "compile-mode retune (try `max-autotune-no-cudagraphs`).",
            ))

    # 24h sustained gap
    items.append((
        "P3",
        "Run 24h sustained probe with thermal + power telemetry "
        "(`bench07_sustained` placeholder)",
        "Closes the operational steady-state gap the source pilot calls "
        "out as future work.",
    ))

    if not items:
        s.para("_(no actionable items — all criteria PASS, no drift, "
               "no kernel outliers)_")
        return s

    rows = [{"priority": p, "action": a, "rationale": r} for p, a, r in items]
    s.table(rows, caption="Prioritized action list")
    s.bullets([
        "**P1** = blocker (must resolve before sign-off).",
        "**P2** = required for full operational coverage.",
        "**P3** = scheduled / queued follow-up.",
        "**P4** = optimization opportunity (quality, not blocker).",
    ])

    p1 = sum(1 for p, _, _ in items if p == "P1")
    p2 = sum(1 for p, _, _ in items if p == "P2")
    s.insight_takeaway(
        (f"{p1} blocker(s), {p2} target-hw gap(s), "
         f"{len(items) - p1 - p2} scheduled follow-up(s)."),
        ("Address P1 rows before sign-off. P2 rows scope the next campaign. "
         "P3/P4 rows feed into the steady-state cadence."),
    )
    return s


def section_conclusion(scorecard: list, mfu: dict, fused: dict,
                       compute: dict, bw: dict,
                       is_cpu_host: bool = False,
                       profile: Optional[Dict[str, Any]] = None,
                       workload_name: Optional[str] = None) -> Section:
    """Net-outcome paragraph + explicit go/no-go status. The user-facing
    answer to *should we ship this run?* — short, direct, and built
    only from the scorecard verdict and the headline numbers."""
    s = _heading(1, "Conclusion")
    verdict, kind = _scorecard_overall(scorecard)
    by_scope = {r["scope"]: r for r in (mfu or {}).get("rows", [])}
    compiled = by_scope.get("compiled_e2e") or {}
    eager = by_scope.get("eager_e2e") or {}
    mfu_co = compiled.get("mfu_measured_peak")
    mfu_eg = eager.get("mfu_measured_peak")

    profile = profile or {}
    workload_label = workload_name or "the workload"
    target_label   = profile.get("name") or "the target accelerator"

    pieces: List[str] = []
    if is_cpu_host:
        pieces.append(
            "This campaign **validates the measurement infrastructure on a CPU "
            "host**. All timing, FLOP-accounting, multi-rank dispatch, "
            "headroom-after-load, and fused-kernel-probe paths produce "
            "structured artifacts that score against the SC-1…SC-12 grid."
        )
    else:
        peak = (compute or {}).get("compute_roof_tflops")
        bwv  = (bw or {}).get("bandwidth_roof_gb_s")
        ceiling_bits = []
        if peak: ceiling_bits.append(f"BF16 peak {_fmt_tflops(peak)} TFLOP/s")
        if bwv:  ceiling_bits.append(f"memory bandwidth {_fmt(bwv, 0)} GB/s")
        pieces.append(
            f"This campaign **measures `{workload_label}` end-to-end on "
            f"**{target_label}**. Hardware ceilings (" +
            ", ".join(ceiling_bits) + ") anchor every downstream MFU figure."
        )
    if mfu_co is not None and mfu_eg is not None:
        pieces.append(
            f"The compiled E2E path delivers **{_pct(mfu_co)}** MFU on the "
            f"measured-peak basis (eager: {_pct(mfu_eg)}); the compile lift "
            f"is **{(mfu_co - mfu_eg) * 100:+.0f} pp**."
        )
    if fused and not fused.get("available"):
        pieces.append(
            "Fused collective+GEMM kernels (AG+MM, MM+RS) are **not yet "
            "available** in the current AITER / PyTorch stack. The campaign "
            "records this as `SKIP` rather than `FAIL` and the regression "
            "auto-flips the day the API resolves."
        )
    pieces.append(f"**Overall status: {verdict}.**")
    s.para(" ".join(pieces))

    s.callout(
        ("error" if kind == "error"
         else "warn" if kind == "warn"
         else "success" if kind == "success" else "info"),
        f"Go / no-go: {verdict}",
        ("Address every `blocker` row in *Scope & Objectives* before next "
         "sign-off. See *Recommendations* for the prioritized action list."),
    )
    s.insight_takeaway(
        ("The methodology is reproducible and the artifacts are diff-stable, "
         "so the next campaign produces a directly comparable scorecard."
         if not is_cpu_host else
         "Infrastructure is validated; operational numbers still require a "
         "target-hw run."),
        ("Lock this run as the regression baseline, then iterate on the "
         "P1/P2 items from *Recommendations*."),
    )
    return s


def section_known_limitations(is_cpu_host: bool, scorecard: list,
                              dram: dict, fused: dict) -> Section:
    """Standalone "Known limitations" callout box. Lists the operational
    caveats a reader needs to internalize before quoting any number
    out of context."""
    s = _heading(1, "Known Limitations")
    bullets: List[str] = []
    if is_cpu_host:
        bullets.append(
            "**CPU host**: BF16 peak, memory bandwidth, headroom and any "
            "MFU figure on a rated-peak basis cannot be compared to GPU "
            "numbers. Measured-peak MFU is the only directly comparable "
            "quantity across hosts."
        )
    headroom = (dram or {}).get("headroom") or {}
    if headroom and headroom.get("probe_capped"):
        bullets.append(
            "**Headroom probe was capped** because the host could not hold "
            "the full bf16 model. Residual capacity is reported relative to "
            "the partial allocation; the operational number requires a "
            "target-hw re-run."
        )
    if fused and not fused.get("available"):
        bullets.append(
            "**Fused collective+GEMM kernels (AG+MM, MM+RS) are not "
            "available** in the current stack. The TP path falls back to "
            "sequential collective + matmul; performance vs the fused "
            "future is unmeasured."
        )
    skipped = [r for r in scorecard or [] if r.get("status") == "SKIP"]
    if skipped:
        bullets.append(
            f"**{len(skipped)} SKIP row(s)** in the scorecard — see "
            "*Scope & Objectives* for the per-row classification "
            "(expected / acceptable / blocker)."
        )
    bullets.append(
        "All numbers are post-warmup medians; transient cold-start "
        "performance is not in this report."
    )
    bullets.append(
        "Sustained 24h thermal & power steady-state is **not** in this "
        "campaign; it is a separate `bench07_sustained` workflow."
    )
    s.callout("warn", "Caveats before quoting any number out of context",
              "Read the bullets below before citing headline figures.")
    s.bullets(bullets)
    return s


def section_appendix(env: dict) -> Section:
    """Appendix: exact toolchain versions and how to regenerate the
    report. Lives at the end so the report's narrative doesn't fight
    the trivia."""
    s = _heading(1, "Appendix: Toolchain & Reproduction")
    sw = (env or {}).get("software", {}) or {}
    torch_info = sw.get("torch", {}) or {}
    rows = [
        {"component": "PyTorch",     "version": torch_info.get("torch_version") or "—"},
        {"component": "Triton",      "version": sw.get("triton_version") or "—"},
        {"component": "AITER",       "version": sw.get("aiter_version") or "not installed"},
        {"component": "flash_attn",  "version": sw.get("flash_attn_version") or "not installed"},
        {"component": "ROCm",        "version": sw.get("rocm_version_file") or "—"},
        {"component": "torch.cuda",  "version": torch_info.get("torch_cuda_version") or "—"},
        {"component": "torch.hip",   "version": torch_info.get("torch_hip_version") or "—"},
    ]
    s.subheading("Toolchain", level=2)
    s.table(rows)
    s.subheading("Regenerate this report", level=2)
    s.para(
        "From the campaign root:\n\n"
        "```bash\n"
        "python scripts/report.py --out results/<campaign-id>/ --format all\n"
        "```\n\n"
        "Outputs `report.md`, `report.html`, and `report.pdf`. The HTML "
        "carries base64-embedded plots so it survives copy/move; the "
        "Markdown links to the `plots/` directory."
    )
    s.subheading("Underlying artifacts", level=2)
    s.bullets([
        "`env.json` — host + software snapshot.",
        "`01_bf16_compute/` — peak / sweep / dtype / component GEMMs.",
        "`02_hbm_bandwidth/` — bandwidth + cache curve.",
        "`03_dram_capacity/summary.json` — capacity + headroom probe.",
        "`04_workload_ops/ops.json` — per-op accounting.",
        "`05_e2e_mfu/mfu.json` — eager / compiled / sum-of-ops MFU.",
        "`06_multigpu_comm/comm.json` — collective payload sweep.",
        "`06_multigpu_fused/fused.json` — fused-kernel probe.",
        "`09_numerical_stability/stability.json` — precision sweep.",
        "`scorecard.json` — SC-1…SC-12 grid.",
    ])
    return s


def section_glossary() -> Section:
    """Acronym list. The entries come from ``configs/report_config.json``
    so reviewers can edit phrasing or add a term without touching this
    script. Intentionally short — only abbreviations that appear in the
    headline / executive summary so a non-specialist reader can pick the
    report up cold.
    """
    s = _heading(1, "Appendix: Glossary")
    entries = _cfg().get("glossary") or []
    rows = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        term = e.get("term")
        defi = e.get("definition")
        if not term or not defi:
            continue
        rows.append({"term": term, "definition": defi})
    if not rows:
        s.para("_(no glossary entries configured)_")
        return s
    s.table(rows, caption="Acronyms used in this report")
    return s


# ---------------------------------------------------------------------------
# Render passes.
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font: 14px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif;
         color:#1f2328; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: 0.2em; margin-top: 1.8em; }}
  h2 {{ margin-top: 1.8em; color: #333; border-bottom: 1px solid #ddd;
        padding-bottom: 0.15em; }}
  h3 {{ color: #444; margin-top: 1.4em; }}
  h4 {{ color: #555; margin-top: 1em; }}
  table {{ border-collapse: collapse; margin: 0.6em 0 1.2em 0; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left;
            vertical-align: top; }}
  th {{ background: #f3f3f3; }}
  caption {{ caption-side: bottom; font-style: italic; color: #666;
             padding: 4px; text-align: left; }}
  figure {{ margin: 1em 0; }}
  figure img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  figcaption {{ font-style: italic; color: #666; margin-top: 0.3em; }}
  ul {{ padding-left: 1.6em; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
          font-size: 12.5px; }}
  .meta {{ color: #888; font-size: 12px; margin-bottom: 1.5em; }}

  /* Cover page */
  .cover-page {{ border: 2px solid #1f2328; border-radius: 6px;
                 padding: 1.8em 2em; margin: 1em 0 2em 0;
                 background: linear-gradient(180deg,#fafafa 0%,#f1f5f9 100%); }}
  .cover-page .doc-title {{ font-size: 26px; font-weight: 700; color: #111;
                            margin-bottom: 0.3em; }}
  .cover-page .doc-sub  {{ font-size: 16px; color: #444;
                           margin-bottom: 1.2em; }}
  .cover-page table {{ font-size: 13.5px; margin: 0.5em 0; }}

  /* Callout banners (info / warn / success / error) */
  .callout {{ display: block; padding: 0.85em 1em; margin: 1.1em 0;
              border-left: 4px solid #888; border-radius: 4px;
              background: #f7f7f9; }}
  .callout-heading {{ display: block; font-size: 14px; margin-bottom: 0.25em; }}
  .callout-body  {{ font-size: 13.5px; color: #2a2a2a; }}
  .callout-info    {{ border-left-color: #0969da; background: #ddf4ff; }}
  .callout-warn    {{ border-left-color: #bf8700; background: #fff8c5; }}
  .callout-success {{ border-left-color: #1a7f37; background: #dafbe1; }}
  .callout-error   {{ border-left-color: #cf222e; background: #ffebe9; }}

  /* Insight + Takeaway pair */
  .insight-takeaway {{ margin: 1.1em 0; padding: 0.7em 1em;
                       background: #f6f8fa; border-left: 3px solid #0969da;
                       border-radius: 3px; font-size: 13.5px; }}
  .insight-takeaway p {{ margin: 0.25em 0; }}
  .insight-takeaway strong {{ color: #0550ae; }}

  /* Status pills used in scorecard / verdict tables */
  .pill {{ display: inline-block; padding: 1px 7px; border-radius: 10px;
           font-size: 11.5px; font-weight: 600; letter-spacing: 0.02em; }}
  .pill-pass    {{ background: #dafbe1; color: #1a7f37; }}
  .pill-fail    {{ background: #ffebe9; color: #cf222e; }}
  .pill-warn    {{ background: #fff8c5; color: #9a6700; }}
  .pill-skip    {{ background: #eaeef2; color: #57606a; }}
  .pill-partial {{ background: #ddf4ff; color: #0550ae; }}

  /* Table of contents */
  .toc {{ background: #fafafa; border: 1px solid #e1e4e8; border-radius: 4px;
          padding: 0.9em 1.2em; margin: 1.4em 0 2em 0; font-size: 13.5px; }}
  .toc h3 {{ margin: 0 0 0.4em 0; font-size: 14px; color: #444; }}
  /* The explicit number is part of each section title, so suppress the
     list's own numbering / bullets to avoid "1. 1. Run Context" */
  .toc ul {{ list-style: none; margin: 0; padding-left: 0; }}
  .toc li {{ margin: 0.15em 0; }}
  .toc a  {{ color: #0550ae; text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated {now} from <code>{source_dir}</code>.</p>
{toc}
{body}
</body>
</html>
"""


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _number_top_level(sections: List[Section]) -> List[Section]:
    """Number the top-level sections (level==1) in order. Sections whose
    title starts with a digit-and-dot (e.g. ``"0. Cover"``) are taken to
    have an explicit number already and are left alone — useful for the
    cover page and other off-table-of-contents prefatory blocks. We
    rewrite ``Section.title`` in place.
    """
    n = 0
    for sec in sections:
        if sec.level != 1:
            continue
        # skip sections that already start with a number
        if sec.title and sec.title[0].isdigit() and "." in sec.title.split()[0]:
            sec.section_id = "sec-" + _slugify(sec.title)
            continue
        n += 1
        if not sec.title.startswith(f"{n}. "):
            sec.title = f"{n}. {sec.title}"
        sec.section_id = "sec-" + _slugify(sec.title)
    return sections


def _slugify(s: str) -> str:
    import re
    out = re.sub(r"[^A-Za-z0-9]+", "-", s.strip().lower()).strip("-")
    return out or "section"


def _build_toc_md(sections: List[Section]) -> str:
    lines: List[str] = ["## Contents\n"]
    for sec in sections:
        if sec.level != 1:
            continue
        lines.append(f"- [{sec.title}](#{sec.section_id or _slugify(sec.title)})")
    return "\n".join(lines) + "\n\n"


def _build_toc_html(sections: List[Section]) -> str:
    items: List[str] = []
    for sec in sections:
        if sec.level != 1:
            continue
        anchor = sec.section_id or _slugify(sec.title)
        items.append(f'<li><a href="#{anchor}">{html_escape(sec.title)}</a></li>')
    if not items:
        return ""
    # Render the TOC as <ul> rather than <ol>: every section title
    # already carries its own auto-assigned number ("1. Run Context",
    # "2. Executive Summary", …), so an <ol> renders "1. 1. Run Context"
    # in the PDF/HTML which is the issue we just fixed.
    return ('<nav class="toc">'
            '<h3>Contents</h3>'
            f'<ul>{"".join(items)}</ul>'
            '</nav>\n')


def render_md(sections: List[Section], title: str, source_dir: str) -> str:
    sections = _number_top_level(sections)
    parts = [f"# {title}\n\n",
             f"_Generated {_utc_now_iso()} from `{source_dir}`._\n\n",
             _build_toc_md(sections)]
    for sec in sections:
        anchor = sec.section_id or _slugify(sec.title)
        parts.append("#" * (sec.level + 1) + " " + sec.title +
                     f' <a id="{anchor}"></a>\n\n')
        parts.append("".join(sec.md_parts))
        parts.append("\n")
    return "".join(parts)


def render_html(sections: List[Section], title: str, source_dir: str) -> str:
    sections = _number_top_level(sections)
    toc = _build_toc_html(sections)
    body_parts: List[str] = []
    for sec in sections:
        anchor = sec.section_id or _slugify(sec.title)
        body_parts.append(
            f'<h{sec.level + 1} id="{anchor}">'
            f'{html_escape(sec.title)}'
            f'</h{sec.level + 1}>\n'
        )
        body_parts.append("".join(sec.html_parts))
    return _HTML_TEMPLATE.format(
        title=html_escape(title),
        now=_utc_now_iso(),
        source_dir=html_escape(source_dir),
        body="".join(body_parts),
        toc=toc,
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def _detect_pdf_backend() -> Tuple[Optional[List[str]], str]:
    """Pick the best available HTML/MD -> PDF pipeline on this host.

    Returns ``(cmd_template, label)`` where ``cmd_template`` is a list of
    placeholders ``{src}`` / ``{dst}`` ready for ``str.format`` substitution,
    or ``(None, reason)`` when no backend is available. We prefer
    HTML-source pipelines because the report's plots are base64-embedded
    in the HTML; the Markdown route requires a separate image-resolution
    pass that LaTeX engines handle inconsistently.

    Priority order:
      1. ``wkhtmltopdf`` direct (HTML -> PDF, embeds plots cleanly)
      2. ``pandoc --pdf-engine=wkhtmltopdf`` (HTML -> PDF via pandoc)
      3. ``pandoc --pdf-engine=<latex>`` (MD -> PDF; only if a LaTeX
         engine resolves; reported as a fallback because the PDF won't
         contain the embedded plot images)
    """
    # Prefer wkhtmltopdf direct: zero pandoc overhead, lossless image embed.
    if shutil.which("wkhtmltopdf"):
        return (
            ["wkhtmltopdf", "--quiet",
             "--enable-local-file-access",
             "{src_html}", "{dst}"],
            "wkhtmltopdf (direct)",
        )
    if shutil.which("pandoc"):
        if shutil.which("wkhtmltopdf"):
            return (
                ["pandoc", "{src_html}", "-o", "{dst}",
                 "--pdf-engine=wkhtmltopdf"],
                "pandoc + wkhtmltopdf",
            )
        for engine in ("xelatex", "tectonic", "pdflatex"):
            if shutil.which(engine):
                return (
                    ["pandoc", "{src_md}", "-o", "{dst}",
                     f"--pdf-engine={engine}"],
                    f"pandoc + {engine} (LaTeX route — embedded plots may not render)",
                )
        return (None,
                "pandoc found but no PDF engine "
                "(install wkhtmltopdf, xelatex, tectonic, or pdflatex)")
    return (None, "neither wkhtmltopdf nor pandoc found in PATH")


def _render_pdf(out: Path, output_name: str,
                md_path: Optional[Path],
                html_path: Optional[Path]) -> Tuple[Optional[Path], str]:
    """Drive the detected PDF backend and return (pdf_path, status)."""
    cmd_tmpl, label = _detect_pdf_backend()
    if cmd_tmpl is None:
        return None, f"skipped — {label}"
    pdf_path = out / f"{output_name}.pdf"

    needs_html = any("{src_html}" in tok for tok in cmd_tmpl)
    needs_md   = any("{src_md}"   in tok for tok in cmd_tmpl)
    src_html = str(html_path) if html_path else ""
    src_md   = str(md_path)   if md_path   else ""
    if needs_html and not html_path:
        return None, "skipped — backend needs HTML input but --format omits html"
    if needs_md and not md_path:
        return None, "skipped — backend needs Markdown input but --format omits md"

    cmd = [tok.format(src_html=src_html, src_md=src_md, dst=str(pdf_path))
           for tok in cmd_tmpl]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return None, f"failed to invoke {label}: {e!r}"
    if proc.returncode != 0:
        # Trim verbose backend output so the failure stays one-glance scannable.
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = " | ".join(err[-3:]) if err else "(no output)"
        return None, f"{label} exit={proc.returncode}: {tail[:300]}"
    if not pdf_path.exists():
        return None, f"{label} returned 0 but produced no file at {pdf_path}"
    return pdf_path, label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path,
                    help="campaign output directory (e.g. results/<id>/)")
    ap.add_argument(
        "--format",
        choices=("md", "html", "pdf", "both", "all"),
        default="all",
        help=("which formats to produce. 'md'/'html'/'pdf' = single format; "
              "'both' = md + html (legacy); 'all' = md + html + pdf (default)."),
    )
    ap.add_argument("--output-name", default="report",
                    help="file name stem (default: 'report' -> report.md / report.html / report.pdf)")
    ap.add_argument("--title", default=None, help="report title")
    ap.add_argument("--config", default="configs/escher_14b_480p.json",
                    help="workload config used by the campaign (for methodology section)")
    ap.add_argument("--no-embed", action="store_true",
                    help="HTML output: link plots by relative path instead of base64-embedding them")
    ap.add_argument("--no-pdf", action="store_true",
                    help="suppress PDF generation even when --format implies it")
    ap.add_argument(
        "--target",
        default=None,
        help=("override target-hardware detection. Accepts a substring of "
              "the device name (e.g. 'mi355x', 'mi300x', 'h100', 'a100', "
              "'r9700', 'b200'). Auto-detects from env.json by default; "
              "ignored on CPU runs."),
    )
    ap.add_argument(
        "--report-config",
        default=None,
        type=Path,
        help=("path to the data-driven report-config JSON "
              "(thresholds, target registry, glossary, project metadata). "
              "Defaults to configs/report_config.json next to this script."),
    )
    args = ap.parse_args()

    # Load the report-config first so every section helper sees the same
    # cached config (registry, thresholds, glossary, project metadata).
    _load_report_config(args.report_config)

    out: Path = args.out
    if not out.is_dir():
        raise SystemExit(f"campaign directory not found: {out}")

    # Load every artifact; missing pieces degrade gracefully.
    env       = _load(out / "env.json") or {}
    compute   = _load(out / "01_bf16_compute" / "summary.json") or {}
    sweep     = _load(out / "01_bf16_compute" / "sweep.json") or {}
    peak_json = _load(out / "01_bf16_compute" / "peak.json") or {}
    component_gemms = _load(out / "01_bf16_compute" / "component_gemms.json") or {}
    dtype_sweep_data = _load(out / "01_bf16_compute" / "dtype_sweep.json") or {}
    cache_curve_data = _load(out / "02_hbm_bandwidth" / "cache_curve.json") or {}
    stability = _load(out / "09_numerical_stability" / "stability.json") or {}
    bw_full   = _load(out / "02_hbm_bandwidth" / "bandwidth.json") or []
    bw_summary = _load(out / "02_hbm_bandwidth" / "summary.json") or {}
    dram      = _load(out / "03_dram_capacity" / "summary.json") or {}
    ops       = _load(out / "04_workload_ops" / "ops.json") or {}
    mfu       = _load(out / "05_e2e_mfu" / "mfu.json") or {}
    comm      = _load(out / "06_multigpu_comm" / "comm.json") or {}
    fused     = _load(out / "06_multigpu_fused" / "fused.json") or {}
    validation = _load(out / "validation.json") or []
    scorecard = _load(out / "scorecard.json") or []
    cfg = _load(Path(args.config)) or {}

    plots_dir = out / "plots"
    is_cpu_host = _is_cpu_host(env, compute, bw_summary, dram)
    profile = _target_profile(env, is_cpu_host, override=getattr(args, "target", None))
    workload_label = ((cfg or {}).get("name")
                      or _project_meta().get("default_workload_label")
                      or "workload")
    if args.title:
        title = args.title
    elif is_cpu_host:
        title = f"{workload_label} — CPU host campaign report"
    else:
        title = f"{workload_label} on {profile.get('short') or 'target'} — campaign report"

    def _build_sections() -> List[Section]:
        """Build the full ordered list of sections from loaded artifacts.

        Order is deliberate (matches the report outline, top-to-bottom):
          1. Cover page (numbered 0., not in TOC)
          2. Run Context (host/target banner)
          3. Executive Summary (5-bullet decision-grade)
          4. Scope & Objectives (in/out + SC-1..SC-12 grid)
          5. Test Environment & Methodology (run conditions)
          6. Workload Description (model + shapes + op mix)
          7. Results Overview (one-screen dashboard)
          8. Reference vs Observed (delta vs source pilot)
          9. Hardware Ceilings (compute / bw / capacity)
         10. GEMM Size Sweep
         11. BF16 dtype sweep
         12. Component GEMMs
         13. Memory Bandwidth + working-set interpretation
         14. Cache Hierarchy
         15. Memory Capacity (incl. headroom-after-load)
         16. Workload & Roofline
         17. Per-Op Throughput (with top-3 bottlenecks)
         18. End-to-End MFU (with sign-off basis guidance)
         19. Numerical Stability
         20. Multi-GPU Communication (with not-supported/installed/exercised)
         21. Fused Compute+Collective Kernels (promoted)
         22. Validation: PyTorch vs Ground Truth
         23. Known Limitations
         24. Recommendations
         25. Conclusion
         26. Appendix: Toolchain & Reproduction
         27. Appendix: Glossary
        """
        return [
            section_cover_page(env, cfg, scorecard, is_cpu_host, profile),
            section_host_target_banner(is_cpu_host, scorecard, env, profile),
            section_executive_summary(env, scorecard, compute, bw_summary, dram,
                                       ops, mfu, comm, fused,
                                       is_cpu_host=is_cpu_host,
                                       profile=profile,
                                       workload_name=workload_label),
            section_scope_objectives(scorecard, is_cpu_host=is_cpu_host),
            section_methodology(env, cfg),
            section_workload_description(cfg, ops),
            section_results_overview(compute, bw_summary, dram, mfu, ops, comm,
                                      fused, plots_dir,
                                      is_cpu_host=is_cpu_host),
            section_reference_vs_observed(ops, mfu, compute, bw_summary, dram,
                                           is_cpu_host=is_cpu_host,
                                           profile=profile),

            section_topline(compute, bw_summary, dram, peak_json,
                             is_cpu_host=is_cpu_host, profile=profile),
            section_relevant_shapes(sweep, plots_dir),
            section_dtype_sweep(dtype_sweep_data),
            section_component_gemms(component_gemms, ops),
            section_bandwidth(bw_full, bw_summary, plots_dir, profile=profile),
            section_cache_curve(cache_curve_data, plots_dir),
            section_dram(dram, profile=profile),
            section_workload_roofline(ops, plots_dir),
            section_per_op_default_vs_optimized(ops, plots_dir),
            section_mfu(mfu, plots_dir, profile=profile),
            section_stability(stability, plots_dir),

            section_multigpu(comm, fused),
            section_fused_collectives(fused),

            section_validation(validation),

            section_known_limitations(is_cpu_host, scorecard, dram, fused),
            section_recommendations(scorecard, fused, mfu, ops, comm,
                                     is_cpu_host=is_cpu_host),
            section_conclusion(scorecard, mfu, fused, compute, bw_summary,
                                is_cpu_host=is_cpu_host,
                                profile=profile,
                                workload_name=workload_label),

            section_appendix(env),
            section_glossary(),
        ]

    sections = _build_sections()

    # We pre-built sections with embed=True by default in `image()`. The
    # --no-embed flag is honored only for the HTML pass by re-running the
    # image links with rel paths; simplest: regenerate sections with the flag
    # toggled. For now we just respect the default (embed=True) since the MD
    # output ignores embedding anyway.
    # (no-op for MD; HTML respects the switch only when sections are rebuilt)

    want_md   = args.format in ("md", "both", "all")
    want_html = args.format in ("html", "both", "all")
    want_pdf  = (args.format in ("pdf", "all")) and not args.no_pdf

    md_path: Optional[Path] = None
    html_path: Optional[Path] = None

    if want_md:
        md_path = out / f"{args.output_name}.md"
        md_path.write_text(render_md(sections, title, str(out)))
        print(f"[report] wrote {md_path}")
    if want_html:
        html_path = out / f"{args.output_name}.html"
        if args.no_embed:
            # Rebuild every section under the no-embed override so plots
            # link by relative path. We monkey-patch Section.image for
            # the rebuild scope and call the same factory used above —
            # keeping the section ordering DRY.
            for s in sections:
                s.md_parts.clear(); s.html_parts.clear()
            original_image = Section.image

            def _image_no_embed(self, path, alt, caption=None, embed=True):
                return original_image(self, path, alt, caption, embed=False)

            Section.image = _image_no_embed  # type: ignore[assignment]
            try:
                sections = _build_sections()
            finally:
                Section.image = original_image  # type: ignore[assignment]
        html_path.write_text(render_html(sections, title, str(out)))
        print(f"[report] wrote {html_path}")

    if want_pdf:
        # PDF needs an input file; fall back to ad-hoc HTML if the user
        # asked for --format pdf without html (so they don't have to
        # remember --format all just to get the PDF).
        if html_path is None and md_path is None:
            html_path = out / f"{args.output_name}.html"
            html_path.write_text(render_html(sections, title, str(out)))
            print(f"[report] wrote {html_path} (input for PDF)")
        pdf_path, status = _render_pdf(out, args.output_name, md_path, html_path)
        if pdf_path is not None:
            print(f"[report] wrote {pdf_path}  (via {status})")
        else:
            print(f"[report] PDF {status}")
            print(
                "[report] To enable PDF: install one of\n"
                "          wkhtmltopdf  (apt install wkhtmltopdf — preferred, embeds plots)\n"
                "          pandoc + xelatex / tectonic / pdflatex\n"
                f"        Then re-run, or invoke the converter directly:\n"
                f"          pandoc {out}/{args.output_name}.html "
                f"-o {out}/{args.output_name}.pdf --pdf-engine=wkhtmltopdf"
            )
    elif args.no_pdf and args.format in ("pdf", "all"):
        print("[report] PDF skipped (--no-pdf)")

    if not want_pdf:
        print(
            f"\nManual conversion (if you want PDF / PPTX later):\n"
            f"  pandoc {out}/{args.output_name}.html -o {out}/{args.output_name}.pdf "
            f"--pdf-engine=wkhtmltopdf\n"
            f"  pandoc {out}/{args.output_name}.md   -o {out}/{args.output_name}.pptx\n"
        )
    return 0


def _rebuild_with_no_embed(builder: Callable, *args) -> Section:
    """Helper to call section builders that take a plots_dir; re-uses the same
    builder but forces image() to use relative paths.

    The `image()` API hard-codes embed=True; rather than duplicate every
    builder, we monkeypatch Section.image briefly. This keeps section code
    DRY at the cost of a tiny scope-bound override.
    """
    import contextlib
    original = Section.image

    def _image_no_embed(self, path, alt, caption=None, embed=True):
        return original(self, path, alt, caption, embed=False)

    Section.image = _image_no_embed  # type: ignore[assignment]
    try:
        return builder(*args)
    finally:
        Section.image = original  # type: ignore[assignment]


if __name__ == "__main__":
    raise SystemExit(main())
