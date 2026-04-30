# Microbenchmarks Report Style Guide

This document defines the official style, branding, and typographic conventions used for the automatically generated HTML and PDF benchmark reports. 

These styles are enforced primarily through the `configs/report_config.json` definitions and the HTML/CSS template embedded in `scripts/report.py`. By keeping these centralized, we ensure all microbenchmark reports share a consistent, high-quality, and recognizable "AMD Confidential" presentation.

---

## 1. Branding & Document Structure

*   **Sensitivity & Handling**: All reports generated from this pipeline carry an explicit `AMD Confidential - Distribution Under NDA` header applied via the markdown preamble.
*   **Cover Page**: Centered content block with heavy padding (10em top/bottom margins). Features:
    *   **Main Title**: No bottom border.
    *   **Author**: Hardcoded / derived from the runtime environment.
    *   **Date**: UTC ISO-8601 string.
    *   **Source Directory**: Monospaced reference to the exact run path to guarantee reproducibility.
*   **Pagination**: Explicit `<div style="page-break-after: always;"></div>` tags are inserted after the cover page, after the Table of Contents, and at the end of every top-level section.
*   **Table of Contents**: Rendered as an unordered list (`<ul>`) rather than an ordered list (`<ol>`) to prevent redundant numbering, since top-level sections already auto-prefix their own numbers (e.g., "1. Run Context").

## 2. Typography & Layout

*   **Font Family**: `14px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif` (Configurable via `plot_colors.font_family` in `report_config.json`).
*   **Text Color**: `#1f2328` (GitHub dark-slate style).
*   **Body Constraints**: `max-width: 1100px; margin: 2em auto; padding: 0 1em;`
*   **Headings**:
    *   **H1 (Top-level section)**: `#444`, 2px solid bottom border, heavily padded top margin.
    *   **H2 (Sub-section)**: `#333`, 1px solid `#ddd` bottom border.
    *   **H3 & H4**: Smaller, lighter gray (`#444` and `#555`), no borders.
*   **Inline Code**: Rendered with a `#f4f4f4` background, `3px` border radius, and slightly reduced font size (`12.5px`).

## 3. Color Palette (Plots & Charts)

The standard color cycle is defined in `configs/report_config.json` under `plot_colors`.

### Primary Palette (for multi-series charts)
1.  `#1f77b4` (Blue)
2.  `#ff7f0e` (Orange)
3.  `#2ca02c` (Green)
4.  `#d62728` (Red)
5.  `#9467bd` (Purple)
6.  `#8c564b` (Brown)
7.  `#e377c2` (Pink)
8.  `#7f7f7f` (Gray)

### Semantic Colors
Used when a specific color inherently carries meaning:
*   **Theory/Ceilings**: `#7f7f7f` (Gray)
*   **Measured Default**: `#ff7f0e` (Orange)
*   **Measured Optimized**: `#2ca02c` (Green)
*   **All-Gather Comm**: `#1f77b4` (Blue)
*   **Reduce-Scatter Comm**: `#9467bd` (Purple)
*   **Positive Delta/Win**: `#2ca02c` (Green)
*   **Negative Delta/Loss**: `#d62728` (Red)

## 4. UI Components & Elements

### 4.1. Tables
*   **Borders**: `1px solid #ccc` collapsed borders.
*   **Header Row**: `#f3f3f3` background for differentiation.
*   **Cell Alignment**: Left-aligned, top-vertical alignment with `4px 8px` padding.
*   **Captions**: Always placed at the bottom (`caption-side: bottom;`), italicized, and colored `#666`.

### 4.2. Status Pills
Used in validation tables, the scorecard, and the executive summary dashboard.
*   🟢 **PASS**: Background `#dafbe1`, Text `#1a7f37`
*   🔵 **PARTIAL**: Background `#ddf4ff`, Text `#0550ae`
*   🟡 **WARN**: Background `#fff8c5`, Text `#9a6700`
*   ⚪ **SKIP**: Background `#eaeef2`, Text `#57606a`
*   🔴 **FAIL**: Background `#ffebe9`, Text `#cf222e`

### 4.3. Callout Banners
Used for inline warnings or critical informational notes (`s.callout()`). Feature a 4px thick left-border and light background.
*   **Info**: Blue border (`#0969da`), light blue background (`#ddf4ff`).
*   **Warn**: Gold border (`#bf8700`), light yellow background (`#fff8c5`).
*   **Success**: Green border (`#1a7f37`), light green background (`#dafbe1`).
*   **Error**: Red border (`#cf222e`), light red background (`#ffebe9`).

### 4.4. Insight & Takeaway Boxes
Used at the end of sections to provide TL;DR executive interpretations (`s.insight_takeaway()`).
*   Background: `#f6f8fa`
*   Border: `3px solid #0969da` (Left only)
*   Bold Text: `#0550ae`
*   Font Size: slightly reduced (`13.5px`) for density.
