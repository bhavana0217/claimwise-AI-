"""ClaimWise AI - Dash dashboard.

Streamlit cannot run on IBM Z Linux (s390x) because it needs pyarrow, which has no s390x
build. This Dash version uses the same tested engine (src/) with pure-Python web libraries.

Run:  python app_dash.py --port 8050
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dash import ALL, Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Group, Scheme, Symbol

from src import plotly_charts as pc
from src.appeal_queue import (
    AGE_CAP_DAYS, AGE_MAX_POINTS, AMOUNT_CAP, AMOUNT_MAX_POINTS, CLOSED_STATUSES,
    GUIDANCE_DISCLAIMER, OPEN_STATUSES, STATUS_POINTS, build_appeal_queue,
)
from src.claims_processor import load_claims, load_denials, merge_claims_denials
from src.financial_analytics import (
    appeal_summary, claim_check_summary, denial_rate_by_amount_band, denial_rate_by_payer,
    denial_summary, denied_by_month, denied_dollars_by_stage, denials_by_reason,
)
from src.notifications import (
    NotificationSettings, build_email, evaluate_events, parse_recipients, send_email,
    smtp_settings_from_env,
)
from src.payer_alerts import SEVERITIES, alert_impact, load_updates, new_update, suggested_action
from src.risk_engine import (
    HIGH_AMOUNT_THRESHOLD, LEVEL_BANDS, REVIEW_SCORE_THRESHOLD, RULES, SCORE_CAP, analyze_claims,
    breakdown_frame, score_if_corrected,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
OUTBOX_DIR = APP_DIR / "output" / "outbox"
REVIEW_STATES = ["Open", "In review", "Resolved"]
MAX_ROWS = 1000
RISK_LEVELS = ["Clean", "Low", "Medium", "High"]
GRAPH_CONFIG = {"displayModeBar": False}

# ------------------------------------------------------------------------------ data
try:
    CLAIMS, CLAIMS_INFO = load_claims(DATA_DIR / "claims_and_billing.csv")
    DENIALS, _ = load_denials(DATA_DIR / "denials.csv")
except FileNotFoundError:
    sys.exit("Could not find data/claims_and_billing.csv and data/denials.csv. Copy the two datathon files into data/.")

MERGED = merge_claims_denials(CLAIMS, DENIALS)
QUEUE = build_appeal_queue(MERGED)
AS_OF = MERGED["denial_date"].max()
ANALYZED = {"main": analyze_claims(CLAIMS)}
DEMO_CLAIMS, DEMO_INFO = load_claims(DATA_DIR / "sample_claims.csv")
ANALYZED["demo"] = analyze_claims(DEMO_CLAIMS)
BASE_UPDATES = load_updates(DATA_DIR / "payer_rule_updates.csv")
CHECKER_SUMMARY_MAIN = claim_check_summary(ANALYZED["main"])
PAYERS = sorted(QUEUE["insurance_provider"].dropna().unique())
CODES = sorted(QUEUE["denial_reason_code"].dropna().unique())
_AI_KEY_PRESENT = bool(os.environ.get("OPENAI_API_KEY", "").strip())
_AI_MODEL = os.environ.get("COUVERTE_AGENT_MODEL", "gpt-4.1-mini")


# ------------------------------------------------------------------------------ helpers
def money(x: float) -> str:
    return f"${x:,.0f}"


def kpi(label: str, value: str, sublabel: str = "", hero: bool = False) -> html.Div:
    cls = "kpi hero" if hero else "kpi"
    children = [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value")]
    if sublabel:
        children.append(html.Div(sublabel, className="kpi-sub"))
    return html.Div(children, className=cls)


def kpi_row(items) -> html.Div:
    """Accept 2-tuples (label, value), 3-tuples (label, value, sublabel), or dicts."""
    tiles = []
    for item in items:
        if isinstance(item, dict):
            tiles.append(kpi(**item))
        elif len(item) >= 3:
            tiles.append(kpi(item[0], item[1], sublabel=item[2]))
        else:
            tiles.append(kpi(item[0], item[1]))
    return html.Div(tiles, className="kpi-row")


def page_header(title: str, desc: str = "", notice: html.Div | None = None) -> html.Div:
    """Consistent page title + optional caption + optional banner notice."""
    children: list = [html.H3(title, style={"margin": "0 0 4px"})]
    if desc:
        children.append(html.Div(desc, className="caption"))
    if notice:
        children.append(html.Div(notice, style={"marginTop": "10px"}))
    return html.Div(children, style={"marginBottom": "20px"})


_BADGE_CLASS: dict[str, str] = {
    # risk / priority levels
    "high": "high", "medium": "medium", "low": "low",
    "clean": "clean", "closed": "closed",
    # payer update status
    "in effect": "in-effect", "upcoming": "upcoming", "date unknown": "closed",
    # denial stage (work_status)
    "open - no appeal filed": "stage-open-no-appeal",
    "open - appeal pending": "stage-appeal-pending",
    "open - status unclear": "stage-unclear",
    "closed - recovered": "stage-recovered",
    "closed - written off": "stage-written-off",
}


def badge(level: str) -> html.Span:
    key = (level or "").lower()
    cls = _BADGE_CLASS.get(key, key.replace(" ", "-").replace("/", "-"))
    return html.Span(level, className=f"badge {cls}")


def simple_table(df: pd.DataFrame) -> html.Table:
    head = html.Thead(html.Tr([html.Th(c) for c in df.columns]))
    body = html.Tbody([html.Tr([html.Td(str(v)) for v in row]) for row in df.itertuples(index=False)])
    return html.Table([head, body], className="plain-table")


def records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    sub = df[cols].astype(object)
    return sub.where(df[cols].notna(), None).to_dict("records")


def money_col(name: str, col: str) -> dict:
    return {"name": name, "id": col, "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed, group=Group.yes, symbol=Symbol.yes, symbol_prefix="$")}


def int_col(name: str, col: str) -> dict:
    return {"name": name, "id": col, "type": "numeric", "format": Format(precision=0, scheme=Scheme.fixed, group=Group.yes)}


def text_col(name: str, col: str) -> dict:
    return {"name": name, "id": col}


STATUS_COL = {"name": "Reviewer status", "id": "review_status", "editable": True, "presentation": "dropdown"}
LEVEL_STYLE = {
    "High": ("#9A2A12", "#FDE8E4"), "Medium": ("#7A4B00", "#FEF1D6"),
    "Low": ("#1E4F80", "#E3EEF9"), "Clean": ("#1F6B45", "#E4F4EA"),
    "Closed": ("#4B5A69", "#EEF1F4"),
}
# Denial stage (work_status) colors
_STAGE_STYLE = {
    "Open - no appeal filed": ("#9A3412", "#FFF7ED"),
    "Open - appeal pending":  ("#7A4B00", "#FFFBEB"),
    "Open - status unclear":  ("#374151", "#F3F4F6"),
    "Closed - recovered":     ("#1F6B45", "#E4F4EA"),
    "Closed - written off":   ("#4B5A69", "#EEF1F4"),
}
# Payer update status colors
_PAYER_STATUS_STYLE = {
    "In effect":   ("#0A6B63", "#E0F2F0"),
    "Upcoming":    ("#1E3A8A", "#EFF6FF"),
    "Date unknown":("#4B5A69", "#EEF1F4"),
}


def level_styles(col: str) -> list[dict]:
    return [
        {"if": {"filter_query": f'{{{col}}} = "{lvl}"', "column_id": col},
         "color": fg, "backgroundColor": bg, "fontWeight": "700", "borderRadius": "4px"}
        for lvl, (fg, bg) in LEVEL_STYLE.items()
    ]


def stage_styles() -> list[dict]:
    return [
        {"if": {"filter_query": f'{{"work_status"}} = "{s}"', "column_id": "work_status"},
         "color": fg, "backgroundColor": bg, "fontWeight": "600"}
        for s, (fg, bg) in _STAGE_STYLE.items()
    ]


def payer_status_styles() -> list[dict]:
    return [
        {"if": {"filter_query": f'{{"status"}} = "{s}"', "column_id": "status"},
         "color": fg, "backgroundColor": bg, "fontWeight": "600"}
        for s, (fg, bg) in _PAYER_STATUS_STYLE.items()
    ]


def score_bar_styles() -> list[dict]:
    """Stacked background-gradient rules giving priority_score column a proportional bar."""
    bands = [
        (20, "rgba(34,197,94,0.15)"),
        (40, "rgba(59,130,246,0.15)"),
        (60, "rgba(245,158,11,0.18)"),
        (80, "rgba(239,68,68,0.2)"),
    ]
    return [
        {"if": {"filter_query": f"{{priority_score}} >= {t}", "column_id": "priority_score"},
         "background": f"linear-gradient(to right, {c} {t}%, transparent {t}%)",
         "fontVariantNumeric": "tabular-nums", "fontWeight": "700"}
        for t, c in bands
    ]


def make_table(table_id: str, columns: list[dict], styles: list[dict], page_size: int = 10, with_status: bool = True):
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=[],
        page_size=page_size,
        page_action="native",
        sort_action="native",
        row_selectable="single",
        selected_rows=[],
        editable=with_status,
        dropdown={"review_status": {"options": [{"label": s, "value": s} for s in REVIEW_STATES], "clearable": False}} if with_status else {},
        style_table={"overflowX": "auto", "border": "1px solid var(--border, #DCE6EC)",
                     "borderRadius": "10px", "maxHeight": "480px", "overflowY": "auto"},
        fixed_rows={"headers": True},
        style_header={"backgroundColor": "var(--bg-subtle, #F1F6F9)",
                      "color": "var(--text-secondary, #5B6B7B)",
                      "fontWeight": "600", "border": "none",
                      "borderBottom": "1px solid var(--border, #DCE6EC)",
                      "position": "sticky", "top": "0", "zIndex": "1"},
        style_cell={"padding": "9px 12px", "fontSize": "13px", "border": "none",
                    "borderBottom": "1px solid var(--border, #EEF2F5)",
                    "textAlign": "left", "fontFamily": "inherit",
                    "color": "var(--text-primary, #1B2A3A)"},
        style_cell_conditional=[
            {"if": {"column_type": "numeric"},
             "textAlign": "right", "fontVariantNumeric": "tabular-nums"},
        ],
        style_data_conditional=styles + [
            # Selected row
            {"if": {"state": "selected"},
             "backgroundColor": "var(--teal-100, #D9EEEA)",
             "border": "1px solid var(--teal-600, #0F766E)"},
            # Row hover (whole row)
            {"if": {"state": "active"},
             "backgroundColor": "var(--bg-hover, #EBF2F8)",
             "border": "none"},
        ],
        css=[
            {"selector": ".Select-menu-outer", "rule": "display: block !important"},
            # Make the entire row trigger selection, not just the radio circle
            {"selector": "tr", "rule": "cursor: pointer;"},
            # Sticky header support
            {"selector": ".dash-spreadsheet-container", "rule": "overflow: visible !important;"},
        ],
    )


def merge_status(store: dict | None, rows: list[dict] | None) -> dict:
    out = dict(store or {})
    for r in rows or []:
        if r.get("claim_id"):
            out[str(r["claim_id"])] = r.get("review_status") or "Open"
    return out


def apply_status(df: pd.DataFrame, store: dict | None) -> pd.DataFrame:
    out = df.copy()
    store = store or {}
    out["review_status"] = [store.get(str(c), "Open") for c in out["claim_id"]]
    return out


# ------------------------------------------------------------------------------ SVG icons (inline via dcc.Markdown dangerously_allow_html)
def _icon(svg_str: str):
    """Wrap a raw SVG string in a dcc.Markdown so it renders inline."""
    return dcc.Markdown(svg_str, dangerously_allow_html=True,
                        style={"display": "inline-flex", "alignItems": "center", "flexShrink": "0", "lineHeight": "0"})


_OVERVIEW_ICON = _icon(
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="1" y="1" width="6" height="6" rx="1" stroke="currentColor" stroke-width="1.5"/>'
    '<rect x="9" y="1" width="6" height="6" rx="1" stroke="currentColor" stroke-width="1.5"/>'
    '<rect x="1" y="9" width="6" height="6" rx="1" stroke="currentColor" stroke-width="1.5"/>'
    '<rect x="9" y="9" width="6" height="6" rx="1" stroke="currentColor" stroke-width="1.5"/>'
    '</svg>'
)

_PAYER_ICON = _icon(
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M2 14V7L8 2L14 7V14" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>'
    '<rect x="6" y="10" width="4" height="4" stroke="currentColor" stroke-width="1.5"/>'
    '</svg>'
)

_APPEAL_ICON = _icon(
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M3 4H13M3 8H10M3 12H8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>'
    '</svg>'
)

_CHECKER_ICON = _icon(
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M8 2L13 4V8C13 11 10.5 13.5 8 14C5.5 13.5 3 11 3 8V4L8 2Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>'
    '<path d="M5.5 8L7 9.5L10.5 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)

_INVESTIGATOR_ICON = _icon(
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="7" cy="7" r="4.5" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M10.5 10.5L14 14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>'
    '</svg>'
)

_DATASOURCES_ICON = _icon(
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<ellipse cx="8" cy="4" rx="5" ry="2" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M3 4V8C3 9.1 5.24 10 8 10C10.76 10 13 9.1 13 8V4" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M3 8V12C3 13.1 5.24 14 8 14C10.76 14 13 13.1 13 12V8" stroke="currentColor" stroke-width="1.5"/>'
    '</svg>'
)

_NOTIFY_ICON = _icon(
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M8 2C5.79 2 4 3.79 4 6V10L2 12H14L12 10V6C12 3.79 10.21 2 8 2Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>'
    '<path d="M6.5 12C6.5 12.83 7.17 13.5 8 13.5C8.83 13.5 9.5 12.83 9.5 12" stroke="currentColor" stroke-width="1.5"/>'
    '</svg>'
)

_METHOD_ICON = _icon(
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="2" y="1" width="9" height="13" rx="1" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M11 4H13C13.55 4 14 4.45 14 5V14C14 14.55 13.55 15 13 15H5" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M5 5H8M5 8H8M5 11H7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>'
    '</svg>'
)


def nav_item(page_id: str, label: str, icon) -> html.Button:
    return html.Button(
        [icon, html.Span(label)],
        id={"type": "nav-btn", "page": page_id},
        className="cw-nav-item",
        n_clicks=0,
        title=label,
        **{"data-tooltip": label},
    )


# ------------------------------------------------------------------------------ app shell
app = Dash(__name__, title="ClaimWise AI", suppress_callback_exceptions=True)
app.index_string = app.index_string.replace(
    "{%favicon%}", '<link rel="icon" type="image/png" href="/assets/favicon.png">'
)
server = app.server

_SUN_SVG = (
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<circle cx="8" cy="8" r="3" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M8 1V3M8 13V15M1 8H3M13 8H15M3.05 3.05L4.46 4.46M11.54 11.54L12.95 12.95'
    'M12.95 3.05L11.54 4.46M4.46 11.54L3.05 12.95" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>'
    '</svg>'
)
_MOON_SVG = (
    '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M13.5 9.5A6 6 0 016.5 2.5a6 6 0 100 11 6 6 0 007-4z"'
    ' stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>'
    '</svg>'
)
_CHEVRON_L_SVG = (
    '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M9 11L5 7L9 3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
    '</svg>'
)

# Restore collapsed state on page load (reads persisted store, applies class to root instantly)
app.clientside_callback(
    """
    function(stored) {
        var root = document.querySelector('.cw-shell');
        if (root) root.classList.toggle('sidebar-collapsed', !!stored);
        return stored ? 'Expand navigation' : 'Collapse navigation';
    }
    """,
    Output("nav-collapse-btn", "title"),
    Input("nav-state", "data"),
)

# Toggle collapsed on button click — instant, no server round-trip
app.clientside_callback(
    """
    function(n_clicks, current) {
        if (!n_clicks) return window.dash_clientside.no_update;
        var newVal = !current;
        var root = document.querySelector('.cw-shell');
        if (root) root.classList.toggle('sidebar-collapsed', newVal);
        return newVal;
    }
    """,
    Output("nav-state", "data"),
    Input("nav-collapse-btn", "n_clicks"),
    State("nav-state", "data"),
    prevent_initial_call=True,
)

# Clientside callback: apply theme from store to document element (instant, no Python round-trip)
app.clientside_callback(
    """
    function(theme) {
        document.documentElement.setAttribute('data-theme', theme || 'light');
        return theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    }
    """,
    Output("theme-toggle-btn", "title"),
    Input("theme-store", "data"),
    prevent_initial_call=False,
)


# ------------------------------------------------------------------------------ overview
def overview_layout() -> html.Div:
    ds, asum = denial_summary(MERGED), appeal_summary(QUEUE)
    by_payer, by_reason, monthly = denial_rate_by_payer(MERGED), denials_by_reason(MERGED), denied_by_month(MERGED)
    partial_note = (
        "Lighter bars are months only partly covered by the data, so they are not comparable to full months."
        if len(monthly) and monthly["is_partial"].any() else ""
    )
    return html.Div([
        page_header(
            "Denial history",
            f"{ds['insurance_claims']:,} insurance claims · "
            f"{ds['denied_claims']:,} denied ({ds['denial_rate_pct']:.1f}%) · "
            f"{asum['open_count']:,} open denials to work",
        ),
        kpi_row([
            ("Insurance claims", f"{ds['insurance_claims']:,}", "in dataset"),
            ("Total billed", money(ds["total_billed"]), "across all claims"),
            ("Denied", f"{ds['denied_claims']:,}", f"{ds['denial_rate_pct']:.1f}% of claims"),
            ("Denied amount", money(ds["denied_amount"]), "total denied value"),
            ("Open denials", f"{asum['open_count']:,}", "need human review"),
            {"label": "Open denied value", "value": money(asum["open_value"]),
             "sublabel": "estimated recoverable", "hero": True},
            ("Recovered on appeal", money(asum["recovered_value"]), "historical wins"),
            ("Written off", money(asum["written_off_value"]), "historical losses"),
        ]),
        html.Div(
            f"Of {asum['recovered_count'] + asum['written_off_count']:,} resolved appeals, "
            f"{asum['resolved_appeal_win_rate_pct']:.1f}% were approved. "
            "Open denied value is an estimate, not a guarantee.",
            className="caption", style={"marginBottom": "16px"}),
        dcc.Loading(
            html.Div([
                dcc.Graph(id="overview-stage-chart", figure={}, config=GRAPH_CONFIG),
                dcc.Graph(id="overview-reason-chart", figure={}, config=GRAPH_CONFIG),
                html.Div([dcc.Graph(id="overview-payer-chart", figure={}, config=GRAPH_CONFIG),
                          html.Div("Dashed line = overall denial rate.", className="caption")]),
                html.Div([dcc.Graph(id="overview-monthly-chart", figure={}, config=GRAPH_CONFIG) if len(monthly) else html.Div("No denial dates to chart."),
                          html.Div(partial_note, className="caption")]),
            ], className="chart-row"),
            type="circle", color="var(--teal-600)",
        ),
    ])


# ------------------------------------------------------------------------------ overview chart callbacks
@app.callback(
    Output("overview-stage-chart", "figure"),
    Output("overview-reason-chart", "figure"),
    Output("overview-payer-chart", "figure"),
    Output("overview-monthly-chart", "figure"),
    Input("theme-store", "data"),
    Input("nav-page", "data"),
)
def update_overview_charts(theme, page):
    dark = theme == "dark"
    asum = appeal_summary(QUEUE)
    ds = denial_summary(MERGED)
    by_payer = denial_rate_by_payer(MERGED)
    by_reason = denials_by_reason(MERGED)
    monthly = denied_by_month(MERGED)
    monthly_fig = (
        pc.monthly_fig(monthly, dark=dark) if len(monthly) else pc.monthly_fig(pd.DataFrame(columns=["month", "denied_amount", "denied", "is_partial"]), dark=dark)
    )
    return (
        pc.stage_fig(denied_dollars_by_stage(asum), dark=dark),
        pc.reasons_fig(by_reason, dark=dark),
        pc.payer_fig(by_payer, ds["denial_rate_pct"], dark=dark),
        monthly_fig,
    )


# ------------------------------------------------------------------------------ appeal queue
APL_COLS = ["claim_id", "insurance_provider", "denied_amount", "reason_label", "work_status",
            "priority_score", "priority_level", "review_status"]
_APL_STAGE_OPTS = [{"label": s, "value": s} for s in OPEN_STATUSES + CLOSED_STATUSES]
_APL_PAYER_OPTS = [{"label": p, "value": p} for p in PAYERS]
_APL_ACTION_OPTS = [{"label": a, "value": a} for a in sorted(QUEUE["action_category"].dropna().unique())]


def appeal_layout() -> html.Div:
    columns = [text_col("Claim ID", "claim_id"), text_col("Payer", "insurance_provider"),
               money_col("Denied amount", "denied_amount"), text_col("Denial reason", "reason_label"),
               text_col("Denial stage", "work_status"), int_col("Priority", "priority_score"),
               text_col("Level", "priority_level"), STATUS_COL]
    return html.Div([
        page_header("Appeal & rework queue",
                    "Denied claims ranked by amount, appeal stage, and days since denial. " + GUIDANCE_DISCLAIMER),
        # Three-dropdown filter bar
        html.Div([
            html.Div([
                html.Div("Denial stage", className="field-label"),
                dcc.Dropdown(id="apl-filter-stage", options=_APL_STAGE_OPTS,
                             value=OPEN_STATUSES[:], multi=True, placeholder="All stages"),
            ], style={"flex": "1 1 200px"}),
            html.Div([
                html.Div("Payer", className="field-label"),
                dcc.Dropdown(id="apl-filter-payer", options=_APL_PAYER_OPTS,
                             multi=True, placeholder="All payers"),
            ], style={"flex": "1 1 180px"}),
            html.Div([
                html.Div("Action category", className="field-label"),
                dcc.Dropdown(id="apl-filter-action", options=_APL_ACTION_OPTS,
                             multi=True, placeholder="All categories"),
            ], style={"flex": "1 1 200px"}),
            html.Div([
                html.Div("Claim ID search", className="field-label"),
                dcc.Input(id="apl-search", type="text", placeholder="Search claim ID\u2026",
                          debounce=True, style={"width": "100%", "boxSizing": "border-box"}),
            ], style={"flex": "0 1 160px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "alignItems": "flex-end", "marginBottom": "4px"}),
        html.Button("Clear filters", id="apl-clear-btn", n_clicks=0,
                    style={"background": "none", "border": "none", "color": "var(--teal-600)",
                           "fontSize": "0.82rem", "cursor": "pointer", "padding": "2px 0",
                           "fontFamily": "inherit", "textDecoration": "underline"}),
        html.Div(id="apl-kpis"),
        make_table("apl-table", columns, level_styles("priority_level") + stage_styles() + score_bar_styles()),
        html.Div(id="apl-note", className="caption"),
        html.Button("Download this view as CSV", id="apl-download-btn", className="btn secondary"),
        dcc.Download(id="apl-download"),
        html.Div(id="apl-detail"),
    ])


def filter_queue(stages, payers, actions, search) -> pd.DataFrame:
    df = QUEUE.copy()
    if stages:
        df = df[df["work_status"].isin(stages)]
    if payers:
        df = df[df["insurance_provider"].isin(payers)]
    if actions:
        df = df[df["action_category"].isin(actions)]
    if search and (s := (search or "").strip()):
        df = df[df["claim_id"].astype(str).str.contains(s, case=False, na=False, regex=False)]
    return df


# Clear filters callback
@app.callback(
    Output("apl-filter-stage", "value"),
    Output("apl-filter-payer", "value"),
    Output("apl-filter-action", "value"),
    Output("apl-search", "value"),
    Input("apl-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def apl_clear(_):
    return OPEN_STATUSES[:], [], [], ""


@app.callback(Output("apl-table", "data"), Output("apl-table", "selected_rows"), Output("apl-kpis", "children"),
              Output("apl-note", "children"),
              Input("apl-filter-stage", "value"), Input("apl-filter-payer", "value"),
              Input("apl-filter-action", "value"), Input("apl-search", "value"),
              State("status-apl", "data"))
def apl_table(stages, payers, actions, search, store):
    view = apply_status(filter_queue(stages, payers, actions, search), store)
    kpis = kpi_row([("Claims in view", f"{len(view):,}"),
                    ("Denied value in view", money(view["denied_amount"].fillna(0).sum())),
                    ("High priority", f"{(view['priority_level'] == 'High').sum():,}")])
    note = f"Showing the top {MAX_ROWS:,} of {len(view):,} rows. The download has all rows." if len(view) > MAX_ROWS else (
        "" if len(view) else "No denied claims match this filter.")
    return records(view.head(MAX_ROWS), APL_COLS), [], kpis, note


@app.callback(Output("status-apl", "data"), Input("apl-table", "data_timestamp"),
              State("apl-table", "data"), State("status-apl", "data"), prevent_initial_call=True)
def apl_save_status(_, rows, store):
    return merge_status(store, rows)


@app.callback(Output("apl-detail", "children"), Input("apl-table", "selected_rows"), State("apl-table", "data"))
def apl_detail(selected, rows):
    if not selected or not rows or selected[0] >= len(rows):
        return html.Div("Select a row in the table to inspect that claim.", className="caption")
    cid = rows[selected[0]]["claim_id"]
    row = QUEUE[QUEUE["claim_id"] == cid].iloc[0]
    if row["is_open"]:
        factors = pd.DataFrame([
            {"Factor": f"Denied amount (maxes out at {money(AMOUNT_CAP)})", "Points": f"{row['amount_points']:.0f}", "Max": AMOUNT_MAX_POINTS},
            {"Factor": f"Appeal stage: {row['work_status'].replace('Open - ', '')}", "Points": f"{row['status_points']:.0f}", "Max": max(STATUS_POINTS.values())},
            {"Factor": f"Time since denial: {row['days_since_denial']} days (maxes out at {AGE_CAP_DAYS})", "Points": f"{row['age_points']:.0f}", "Max": AGE_MAX_POINTS},
        ])
        left = [html.H4("Why this priority"), simple_table(factors),
                html.Div(f"Total priority {row['priority_score']:.0f} of 100.", className="caption")]
    else:
        left = [html.H4("Status"), html.Div(row["explanation"])]
    return html.Div([
        html.Div([html.Div([html.Div(cid, className="cw-id"),
                            html.Div(f"{row['insurance_provider']} · {row['reason_label']} ({row['denial_reason_code']}) · denied {money(row['denied_amount'])}", className="cw-meta")]),
                  badge(row["priority_level"] if row["is_open"] else "Closed")],
                 style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}),
        html.Div([html.Div(left),
                  html.Div([html.H4("How to fix it"),
                            html.Div([html.B(f"{row['action_category']}: "), row["recommended_next_step"]], className="cw-info"),
                            html.Div("Change the reviewer status in the last column of the table.", className="caption")])],
                 className="split"),
    ], className="card")


@app.callback(Output("apl-download", "data"), Input("apl-download-btn", "n_clicks"),
              State("apl-filter-stage", "value"), State("apl-filter-payer", "value"),
              State("apl-filter-action", "value"), State("apl-search", "value"),
              State("status-apl", "data"), prevent_initial_call=True)
def apl_download(_, stages, payers, actions, search, store):
    view = apply_status(filter_queue(stages, payers, actions, search), store)
    return dcc.send_data_frame(view.to_csv, "claimwise_appeal_queue.csv", index=False)


# ------------------------------------------------------------------------------ claim checker
CHK_COLS = ["row_id", "claim_id", "insurance_provider", "billed_amount", "risk_score", "risk_level",
            "finding_type", "confirmed_issues", "risk_indicators", "review_status"]


def _fmt_rules(codes_str) -> str:
    """Convert semicolon-separated rule codes to human-readable labels."""
    if not codes_str or (isinstance(codes_str, float) and codes_str != codes_str):
        return ""
    codes = [c.strip() for c in str(codes_str).split(";") if c.strip()]
    return "; ".join(RULES[c].label if c in RULES else c for c in codes)


def checker_layout() -> html.Div:
    columns = [
        text_col("Claim ID", "claim_id"),
        text_col("Payer", "insurance_provider"),
        money_col("Billed", "billed_amount"),
        int_col("Risk score", "risk_score"),
        text_col("Level", "risk_level"),
        text_col("Finding", "finding_type"),
        text_col("Confirmed issues", "confirmed_issues"),
        text_col("Risk indicators", "risk_indicators"),
        STATUS_COL,
    ]
    level_opts = [{"label": l, "value": l} for l in RISK_LEVELS[1:]]
    payer_opts = [{"label": p, "value": p} for p in sorted(ANALYZED["main"]["insurance_provider"].dropna().unique())]
    rule_opts = [{"label": r.label, "value": c} for c, r in RULES.items()]

    return html.Div([
        page_header(
            "Claim checker",
            "Finds confirmed data issues (facts in the record) and risk indicators (patterns linked to denials, "
            "not predictions). No machine learning is used.",
        ),
        html.Div("Dataset", className="field-label"),
        dcc.RadioItems(
            id="chk-source",
            options=[{"label": "Loaded dataset", "value": "main"},
                     {"label": "Synthetic demo set (defects added on purpose)", "value": "demo"}],
            value="main",
            inline=True,
            inputStyle={"marginLeft": "14px", "marginRight": "5px"},
        ),
        html.Div(id="chk-demo-note"),
        html.Div(id="chk-kpis"),
        dcc.Graph(id="chk-level-chart", config=GRAPH_CONFIG),
        # Filter bar (like appeal queue)
        html.Div([
            dcc.Checklist(
                id="chk-needs-review",
                options=[{"label": " Needs review only", "value": "yes"}],
                value=["yes"],
                inputStyle={"marginRight": "5px", "accentColor": "var(--teal-600)"},
                labelStyle={"fontSize": "13px", "color": "var(--text-primary)", "cursor": "pointer"},
            ),
            dcc.Dropdown(id="chk-filter-level", options=level_opts, multi=True,
                         placeholder="Risk level\u2026", clearable=True,
                         style={"flex": "1", "minWidth": "140px"}),
            dcc.Dropdown(id="chk-filter-payer", options=payer_opts, multi=True,
                         placeholder="Payer\u2026", clearable=True,
                         style={"flex": "2", "minWidth": "160px"}),
            dcc.Dropdown(id="chk-filter-rule", options=rule_opts, multi=True,
                         placeholder="Rule\u2026", clearable=True,
                         style={"flex": "2", "minWidth": "180px"}),
            dcc.Input(id="chk-search", type="text", placeholder="Search claim ID\u2026",
                      debounce=True, className="apl-search-input"),
            html.Button("Clear", id="chk-clear-filters", className="btn-ghost",
                        style={"padding": "0 12px", "height": "38px", "fontSize": "13px"}),
        ], className="apl-filter-bar"),
        html.Div(id="chk-banner"),
        make_table("chk-table", columns,
                   level_styles("risk_level") + [
                       {"if": {"filter_query": '{claim_id} = "Missing"', "column_id": "claim_id"},
                        "color": "#991B1B", "fontWeight": "700"},
                       {"if": {"filter_query": '{insurance_provider} = "Missing"', "column_id": "insurance_provider"},
                        "color": "#991B1B", "fontWeight": "700"},
                   ]),
        html.Div(id="chk-note", className="caption"),
        html.Button("Download this view as CSV", id="chk-download-view-btn", className="btn secondary"),
        html.Button("Download all analyzed claims as CSV", id="chk-download-all-btn", className="btn secondary"),
        dcc.Download(id="chk-download"),
        html.Div(id="chk-detail"),
    ])


def filter_checker(source: str, needs_review: list | None, levels: list | None,
                   payers: list | None, rules: list | None, search: str | None) -> pd.DataFrame:
    df = ANALYZED[source].copy()
    df["row_id"] = df.index
    if needs_review:
        df = df[df["needs_review"]]
    if levels:
        df = df[df["risk_level"].isin(levels)]
    if payers:
        df = df[df["insurance_provider"].isin(payers)]
    if rules:
        both = df["confirmed_issues"].fillna("") + ";" + df["risk_indicators"].fillna("")
        df = df[both.str.contains("|".join(rules), regex=True)]
    if search and (s := (search or "").strip()):
        df = df[df["claim_id"].astype(str).str.contains(s, case=False, na=False, regex=False)]
    return df.sort_values(["risk_score", "billed_amount"], ascending=False)


def _apply_chk_display(view: pd.DataFrame) -> pd.DataFrame:
    """Add readable rule labels and Missing markers for display."""
    v = view.copy()
    v["confirmed_issues"] = v["confirmed_issues"].apply(_fmt_rules)
    v["risk_indicators"] = v["risk_indicators"].apply(_fmt_rules)
    v["claim_id"] = v["claim_id"].fillna("Missing")
    v["insurance_provider"] = v["insurance_provider"].fillna("Missing")
    return v


@app.callback(
    Output("chk-filter-level", "value"),
    Output("chk-filter-payer", "value"),
    Output("chk-filter-rule", "value"),
    Output("chk-search", "value"),
    Output("chk-needs-review", "value"),
    Input("chk-clear-filters", "n_clicks"),
    prevent_initial_call=True,
)
def chk_clear_filters(_):
    return None, None, None, "", ["yes"]


@app.callback(
    Output("chk-filter-payer", "options"),
    Input("chk-source", "value"),
)
def chk_payer_opts(source):
    df = ANALYZED[source]
    return [{"label": p, "value": p} for p in sorted(df["insurance_provider"].dropna().unique())]


@app.callback(
    Output("chk-table", "data"),
    Output("chk-table", "selected_rows"),
    Output("chk-kpis", "children"),
    Output("chk-level-chart", "figure"),
    Output("chk-demo-note", "children"),
    Output("chk-banner", "children"),
    Output("chk-note", "children"),
    Input("chk-source", "value"),
    Input("chk-needs-review", "value"),
    Input("chk-filter-level", "value"),
    Input("chk-filter-payer", "value"),
    Input("chk-filter-rule", "value"),
    Input("chk-search", "value"),
    State("status-chk", "data"),
    State("theme-store", "data"),
)
def chk_table(source, needs_review, levels, payers, rules, search, store, theme):
    analyzed = ANALYZED[source]
    summ = claim_check_summary(analyzed)
    kpis = html.Div([
        kpi_row([("Claims checked", f"{summ['total_claims']:,}"), ("Total billed", money(summ["total_billed"])),
                 (f"Flagged for review ({summ['flagged_pct']:.1f}%)", f"{summ['flagged_claims']:,}"),
                 ("Value to review (est.)", money(summ["billed_value_requiring_review"]))]),
        html.Div(f"{summ['confirmed_issue_claims']:,} claims have a confirmed data issue and "
                 f"{summ['indicator_only_claims']:,} have a risk indicator only. The dollar figure is the billed amount on "
                 "flagged claims \u2014 an estimate of value needing attention, not a predicted loss.",
                 className="caption")])
    fig = pc.level_fig(analyzed["risk_level"].value_counts().to_dict(), dark=(theme == "dark"))
    demo = (html.Div(f"These {len(DEMO_CLAIMS)} insurance claims are synthetic and contain deliberate problems so you can see "
                     f"each rule fire. {DEMO_INFO['self_pay_excluded']} self-pay row was excluded.", className="cw-demo")
            if source == "demo" else None)
    banner = (html.Div("No claims in this set were flagged for review by the rules.", className="cw-ok")
              if summ["flagged_claims"] == 0 else None)
    view = apply_status(filter_checker(source, needs_review, levels, payers, rules, search), store)
    display_view = _apply_chk_display(view)
    note = (f"Showing the top {MAX_ROWS:,} of {len(view):,} rows. The download has all rows."
            if len(view) > MAX_ROWS else ("" if len(view) else "No claims match this filter."))
    return records(display_view.head(MAX_ROWS), CHK_COLS), [], kpis, fig, demo, banner, note


@app.callback(Output("status-chk", "data"), Input("chk-table", "data_timestamp"),
              State("chk-table", "data"), State("status-chk", "data"), prevent_initial_call=True)
def chk_save_status(_, rows, store):
    return merge_status(store, rows)


@app.callback(Output("chk-detail", "children"), Input("chk-table", "selected_rows"),
              State("chk-table", "data"), State("chk-source", "value"))
def chk_detail(selected, rows, source):
    if not selected or not rows or selected[0] >= len(rows):
        return html.Div("Select a row in the table to inspect that claim.", className="caption")
    row = ANALYZED[source].loc[int(rows[selected[0]]["row_id"])]
    cid = row["claim_id"] if pd.notna(row["claim_id"]) else "(no claim ID)"
    payer = row["insurance_provider"] if pd.notna(row["insurance_provider"]) else "no payer recorded"
    amount = money(row["billed_amount"]) if pd.notna(row["billed_amount"]) else "no amount"
    after = score_if_corrected(row["confirmed_issues"], row["risk_indicators"])
    breakdown = breakdown_frame(row["confirmed_issues"], row["risk_indicators"])
    if breakdown.empty:
        body = html.Div("No rule-based issues found on this claim.", className="cw-ok")
    else:
        raw = int(breakdown["Points"].sum())
        body = html.Div([html.H4("Why this score, and how to fix it"), simple_table(breakdown),
                         html.Div(f"Points add up to {raw}" + (f", capped at {SCORE_CAP}." if raw > SCORE_CAP else ".") +
                                  " Risk indicators are patterns linked to denials, not predictions. They stay in the score "
                                  "until a person checks the claim and decides they do not apply.", className="caption")])
    return html.Div([
        html.Div([html.Div([html.Div(cid, className="cw-id"),
                            html.Div(f"{payer} · billed {amount} · status {row['claim_status'] if pd.notna(row['claim_status']) else 'unknown'}", className="cw-meta")]),
                  badge(row["risk_level"])],
                 style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}),
        kpi_row([("Risk score now", f"{row['risk_score']} / {SCORE_CAP}"),
                 ("Score if confirmed issues are corrected", f"{after} / {SCORE_CAP}")]),
        body,
        html.Div([html.B("Recommended next steps: "), row["recommended_next_steps"]], className="cw-info"),
        html.Div("Change the reviewer status in the last column of the table.", className="caption"),
    ], className="card")


@app.callback(
    Output("chk-download", "data"),
    Input("chk-download-view-btn", "n_clicks"),
    Input("chk-download-all-btn", "n_clicks"),
    State("chk-source", "value"),
    State("chk-needs-review", "value"),
    State("chk-filter-level", "value"),
    State("chk-filter-payer", "value"),
    State("chk-filter-rule", "value"),
    State("chk-search", "value"),
    State("status-chk", "data"),
    prevent_initial_call=True,
)
def chk_download(_v, _a, source, needs_review, levels, payers, rules, search, store):
    if ctx.triggered_id == "chk-download-all-btn":
        df = filter_checker(source, None, None, None, None, None)
        name = "claimwise_all_analyzed_claims.csv"
    else:
        df = filter_checker(source, needs_review, levels, payers, rules, search)
        name = "claimwise_checked_claims.csv"
    df = apply_status(df, store).drop(columns=["claim_status_norm", "row_id"])
    return dcc.send_data_frame(df.to_csv, name, index=False)


# ------------------------------------------------------------------------------ payer alerts
ALERT_COLS = ["update_id", "severity", "payer", "topic", "effective", "status", "open_denials_affected",
              "open_value_affected", "historical_denials_same_codes"]


def current_updates(sim: list[dict] | None) -> pd.DataFrame:
    frames = [BASE_UPDATES]
    for s in sim or []:
        frames.append(new_update(s["payer"], s["topic"], s["summary"], pd.Timestamp(s["effective_date"]),
                                 s["severity"], s["codes"], s["update_id"]))
    return pd.concat(frames, ignore_index=True)


def current_impact(sim: list[dict] | None) -> pd.DataFrame:
    return alert_impact(current_updates(sim), QUEUE, AS_OF)


def payer_layout() -> html.Div:
    columns = [text_col("Severity", "severity"), text_col("Payer", "payer"), text_col("Topic", "topic"),
               text_col("Effective", "effective"), text_col("Status", "status"),
               int_col("Open denials affected", "open_denials_affected"), money_col("Open value affected", "open_value_affected"),
               int_col("Past denials, same payer and codes", "historical_denials_same_codes")]
    return html.Div([
        page_header(
            "Payer rule-change alerts",
            "Track payer policy changes and see which open denials they affect.",
            html.Div([html.B("Demo feed. "), "These updates are synthetic examples, not real payer notices. "
                      "Impact numbers are real calculations on the loaded denial data."],
                     className="cw-demo"),
        ),
        html.Div(id="alert-kpis"),
        make_table("alert-table", columns, level_styles("severity") + payer_status_styles(), page_size=8, with_status=False),
        html.Div(id="alert-detail"),
        html.Details([
            html.Summary("Simulate a new payer update arriving", style={"fontWeight": "600", "cursor": "pointer", "padding": "6px 0"}),
            html.Div([
                html.Div([html.Div("Payer", className="field-label"), dcc.Dropdown(id="sim-payer", options=PAYERS, value=PAYERS[0], clearable=False)]),
                html.Div([html.Div("Severity", className="field-label"), dcc.Dropdown(id="sim-severity", options=SEVERITIES, value="Medium", clearable=False)]),
            ], className="split"),
            html.Div("Topic", className="field-label"),
            dcc.Input(id="sim-topic", type="text", placeholder="For example: Prior authorization", value=""),
            html.Div("What changed (your own words)", className="field-label"),
            dcc.Textarea(id="sim-summary", style={"width": "100%", "height": "70px"}, value=""),
            html.Div([
                html.Div([html.Div("Effective date", className="field-label"),
                          dcc.DatePickerSingle(id="sim-date", date=(AS_OF + pd.Timedelta(days=30)).strftime("%Y-%m-%d"), display_format="MMM D, YYYY")]),
                html.Div([html.Div("Related denial codes", className="field-label"), dcc.Dropdown(id="sim-codes", options=CODES, multi=True)]),
            ], className="split"),
            html.Button("Add simulated update", id="sim-add", className="btn"),
            html.Button("Reset to the demo feed", id="sim-reset", className="btn secondary"),
            html.Div(id="sim-msg", className="caption"),
        ], className="card"),
    ])


@app.callback(Output("alert-table", "data"), Output("alert-kpis", "children"), Input("sim-updates", "data"))
def alert_table(sim):
    impact = current_impact(sim)
    impact["effective"] = impact["effective_date"].dt.strftime("%b %d, %Y")
    kpis = kpi_row([("Updates in feed", f"{len(impact):,}"),
                    ("Updates touching open denials", f"{(impact['open_denials_affected'] > 0).sum():,}"),
                    ("Open denied value touched", money(impact["open_value_affected"].sum()))])
    return records(impact, ALERT_COLS), kpis


@app.callback(Output("alert-detail", "children"), Input("alert-table", "selected_rows"),
              State("alert-table", "data"), State("sim-updates", "data"))
def alert_detail(selected, rows, sim):
    if not selected or not rows or selected[0] >= len(rows):
        return html.Div("Select a row to see the details and a suggested action.", className="caption")
    uid = rows[selected[0]]["update_id"]
    row = current_impact(sim).set_index("update_id").loc[uid].copy()
    row["update_id"] = uid
    return html.Div([
        html.Div([badge(row["severity"]), html.B(f"  {row['payer']}: {row['topic']}")]),
        html.Div(f"{row['status']} · effective {row['effective_date']:%b %d, %Y} · related codes {row['related_codes']} · source: {row['source']}", className="cw-meta"),
        html.P(row["summary"]),
        html.Div([html.B("Suggested action: "), suggested_action(row)], className="cw-info"),
    ], className="card")


@app.callback(Output("sim-updates", "data"), Output("sim-msg", "children"), Output("sim-topic", "value"),
              Output("sim-summary", "value"), Output("sim-codes", "value"),
              Input("sim-add", "n_clicks"), Input("sim-reset", "n_clicks"),
              State("sim-payer", "value"), State("sim-severity", "value"), State("sim-topic", "value"),
              State("sim-summary", "value"), State("sim-date", "date"), State("sim-codes", "value"),
              State("sim-updates", "data"), prevent_initial_call=True)
def sim_update(add, reset, payer, severity, topic, summary, date, codes, sim):
    if ctx.triggered_id == "sim-reset":
        return [], "Reset to the demo feed.", "", "", []
    if not (topic or "").strip() or not codes or not date:
        return sim or [], "Please enter a topic, an effective date, and at least one related denial code.", topic, summary, codes
    sim = list(sim or [])
    sim.append({"update_id": f"SIM-{len(sim) + 1:02d}", "payer": payer, "topic": topic.strip(),
                "summary": (summary or "").strip() or "(no description)", "effective_date": date[:10],
                "severity": severity, "codes": list(codes)})
    return sim, "Added. This is a simulated update, not a real payer notice.", "", "", []


# ------------------------------------------------------------------------------ notifications
def notifications_layout() -> html.Div:
    smtp = smtp_settings_from_env()

    # SMTP status chip (static, computed at page load)
    if smtp:
        smtp_chip = html.Span(
            [html.Span(className="n-dot n-dot-on"), f"SMTP \u00b7 {smtp['host']}"],
            className="n-smtp-chip connected",
        )
    else:
        smtp_chip = html.Span(
            [html.Span(className="n-dot n-dot-off"), "Preview mode"],
            className="n-smtp-chip preview",
            title="No email server configured. Sends save a preview file in output/outbox/.",
        )

    # Summary strip with dynamic text
    summary_strip = html.Div([
        html.Div(id="n-summary-text", className="n-summary-text",
                 children="Loading\u2026"),
        html.Div([
            smtp_chip,
            html.Button("Send", id="n-send", className="btn", disabled=True,
                        style={"height": "32px", "padding": "0 14px", "fontSize": "13px"}),
        ], className="n-summary-controls"),
    ], className="n-summary-strip")

    # Left col: alert feed
    left_col = html.Div([
        html.Div([
            html.Div("Active alerts", className="n-col-title"),
            html.Div("Alerts matching your current trigger settings.", className="caption"),
        ], style={"marginBottom": "12px"}),
        html.Div(id="n-alerts"),
    ], className="card n-left-col")

    # Right col: settings + recipients
    toggle_items = [
        ("hp", "High-priority open denials", "Fires when open high-priority claims exist"),
        ("ov", "Open denied value above limit", "Fires when total open denied $$ exceeds threshold"),
        ("pu", "Payer rule update affects denials", "Fires when a payer update matches open denials"),
        ("ck", "Flagged claims from checker", "Fires when claim checker finds issues"),
    ]
    toggle_rows = []
    for val, label, hint in toggle_items:
        toggle_rows.append(
            html.Label([
                html.Div([
                    html.Div(label, className="n-toggle-label"),
                    html.Div(hint, className="n-toggle-hint"),
                ], className="n-toggle-text"),
                dcc.Checklist(
                    id=f"n-toggle-{val}",
                    options=[{"label": "", "value": val}],
                    value=[val],
                    inputStyle={"display": "none"},
                    labelClassName="n-toggle-pill-wrap",
                    labelStyle={"cursor": "pointer"},
                ),
            ], className="n-toggle-row")
        )

    thresholds = html.Div([
        html.Div([
            html.Div("Min high-priority denials", className="n-thresh-label"),
            dcc.Input(id="n-hp-min", type="number", min=1, value=1, className="n-thresh-input"),
        ], className="n-thresh-row"),
        html.Div([
            html.Div("Open denied value limit ($)", className="n-thresh-label"),
            dcc.Input(id="n-ov-limit", type="number", min=0, value=1_000_000, className="n-thresh-input"),
        ], className="n-thresh-row"),
        html.Div([
            html.Div("Min flagged claims", className="n-thresh-label"),
            dcc.Input(id="n-ck-min", type="number", min=1, value=1, className="n-thresh-input"),
        ], className="n-thresh-row"),
    ], className="n-thresholds")

    # Hidden master checklist (drives the callbacks)
    master_events = dcc.Checklist(
        id="n-events",
        value=["hp", "ov", "pu", "ck"],
        options=[
            {"label": "", "value": "hp"},
            {"label": "", "value": "ov"},
            {"label": "", "value": "pu"},
            {"label": "", "value": "ck"},
        ],
        style={"display": "none"},
    )

    right_col = html.Div([
        html.Div("Alert triggers", className="n-settings-title"),
        html.Div(toggle_rows, className="n-toggle-list"),
        master_events,
        html.Hr(className="n-divider"),
        html.Div("Thresholds", className="n-settings-title"),
        thresholds,
        html.Hr(className="n-divider"),
        html.Div("Recipients", className="n-settings-title"),
        html.Div("Separate addresses with commas. Emails contain totals only \u2014 no PHI.",
                 className="caption", style={"marginBottom": "8px"}),
        dcc.Textarea(id="n-recipients", placeholder="admin@hospital.org, billing@hospital.org",
                     className="n-recipients-input"),
        html.Div(id="n-warning", style={"marginTop": "6px"}),
        html.Hr(className="n-divider"),
        html.Details([
            html.Summary("Preview email", className="n-preview-toggle"),
            html.Pre(id="n-preview", className="n-preview-pre"),
        ]),
        html.Div(id="n-result", style={"marginTop": "8px"}),
        html.Div("A real deployment needs privacy review before sending patient-related data.",
                 className="caption", style={"marginTop": "10px", "fontStyle": "italic"}),
    ], className="card n-right-col")

    return html.Div([
        html.Div([
            html.H3("Notifications", style={"margin": "0 0 4px"}),
            html.Div("Configure email alerts that fire when claims data crosses your thresholds.",
                     className="caption"),
        ], style={"marginBottom": "16px"}),
        summary_strip,
        html.Div([left_col, right_col], className="n-layout"),
    ])


def build_alerts(events, hp_min, ov_limit, ck_min, sim):
    events = events or []
    settings = NotificationSettings(
        high_priority_enabled="hp" in events, high_priority_min_count=int(hp_min or 1),
        open_value_enabled="ov" in events, open_value_limit=float(ov_limit or 0),
        payer_update_enabled="pu" in events, checker_enabled="ck" in events, checker_min_flagged=int(ck_min or 1))
    return evaluate_events(QUEUE, CHECKER_SUMMARY_MAIN, current_impact(sim), settings)


def _grouped_alert_feed(alerts: list) -> html.Div:
    """Render alerts grouped by severity, each group as a collapsible section."""
    from collections import defaultdict
    by_sev: dict = defaultdict(list)
    for a in alerts:
        by_sev[a.severity].append(a)

    sections = []
    for sev in ["High", "Medium", "Low"]:
        group = by_sev.get(sev, [])
        if not group:
            continue
        rows = [
            html.Div([
                badge(sev),
                html.Div([
                    html.Div(a.title, className="n-alert-title"),
                    html.Div(a.detail, className="n-alert-detail"),
                ]),
            ], className="n-alert-row")
            for a in group
        ]
        sections.append(html.Details([
            html.Summary([
                html.Span(f"{len(group)}", className=f"n-sev-count n-sev-{sev.lower()}"),
                html.Span(f" {sev} {('alert' if len(group) == 1 else 'alerts')}",
                          className="n-group-label"),
            ], className="n-group-summary"),
            html.Div(rows, className="n-group-rows"),
        ], open=(sev == "High"), className="n-alert-group"))

    if not sections:
        return html.Div("Nothing currently meets your trigger settings.", className="cw-ok")
    return html.Div(sections)


def _alerts_summary_text(alerts: list) -> str:
    counts: dict = {"High": 0, "Medium": 0, "Low": 0}
    for a in alerts:
        sev = a.severity
        if sev in counts:
            counts[sev] += 1
    total = sum(counts.values())
    if total == 0:
        return "No alerts ready"
    parts = [f"{total} alert{'s' if total != 1 else ''} ready"]
    for sev in ("High", "Medium", "Low"):
        if counts[sev]:
            parts.append(f"{counts[sev]} {sev}")
    return " \u00b7 ".join(parts)


NOTIFY_INPUTS = [Input("n-recipients", "value"), Input("n-events", "value"), Input("n-hp-min", "value"),
                 Input("n-ov-limit", "value"), Input("n-ck-min", "value"), Input("sim-updates", "data")]


@app.callback(
    Output("n-alerts", "children"),
    Output("n-warning", "children"),
    Output("n-preview", "children"),
    Output("n-send", "disabled"),
    Output("n-summary-text", "children"),
    *NOTIFY_INPUTS,
)
def notify_preview(recipients, events, hp_min, ov_limit, ck_min, sim):
    alerts = build_alerts(events, hp_min, ov_limit, ck_min, sim)
    valid, invalid = parse_recipients(recipients or "")
    shown = _grouped_alert_feed(alerts)
    warning = (html.Div("Not valid email addresses: " + ", ".join(invalid), className="cw-warn")
               if invalid else None)
    summary_text = _alerts_summary_text(alerts)
    if alerts and valid:
        msg = build_email(alerts, valid, sender=(smtp_settings_from_env() or {}).get("from", "claimwise@localhost"))
        preview = f"To: {msg['To']}\nSubject: {msg['Subject']}\n\n{msg.get_content()}"
        return shown, warning, preview, False, summary_text
    return shown, warning, "Add at least one valid address and at least one alert to see a preview.", True, summary_text


@app.callback(
    Output("n-events", "value"),
    Input("n-toggle-hp", "value"),
    Input("n-toggle-ov", "value"),
    Input("n-toggle-pu", "value"),
    Input("n-toggle-ck", "value"),
)
def sync_events(hp, ov, pu, ck):
    """Keep n-events in sync with individual toggle checkboxes."""
    result = []
    for vals, code in [(hp, "hp"), (ov, "ov"), (pu, "pu"), (ck, "ck")]:
        if vals:
            result.extend(vals)
    return result


@app.callback(Output("n-result", "children"), Input("n-send", "n_clicks"), State("n-recipients", "value"),
              State("n-events", "value"), State("n-hp-min", "value"), State("n-ov-limit", "value"),
              State("n-ck-min", "value"), State("sim-updates", "data"), prevent_initial_call=True)
def notify_send(_, recipients, events, hp_min, ov_limit, ck_min, sim):
    alerts = build_alerts(events, hp_min, ov_limit, ck_min, sim)
    valid, _invalid = parse_recipients(recipients or "")
    if not alerts or not valid:
        return html.Div("Nothing to send.", className="cw-warn")
    smtp = smtp_settings_from_env()
    msg = build_email(alerts, valid, sender=(smtp or {}).get("from", "claimwise@localhost"))
    result = send_email(msg, OUTBOX_DIR, smtp)
    saved = ""
    if result.saved_to:
        try:
            saved = f" Saved to {Path(result.saved_to).relative_to(APP_DIR)}"
        except ValueError:
            saved = f" Saved to {result.saved_to}"
    return html.Div(result.message + saved, className="cw-ok" if result.sent else "cw-warn")


# ------------------------------------------------------------------------------ data sources callbacks

def _decode_upload(contents: str) -> "io.StringIO":
    """Decode a dcc.Upload base-64 payload and return a StringIO."""
    import base64, io as _io
    _header, b64 = contents.split(",", 1)
    decoded = base64.b64decode(b64).decode("utf-8", errors="replace")
    return _io.StringIO(decoded)


def _result_ui(filename: str, ok: bool, stats: dict, errors: list[str]) -> html.Div:
    """Render a compact validation result card."""
    if not ok:
        return html.Div([
            html.Div([
                html.Span("Schema error", style={"fontWeight": "700", "color": "#991B1B", "fontSize": "13px"}),
                html.Span(f" — {filename}", style={"color": "var(--text-muted)", "fontSize": "13px"}),
            ]),
            html.Ul([html.Li(e, style={"fontSize": "13px", "color": "#991B1B"}) for e in errors],
                    style={"margin": "6px 0 0 0", "paddingLeft": "18px"}),
        ], style={"background": "var(--high-bg, #FEF2F2)", "border": "1px solid #FECACA",
                  "borderRadius": "6px", "padding": "10px 12px"})

    stat_items = []
    for k, v in stats.items():
        label = k.replace("_", " ").capitalize()
        stat_items.append(html.Div([
            html.Span(label + ": ", style={"color": "var(--text-muted)", "fontSize": "13px"}),
            html.Span(f"{v:,}" if isinstance(v, int) else str(v),
                      style={"fontWeight": "600", "fontSize": "13px", "color": "var(--text-primary)"}),
        ], style={"marginBottom": "3px"}))
    return html.Div([
        html.Div([
            html.Span("Valid", style={"fontWeight": "700", "color": "#166534", "fontSize": "13px"}),
            html.Span(f" — {filename}", style={"color": "var(--text-muted)", "fontSize": "13px"}),
        ], style={"marginBottom": "8px"}),
        *stat_items,
    ], style={"background": "var(--clean-bg, #F0FDF4)", "border": "1px solid #BBF7D0",
              "borderRadius": "6px", "padding": "10px 12px"})


@app.callback(
    Output("ds-claims-result", "data"),
    Input("ds-claims-upload", "contents"),
    State("ds-claims-upload", "filename"),
    Input("ds-restore-btn", "n_clicks"),
    prevent_initial_call=True,
)
def validate_claims_upload(contents, filename, _restore):
    from src.claims_processor import SchemaError, load_claims
    if ctx.triggered_id == "ds-restore-btn":
        return None
    if not contents:
        return None
    try:
        src = _decode_upload(contents)
        claims, info = load_claims(src)
        return {"ok": True, "filename": filename or "file",
                "stats": {"Rows read": info["rows_read"],
                           "Self-pay excluded": info["self_pay_excluded"],
                           "Insurance claims loaded": len(claims)},
                "errors": []}
    except SchemaError as exc:
        return {"ok": False, "filename": filename or "file", "stats": {}, "errors": [str(exc)]}
    except Exception as exc:
        return {"ok": False, "filename": filename or "file", "stats": {}, "errors": ["Unexpected error: " + str(exc)]}


@app.callback(
    Output("ds-denials-result", "data"),
    Input("ds-denials-upload", "contents"),
    State("ds-denials-upload", "filename"),
    Input("ds-restore-btn", "n_clicks"),
    prevent_initial_call=True,
)
def validate_denials_upload(contents, filename, _restore):
    from src.claims_processor import SchemaError, load_denials
    if ctx.triggered_id == "ds-restore-btn":
        return None
    if not contents:
        return None
    try:
        src = _decode_upload(contents)
        denials, info = load_denials(src)
        extras = {}
        if info.get("denial_rows_without_claim_id"):
            extras["Rows missing claim ID (dropped)"] = info["denial_rows_without_claim_id"]
        if info.get("duplicate_denial_rows_dropped"):
            extras["Duplicate rows dropped"] = info["duplicate_denial_rows_dropped"]
        stats = {"Rows read": info["rows_read"], "Denial records loaded": len(denials)}
        stats.update(extras)
        return {"ok": True, "filename": filename or "file", "stats": stats, "errors": []}
    except SchemaError as exc:
        return {"ok": False, "filename": filename or "file", "stats": {}, "errors": [str(exc)]}
    except Exception as exc:
        return {"ok": False, "filename": filename or "file", "stats": {}, "errors": ["Unexpected error: " + str(exc)]}


@app.callback(
    Output("ds-claims-result-ui", "children"),
    Input("ds-claims-result", "data"),
)
def render_claims_result(data):
    if not data:
        return None
    return _result_ui(data["filename"], data["ok"], data["stats"], data["errors"])


@app.callback(
    Output("ds-denials-result-ui", "children"),
    Input("ds-denials-result", "data"),
)
def render_denials_result(data):
    if not data:
        return None
    return _result_ui(data["filename"], data["ok"], data["stats"], data["errors"])


# ------------------------------------------------------------------------------ AI investigator

_TOOL_LABELS = {
    # get_claim_details
    "claim_id":            "Claim ID",
    "insurance_provider":  "Payer",
    "payment_method":      "Payment method",
    "billed_amount":       "Billed amount",
    "claim_billing_date":  "Billing date",
    # get_denial_history
    "has_denial":                "Has denial record",
    "denial_reason_code":        "Denial reason code",
    "denial_reason_description": "Denial reason",
    "denied_amount":             "Denied amount",
    "denial_date":               "Denial date",
    "appeal_filed":              "Appeal filed",
    "appeal_status":             "Appeal status",
    "final_outcome":             "Final outcome",
    # get_payer_rules / payer_stats
    "payer":              "Payer",
    "claims":             "Total claims",
    "denied":             "Denied claims",
    "denial_rate_pct":    "Denial rate",
    "limitation":         "Limitation",
    # get_similar_claims
    "comparison":                 "Comparison basis",
    "peer_count":                 "Peer claim count",
    "peer_median_billed_amount":  "Peer median billed",
    "selected_billed_amount":     "This claim billed",
}

_DOLLAR_KEYS = {"billed_amount", "denied_amount", "peer_median_billed_amount",
                "selected_billed_amount"}
_RATE_KEYS   = {"denial_rate_pct"}

_TOOL_DISPLAY = {
    "get_claim_details":  "Claim Details",
    "get_denial_history": "Denial History",
    "get_payer_rules":    "Payer Rules",
    "get_similar_claims": "Similar Claims",
}


def _fmt_field(key: str, val) -> str:
    """Format a single tool-result value for the two-column display table."""
    if val is None:
        return "\u2014"
    if key in _DOLLAR_KEYS:
        try:
            return f"${float(val):,.2f}"
        except (TypeError, ValueError):
            pass
    if key in _RATE_KEYS:
        try:
            return f"{float(val):.1f}%"
        except (TypeError, ValueError):
            pass
    # ISO date strings: "2025-03-05" -> "Mar 5, 2025"
    if isinstance(val, str) and len(val) == 10 and val[4] == "-" and val[7] == "-":
        try:
            import datetime
            return datetime.date.fromisoformat(val).strftime("%b %-d, %Y")
        except ValueError:
            pass
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val)


def _tool_table(tool_result: dict) -> html.Table:
    """Render a tool result dict as a clean two-column label/value table."""
    rows = []
    for key, val in tool_result.items():
        if isinstance(val, (dict, list)):
            continue  # nested objects handled separately
        label = _TOOL_LABELS.get(key, key.replace("_", " ").capitalize())
        rows.append(html.Tr([
            html.Td(label, className="ev-lbl"),
            html.Td(_fmt_field(key, val), className="ev-val"),
        ]))
    # Render nested lists (e.g. top_denial_reasons)
    for key, val in tool_result.items():
        if isinstance(val, list) and val:
            label = _TOOL_LABELS.get(key, key.replace("_", " ").capitalize())
            sub_rows = []
            for item in val:
                if isinstance(item, dict):
                    sub_rows.append(
                        ", ".join(
                            f"{_TOOL_LABELS.get(k, k)}: {_fmt_field(k, v)}"
                            for k, v in item.items()
                        )
                    )
                else:
                    sub_rows.append(str(item))
            rows.append(html.Tr([
                html.Td(label, className="ev-lbl"),
                html.Td(html.Ul([html.Li(r, style={"fontSize": "13px"}) for r in sub_rows]),
                        style={"padding": "5px 0"}),
            ]))
    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


def _evidence_card(entry: dict) -> html.Div:
    """One compact evidence card with a raw-data toggle."""
    tool_name = entry.get("tool", "unknown")
    tool_result = entry.get("result", {})
    raw_json = json.dumps(tool_result, indent=2, default=str)
    display = _TOOL_DISPLAY.get(tool_name, tool_name.replace("_", " ").title())
    return html.Div([
        html.Div(display, className="ev-card-title"),
        _tool_table(tool_result),
        html.Details([
            html.Summary("View raw data", className="ev-raw-toggle"),
            html.Pre(raw_json, className="ev-raw-pre"),
        ]),
    ], className="ev-card")


def _inv_empty_state() -> html.Div:
    """Shown when no investigation has been run yet."""
    bullets = [
        ("Reads claim billing fields and denial history", "No PHI is sent to any external service"),
        ("Compares against payer historical denial patterns", "Based on the loaded dataset"),
        ("Generates human-reviewable verification steps", "Not a denial prediction or billing decision"),
    ]
    return html.Div([
        html.Div("What the investigator does", className="inv-empty-title"),
        html.Div([
            html.Div([
                html.Div(html.Span(str(i + 1), className="inv-empty-num"), className="inv-empty-icon-wrap"),
                html.Div([
                    html.Div(b[0], className="inv-empty-bullet-head"),
                    html.Div(b[1], className="inv-empty-bullet-sub"),
                ]),
            ], className="inv-empty-bullet")
            for i, b in enumerate(bullets)
        ], className="inv-empty-bullets"),
        html.Div("Select a claim above and click Investigate to begin.", className="inv-empty-cta"),
    ], className="inv-empty-state")


def investigator_layout() -> html.Div:
    claim_ids = sorted(QUEUE["claim_id"].dropna().unique())

    # Status chip
    if _AI_KEY_PRESENT:
        status_chip = html.Span(
            [html.Span(className="inv-dot inv-dot-on"), f"AI connected \u00b7 {_AI_MODEL}"],
            className="inv-status-chip connected",
        )
    else:
        status_chip = html.Span(
            [html.Span(className="inv-dot inv-dot-off"), "AI unavailable"],
            className="inv-status-chip unavailable",
            title="Set the OPENAI_API_KEY environment variable to enable AI investigation",
        )

    # Top 4 open claims by priority
    top_open = (
        QUEUE[QUEUE["is_open"] == True]
        .sort_values("priority_score", ascending=False)
        .head(4)
    )
    suggest_ids = [str(r["claim_id"]) for _, r in top_open.iterrows()]
    chips = []
    for i, (_, row) in enumerate(top_open.iterrows()):
        cid = str(row["claim_id"])
        payer = str(row.get("insurance_provider") or "")
        amt = row.get("denied_amount")
        amt_str = f"${float(amt):,.0f}" if pd.notna(amt) else "\u2014"
        chips.append(
            html.Button(
                [
                    html.Span(cid, className="inv-chip-id"),
                    html.Span(f"{payer} \u00b7 {amt_str}", className="inv-chip-meta"),
                ],
                id={"type": "inv-suggest-chip", "index": i},
                className="inv-suggest-chip",
                n_clicks=0,
            )
        )

    return html.Div([
        # Page header with status chip
        html.Div([
            html.Div([
                html.H3("AI Investigator", style={"margin": "0 0 2px"}),
                html.Div(
                    "Gathers evidence using AI-assisted tool calls. Results require human review.",
                    className="caption",
                ),
            ]),
            status_chip,
        ], className="inv-page-header"),

        # Stores
        dcc.Store(id="inv-suggest-ids", data=suggest_ids),
        dcc.Store(id="inv-report-store"),
        dcc.Download(id="inv-download"),

        # Claim picker
        html.Div([
            html.Div("Claim ID", className="field-label"),
            dcc.Dropdown(
                id="inv-claim-select",
                options=[{"label": str(c), "value": str(c)} for c in claim_ids],
                placeholder="Type or pick a claim ID\u2026",
                clearable=True,
                searchable=True,
            ),
        ], style={"marginBottom": "12px"}),

        # Suggested claim chips
        html.Div([
            html.Div("Suggested \u2014 top open claims by priority", className="field-label",
                     style={"marginBottom": "6px"}),
            html.Div(chips, className="inv-suggest-row"),
        ], style={"marginBottom": "16px"}) if chips else None,

        # Context card (populated by callback when claim is selected)
        html.Div(id="inv-context-card"),

        # Run button
        html.Button("Investigate", id="inv-run-btn", className="btn",
                    style={"marginTop": "12px", "marginBottom": "16px"}),

        # Result area (starts with empty state)
        dcc.Loading(
            html.Div(id="inv-result", children=[_inv_empty_state()]),
            type="circle",
            color="#0A6B63",
            overlay_style={"visibility": "visible"},
            custom_spinner=html.Div([
                html.Div(className="inv-spinner"),
                html.Div([
                    html.Div("Reading claim details", className="inv-step active"),
                    html.Div("Comparing payer history", className="inv-step"),
                    html.Div("Analyzing denial patterns", className="inv-step"),
                    html.Div("Generating recommendations", className="inv-step"),
                ], className="inv-step-list"),
            ], className="inv-progress-panel"),
        ),

        # Action buttons row (hidden until report is ready)
        html.Div([
            html.Button("Copy report", id="inv-copy-btn", className="btn-outline"),
            html.Button("Download .txt", id="inv-download-btn", className="btn-outline"),
            html.Button("Investigate another", id="inv-another-btn", className="btn-ghost"),
        ], id="inv-actions-row", style={"display": "none", "gap": "8px",
                                        "marginTop": "12px", "flexWrap": "wrap"}),
    ])


@app.callback(
    Output("inv-result", "children"),
    Output("inv-report-store", "data"),
    Output("inv-actions-row", "style"),
    Input("inv-run-btn", "n_clicks"),
    State("inv-claim-select", "value"),
    prevent_initial_call=True,
)
def run_investigation(n_clicks, claim_id):
    import datetime
    claim_id = (claim_id or "").strip()
    if not claim_id:
        return (
            html.Div("Please select or enter a claim ID before running.", className="cw-warn"),
            no_update,
            {"display": "none"},
        )
    from src.agent import investigate
    result = investigate(claim_id)
    status = result.get("status", "error")
    summary = result.get("summary", "")

    if status == "unavailable":
        return (
            html.Div([
                html.Div("AI investigator unavailable",
                         style={"fontWeight": "700", "marginBottom": "4px"}),
                html.Div("Set OPENAI_API_KEY to enable AI investigation.", className="caption"),
            ], className="cw-warn"),
            no_update,
            {"display": "none"},
        )
    if status == "error":
        return (
            html.Div(["Error: ", summary], className="cw-warn"),
            no_update,
            {"display": "none"},
        )

    ts = datetime.datetime.now().strftime("%b %-d, %Y at %-I:%M %p")
    trace = result.get("tool_trace", [])

    # Report card
    report_card = html.Div([
        html.Div([
            html.Div([
                html.Span(f"Claim {claim_id}", className="inv-report-claim-id"),
                html.Span(ts, className="inv-report-ts"),
            ], className="inv-report-meta"),
            html.Span("Human review required", className="badge inv-human-review"),
        ], className="inv-report-header"),
        dcc.Markdown(summary, className="inv-report-body"),
    ], className="inv-report-card card")

    # Evidence panel
    if trace:
        evidence_section = html.Div([
            html.Div(
                f"Evidence \u2014 {len(trace)} tool{'s' if len(trace) != 1 else ''} called",
                className="inv-evidence-heading",
            ),
            html.Div([_evidence_card(e) for e in trace], className="inv-evidence-grid"),
        ], className="inv-evidence-section")
    else:
        evidence_section = None

    result_div = html.Div([report_card, evidence_section])
    actions_style = {"display": "flex", "gap": "8px", "marginTop": "12px", "flexWrap": "wrap"}
    return result_div, summary, actions_style


@app.callback(
    Output("inv-claim-select", "value"),
    Input({"type": "inv-suggest-chip", "index": ALL}, "n_clicks"),
    State("inv-suggest-ids", "data"),
    prevent_initial_call=True,
)
def chip_select_claim(n_clicks_list, suggest_ids):
    if not ctx.triggered_id or not any(n_clicks_list):
        return no_update
    idx = ctx.triggered_id.get("index", 0)
    if suggest_ids and idx < len(suggest_ids):
        return suggest_ids[idx]
    return no_update


@app.callback(
    Output("inv-context-card", "children"),
    Input("inv-claim-select", "value"),
    prevent_initial_call=True,
)
def show_context_card(claim_id):
    if not claim_id:
        return []
    row = QUEUE[QUEUE["claim_id"].astype(str) == str(claim_id)]
    if row.empty:
        return []
    r = row.iloc[0]
    payer = str(r.get("insurance_provider") or "\u2014")
    amt = r.get("denied_amount")
    amt_str = f"${float(amt):,.0f}" if pd.notna(amt) else "\u2014"
    stage = str(r.get("work_status") or "\u2014")
    priority = str(r.get("priority_level") or "\u2014")
    reason = str(r.get("reason_label") or r.get("denial_reason_description") or "\u2014")
    if len(reason) > 60:
        reason = reason[:57] + "\u2026"
    return html.Div([
        html.Div("Claim context", className="field-label", style={"marginBottom": "8px"}),
        html.Div([
            html.Div([html.Span("Payer", className="ctx-label"), html.Span(payer, className="ctx-value")]),
            html.Div([html.Span("Denied", className="ctx-label"), html.Span(amt_str, className="ctx-value ctx-money")]),
            html.Div([html.Span("Reason", className="ctx-label"), html.Span(reason, className="ctx-value")]),
            html.Div([html.Span("Stage", className="ctx-label"), badge(stage)]),
            html.Div([html.Span("Priority", className="ctx-label"), badge(priority)]),
        ], className="inv-ctx-grid"),
    ], className="inv-ctx-card", style={"marginBottom": "12px"})


@app.callback(
    Output("inv-download", "data"),
    Input("inv-download-btn", "n_clicks"),
    State("inv-report-store", "data"),
    State("inv-claim-select", "value"),
    prevent_initial_call=True,
)
def download_report(n_clicks, report_text, claim_id):
    if not report_text:
        return no_update
    fname = f"investigation_{claim_id or 'unknown'}.txt"
    return {"content": report_text, "filename": fname, "type": "text/plain"}


@app.callback(
    Output("inv-result", "children", allow_duplicate=True),
    Output("inv-claim-select", "value", allow_duplicate=True),
    Output("inv-actions-row", "style", allow_duplicate=True),
    Output("inv-context-card", "children", allow_duplicate=True),
    Input("inv-another-btn", "n_clicks"),
    prevent_initial_call=True,
)
def investigate_another(n):
    hidden = {"display": "none"}
    return _inv_empty_state(), None, hidden, []


app.clientside_callback(
    """
    function(n, reportText) {
        if (!n || !reportText) return window.dash_clientside.no_update;
        try {
            navigator.clipboard.writeText(reportText);
        } catch(e) {
            var ta = document.createElement('textarea');
            ta.value = reportText;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("inv-copy-btn", "title"),
    Input("inv-copy-btn", "n_clicks"),
    State("inv-report-store", "data"),
    prevent_initial_call=True,
)


# ------------------------------------------------------------------------------ method
def method_layout() -> html.Div:
    bands = pd.DataFrame(
        [{"Level": "Clean", "Score": "0", "Meaning": "No rule fired"},
         {"Level": "Low", "Score": f"1-{LEVEL_BANDS[0][0] - 1}", "Meaning": "One or more minor items"},
         {"Level": "Medium", "Score": f"{LEVEL_BANDS[0][0]}-{LEVEL_BANDS[1][0] - 1}", "Meaning": "Meaningful issue or pattern"},
         {"Level": "High", "Score": f"{LEVEL_BANDS[1][0]}-{SCORE_CAP}", "Meaning": "Several issues or a serious one"}])
    rules_df = pd.DataFrame([{"Rule": r.label, "Type": "Confirmed" if r.kind == "confirmed" else "Indicator",
                              "Weight": r.weight, "What it checks": r.explanation} for r in RULES.values()])
    band_df = denial_rate_by_amount_band(MERGED).rename(
        columns={"billed_amount_band": "Billed amount band", "claims": "Claims", "denial_rate_pct": "Denial rate (%)"})

    # Three summary cards
    summary_cards = html.Div([
        html.Div([
            html.Div("01", className="hw-card-num"),
            html.Div("Claim checker", className="hw-card-title"),
            html.Div(
                f"Each claim is scored 0\u2013{SCORE_CAP} by rules that fire on billing data. "
                f"Claims with any confirmed issue or score \u2265 {REVIEW_SCORE_THRESHOLD} are flagged for review.",
                className="hw-card-body",
            ),
        ], className="hw-summary-card"),
        html.Div([
            html.Div("02", className="hw-card-num"),
            html.Div("Appeal & rework queue", className="hw-card-title"),
            html.Div(
                f"Open denials are ranked by priority (0\u2013100) based on denied amount, "
                f"appeal stage, and age. Older high-value items with no appeal score highest.",
                className="hw-card-body",
            ),
        ], className="hw-summary-card"),
        html.Div([
            html.Div("03", className="hw-card-num"),
            html.Div("AI investigator", className="hw-card-title"),
            html.Div(
                "A bounded LLM agent reads permitted billing fields and payer history, "
                "then produces human-reviewable verification steps. Not a denial prediction.",
                className="hw-card-body",
            ),
        ], className="hw-summary-card"),
    ], className="hw-summary-cards")

    # Design-choices callout
    design_callout = html.Div([
        html.Div([
            html.Span("!", className="hw-callout-icon"),
            html.Div([
                html.Div("These weights are design choices, not industry standards", className="hw-callout-title"),
                html.Div(
                    "Risk levels, rule weights, and review thresholds were chosen for this prototype and have not been "
                    "validated against real outcomes. A hospital should review and tune them with its own billing staff.",
                    className="hw-callout-body",
                ),
            ]),
        ], className="hw-callout-inner"),
    ], className="hw-callout")

    # Priority score equation chips
    score_eq = html.Div([
        html.Div("Priority score =", className="hw-eq-label"),
        html.Span(f"Denied $ (0\u2013{AMOUNT_MAX_POINTS})", className="hw-eq-chip"),
        html.Span("+", className="hw-eq-op"),
        html.Span("Appeal stage (0, 20, or 30)", className="hw-eq-chip"),
        html.Span("+", className="hw-eq-op"),
        html.Span(f"Age (0\u2013{AGE_MAX_POINTS} over {AGE_CAP_DAYS} days)", className="hw-eq-chip"),
        html.Span("\u2264 100", className="hw-eq-op"),
    ], className="hw-score-eq")

    # Denial-rate chart with takeaway
    dr_fig = pc.payer_fig(
        denial_rate_by_payer(MERGED).head(8),
        float(MERGED["has_denial_record"].mean() * 100),
    )

    def _accordion_section(title: str, body, open_: bool = False) -> html.Details:
        return html.Details([
            html.Summary(title, className="hw-acc-summary"),
            html.Div(body, className="hw-acc-body"),
        ], open=open_, className="hw-accordion")

    return html.Div([
        page_header("How it works",
                    "How the risk engine, appeal queue, and AI investigator work, and what they cannot determine."),

        summary_cards,

        _accordion_section("Risk scoring & claim checker", [
            design_callout,
            html.Div("Risk levels", className="hw-sub-title"),
            simple_table(bands),
            html.Div(
                "Each rule fires as a confirmed data issue (a fact visible in the record) or a risk indicator "
                f"(a pattern associated with denials, not a prediction). Score is capped at {SCORE_CAP}. "
                f"High-amount threshold: {money(HIGH_AMOUNT_THRESHOLD)}.",
                className="caption", style={"marginTop": "8px"},
            ),
            html.Div("Rule weights", className="hw-sub-title", style={"marginTop": "16px"}),
            simple_table(rules_df),
        ], open_=True),

        _accordion_section("Appeal & rework queue", [
            html.Div("How priority is calculated", className="hw-sub-title"),
            score_eq,
            html.Div(
                f"Denied amount contributes up to {AMOUNT_MAX_POINTS} points (capped at {money(AMOUNT_CAP)}). "
                f"Appeal stage adds 30 pts (no appeal) or 20 pts (pending). "
                f"Age adds up to {AGE_MAX_POINTS} pts over {AGE_CAP_DAYS} days. "
                "Payer appeal deadlines vary, so age is a rough urgency signal only.",
                className="caption", style={"marginTop": "8px"},
            ),
        ]),

        _accordion_section("Denial patterns in the loaded data", [
            html.Div("Denial rate by payer", className="hw-sub-title"),
            dcc.Graph(figure=dr_fig, config=GRAPH_CONFIG, className="hw-chart"),
            html.Div("Denial rate by billed-amount band", className="hw-sub-title",
                     style={"marginTop": "16px"}),
            simple_table(band_df),
            html.Div(
                "Rates are nearly flat across amount bands, so billed amount alone does not predict denials. "
                "A predictive model trained on this data would not be honest.",
                className="caption", style={"marginTop": "6px"},
            ),
        ]),

        _accordion_section("Payer alerts & email", [
            html.Div(
                "The payer feed is synthetic and read from a local CSV. Email is preview-only unless an SMTP "
                "server is configured in environment variables. Neither is a live integration.",
                className="caption",
            ),
        ]),

        _accordion_section("Limitations", [
            html.Ul([
                html.Li("Rules are generic examples, not any payer's actual policy."),
                html.Li("All data is synthetic. Dollar figures describe the loaded dataset only."),
                html.Li("Next-step guidance is general and must be verified against payer policy by a human."),
                html.Li("The AI investigator is a risk indicator tool, not a denial prediction."),
                html.Li("No PHI is included in this prototype."),
            ], className="hw-limits-list"),
        ]),
    ])


# ------------------------------------------------------------------------------ data sources placeholder
_UPLOAD_STYLE = {
    "width": "100%", "minHeight": "90px", "border": "2px dashed var(--border)",
    "borderRadius": "8px", "display": "flex", "alignItems": "center",
    "justifyContent": "center", "cursor": "pointer", "padding": "16px",
    "background": "var(--bg-subtle)", "boxSizing": "border-box",
}


def _upload_zone(upload_id: str, label: str) -> html.Div:
    return dcc.Upload(
        id=upload_id,
        children=html.Div([
            html.Span("Drag and drop or ", style={"color": "var(--text-muted)", "fontSize": "14px"}),
            html.Span("browse", style={"color": "var(--teal-600)", "fontWeight": "600",
                                       "fontSize": "14px", "textDecoration": "underline"}),
            html.Div(label, style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "4px"}),
        ], style={"textAlign": "center"}),
        style=_UPLOAD_STYLE,
        accept=".csv",
        multiple=False,
    )


def datasources_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H3("Data sources", style={"margin": "0 0 4px"}),
            html.Div(
                "Upload your own CSV files to validate their schema and preview load statistics. "
                "The running app uses the bundled synthetic demo data; uploaded files are validated "
                "here only and do not replace the live dataset.",
                className="caption",
            ),
        ], style={"marginBottom": "20px"}),

        # Currently loaded card
        html.Div([
            html.Div("Currently loaded (bundled demo)", className="card-section-label",
                     style={"fontWeight": "700", "fontSize": "13px", "color": "var(--text-secondary)",
                            "textTransform": "uppercase", "letterSpacing": "0.06em", "marginBottom": "12px"}),
            kpi_row([
                ("Insurance claims", f"{len(CLAIMS):,}"),
                ("Self-pay rows excluded", f"{CLAIMS_INFO['self_pay_excluded']:,}"),
                ("Data as of", f"{AS_OF:%b %-d, %Y}"),
                ("Denial records", f"{len(DENIALS):,}"),
            ]),
            html.Div(
                f"{CLAIMS_INFO['rows_read']:,} rows read from claims_and_billing.csv "
                f"({CLAIMS_INFO['self_pay_excluded']:,} self-pay excluded). "
                f"Latest denial date: {AS_OF:%b %-d, %Y}.",
                className="caption", style={"marginTop": "8px"},
            ),
        ], className="card"),

        # Upload section
        html.Div([
            html.Div("Validate a custom dataset", style={"fontWeight": "700", "fontSize": "15px",
                                                          "color": "var(--text-primary)", "marginBottom": "4px"}),
            html.Div(
                "Both files must use the exact column names from the bundled demo CSVs. "
                "Schema errors are shown in plain English below each upload zone.",
                className="caption", style={"marginBottom": "16px"},
            ),
            html.Div([
                # Claims upload
                html.Div([
                    html.Div("Claims file (claims_and_billing.csv)",
                             style={"fontWeight": "600", "fontSize": "13px",
                                    "color": "var(--text-primary)", "marginBottom": "8px"}),
                    _upload_zone("ds-claims-upload", "CSV with insurance claims and billing data"),
                    html.Div(id="ds-claims-result-ui", style={"marginTop": "10px"}),
                ], className="card", style={"flex": "1 1 0"}),
                # Denials upload
                html.Div([
                    html.Div("Denials file (denials.csv)",
                             style={"fontWeight": "600", "fontSize": "13px",
                                    "color": "var(--text-primary)", "marginBottom": "8px"}),
                    _upload_zone("ds-denials-upload", "CSV with denial records and appeal status"),
                    html.Div(id="ds-denials-result-ui", style={"marginTop": "10px"}),
                ], className="card", style={"flex": "1 1 0"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

            html.Div(
                html.Button("Restore bundled demo data", id="ds-restore-btn",
                            className="btn-outline", n_clicks=0),
                style={"marginTop": "16px"},
            ),
        ]),
    ])


# ------------------------------------------------------------------------------ navigation sidebar
_NAV_PAGES = ["overview", "payer", "appeal", "checker", "investigator", "datasources", "notify", "method"]

_LOGO_MARK_SVG = (
    '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="28" height="28" rx="7" fill="#0A6B63"/>'
    '<path d="M9 14C9 11.24 11.24 9 14 9C15.66 9 17.13 9.82 18.04 11.07" '
    'stroke="white" stroke-width="1.8" stroke-linecap="round"/>'
    '<circle cx="14" cy="14" r="3" stroke="white" stroke-width="1.8"/>'
    '<path d="M16.5 16.5L19.5 19.5" stroke="white" stroke-width="1.8" stroke-linecap="round"/>'
    '</svg>'
)

sidebar = html.Nav([
    html.Div([
        html.Div([
            _icon(_LOGO_MARK_SVG),
            html.Span("ClaimWise AI", className="cw-nav-brand-text"),
        ], className="cw-logo-area"),
        html.Button(
            _icon(_CHEVRON_L_SVG),
            id="nav-collapse-btn",
            className="cw-nav-collapse-btn",
            title="Collapse navigation",
            n_clicks=0,
            **{"aria-label": "Collapse navigation"},
        ),
    ], className="cw-nav-brand"),
    html.Div([
        html.Div("INSIGHTS", className="cw-nav-group-label"),
        nav_item("overview", "Overview", _OVERVIEW_ICON),
        nav_item("payer", "Payer alerts", _PAYER_ICON),
        html.Div("WORK", className="cw-nav-group-label"),
        nav_item("appeal", "Appeal queue", _APPEAL_ICON),
        nav_item("checker", "Claim checker", _CHECKER_ICON),
        nav_item("investigator", "AI Investigator", _INVESTIGATOR_ICON),
        html.Div("ADMINISTRATION", className="cw-nav-group-label"),
        nav_item("datasources", "Data sources", _DATASOURCES_ICON),
        nav_item("notify", "Notifications", _NOTIFY_ICON),
        nav_item("method", "How it works", _METHOD_ICON),
    ], id="cw-nav-links"),
], className="cw-sidebar", id="cw-sidebar")

topbar = html.Div([
    html.Div([
        html.Span(f"{len(CLAIMS):,} claims", style={"fontWeight": "600"}),
        html.Span(f"data as of {AS_OF:%b %-d, %Y}", style={}),
    ], className="cw-topbar-meta"),
    html.Span("IBM LinuxONE", className="cw-linuxone-badge"),
    html.Span("Synthetic demo data", className="cw-demo-pill"),
    html.Button(
        id="theme-toggle-btn",
        className="cw-theme-btn",
        children=[_icon(_SUN_SVG), html.Span("Light", className="cw-theme-label")],
        title="Switch to dark mode",
        **{"aria-label": "Switch theme"},
    ),
], className="cw-topbar")

# ------------------------------------------------------------------------------ layout
app.layout = html.Div([
    dcc.Store(id="status-apl", storage_type="session", data={}),
    dcc.Store(id="status-chk", storage_type="session", data={}),
    dcc.Store(id="sim-updates", storage_type="session", data=[]),
    dcc.Store(id="nav-page", storage_type="session", data="overview"),
    dcc.Store(id="theme-store", storage_type="local", data="light"),
    dcc.Store(id="ds-claims-result", storage_type="session", data=None),
    dcc.Store(id="ds-denials-result", storage_type="session", data=None),
    dcc.Store(id="nav-state", storage_type="local", data=False),
    sidebar,
    html.Div([
        topbar,
        html.Div(id="cw-page-content", className="cw-page-content"),
        html.Div(
            "Decision support only. Results need human review and are not a substitute for payer policy verification. "
            "A flag is not a determination that a claim will be denied or paid.",
            className="cw-footer-disclaimer",
        ),
    ], className="cw-main"),
], className="cw-shell")


# ------------------------------------------------------------------------------ theme callbacks
@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(n, current):
    return "light" if (current or "light") == "dark" else "dark"


@app.callback(
    Output("theme-toggle-btn", "children"),
    Input("theme-store", "data"),
)
def update_theme_icon(theme):
    if theme == "dark":
        return [_icon(_MOON_SVG), html.Span("Dark", className="cw-theme-label")]
    return [_icon(_SUN_SVG), html.Span("Light", className="cw-theme-label")]


# ------------------------------------------------------------------------------ navigation callbacks
@app.callback(
    Output("nav-page", "data"),
    Input({"type": "nav-btn", "page": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def navigate(n_clicks):
    triggered = ctx.triggered_id
    if triggered:
        return triggered["page"]
    return no_update


@app.callback(
    Output("cw-page-content", "children"),
    Output({"type": "nav-btn", "page": ALL}, "className"),
    Input("nav-page", "data"),
)
def render_page(page):
    pages = {
        "overview": overview_layout,
        "payer": payer_layout,
        "appeal": appeal_layout,
        "checker": checker_layout,
        "investigator": investigator_layout,
        "datasources": datasources_layout,
        "notify": notifications_layout,
        "method": method_layout,
    }
    classes = ["cw-nav-item cw-nav-active" if p == page else "cw-nav-item" for p in _NAV_PAGES]
    content = pages.get(page, overview_layout)()
    return content, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClaimWise AI dashboard (Dash)")
    parser.add_argument("--host", default="127.0.0.1", help="use 127.0.0.1 with an SSH tunnel (default)")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
