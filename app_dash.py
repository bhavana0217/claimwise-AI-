"""ClaimWise AI - Dash dashboard.

Streamlit cannot run on IBM Z Linux (s390x) because it needs pyarrow, which has no s390x
build. This Dash version uses the same tested engine (src/) with pure-Python web libraries.

Run:  python app_dash.py --port 8050
"""
from __future__ import annotations

import argparse
import json
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
        style_table={"overflowX": "auto", "border": "1px solid #DCE6EC", "borderRadius": "10px"},
        style_header={"backgroundColor": "#F1F6F9", "color": "#5B6B7B", "fontWeight": "600", "border": "none",
                      "borderBottom": "1px solid #DCE6EC"},
        style_cell={"padding": "9px 12px", "fontSize": "14px", "border": "none", "borderBottom": "1px solid #EEF2F5",
                    "textAlign": "left", "fontFamily": "inherit", "color": "#1B2A3A"},
        style_data_conditional=styles + [{"if": {"state": "selected"}, "backgroundColor": "#D9EEEA", "border": "1px solid #0F766E"}],
        css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}],
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

app.clientside_callback(
    """
    function(collapsed) {
        var sidebar = document.getElementById('cw-sidebar');
        var main = document.querySelector('.cw-main');
        if (sidebar) sidebar.classList.toggle('collapsed', !!collapsed);
        if (main) main.classList.toggle('nav-collapsed', !!collapsed);
        return window.dash_clientside.no_update;
    }
    """,
    Output("nav-collapse-btn", "title"),
    Input("nav-collapsed", "data"),
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
APL_DIMS = {"Denial stage": "work_status", "Payer": "insurance_provider", "Action category": "action_category"}


def appeal_layout() -> html.Div:
    columns = [text_col("Claim ID", "claim_id"), text_col("Payer", "insurance_provider"),
               money_col("Denied amount", "denied_amount"), text_col("Denial reason", "reason_label"),
               text_col("Denial stage", "work_status"), int_col("Priority", "priority_score"),
               text_col("Level", "priority_level"), STATUS_COL]
    return html.Div([
        page_header(
            "Appeal & rework queue",
            "Denied claims ranked for human follow-up by amount, appeal stage, and days since denial. "
            "Click the circle at the left of a row to see details below. " + GUIDANCE_DISCLAIMER,
        ),
        html.Div("Filter by", className="field-label"),
        dcc.RadioItems(id="apl-dim", options=list(APL_DIMS), value="Denial stage", inline=True,
                       inputStyle={"marginLeft": "14px", "marginRight": "5px"}),
        html.Div("Values", className="field-label"),
        dcc.Dropdown(id="apl-values", multi=True, placeholder="Choose one or more (leave empty to show everything)"),
        html.Div(id="apl-kpis"),
        make_table("apl-table", columns, level_styles("priority_level") + stage_styles() + score_bar_styles()),
        html.Div(id="apl-note", className="caption"),
        html.Button("Download this view as CSV", id="apl-download-btn", className="btn secondary"),
        dcc.Download(id="apl-download"),
        html.Div(id="apl-detail"),
    ])


def filter_queue(dim: str, values: list[str] | None) -> pd.DataFrame:
    col = APL_DIMS.get(dim, "work_status")
    return QUEUE[QUEUE[col].isin(values)] if values else QUEUE


@app.callback(Output("apl-values", "options"), Output("apl-values", "value"), Input("apl-dim", "value"))
def apl_options(dim):
    col = APL_DIMS[dim]
    options = (OPEN_STATUSES + CLOSED_STATUSES) if col == "work_status" else sorted(QUEUE[col].dropna().unique())
    return [{"label": o, "value": o} for o in options], (OPEN_STATUSES if col == "work_status" else [])


@app.callback(Output("apl-table", "data"), Output("apl-table", "selected_rows"), Output("apl-kpis", "children"),
              Output("apl-note", "children"), Input("apl-dim", "value"), Input("apl-values", "value"),
              State("status-apl", "data"))
def apl_table(dim, values, store):
    view = apply_status(filter_queue(dim, values), store)
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


@app.callback(Output("apl-download", "data"), Input("apl-download-btn", "n_clicks"), State("apl-dim", "value"),
              State("apl-values", "value"), State("status-apl", "data"), prevent_initial_call=True)
def apl_download(_, dim, values, store):
    view = apply_status(filter_queue(dim, values), store)
    return dcc.send_data_frame(view.to_csv, "claimwise_appeal_queue.csv", index=False)


# ------------------------------------------------------------------------------ claim checker
CHK_COLS = ["row_id", "claim_id", "insurance_provider", "billed_amount", "risk_score", "risk_level",
            "finding_type", "confirmed_issues", "risk_indicators", "review_status"]
CHK_DIMS = ["Needs review only", "Risk level", "Payer", "Rule", "All claims"]


def checker_layout() -> html.Div:
    columns = [text_col("Claim ID", "claim_id"), text_col("Payer", "insurance_provider"), money_col("Billed", "billed_amount"),
               int_col("Risk score", "risk_score"), text_col("Level", "risk_level"), text_col("Finding", "finding_type"),
               text_col("Confirmed issues", "confirmed_issues"), text_col("Risk indicators", "risk_indicators"), STATUS_COL]
    return html.Div([
        page_header(
            "Claim checker",
            "Finds confirmed data issues (facts in the record) and risk indicators (patterns linked to denials, "
            "not predictions). No machine learning is used.",
        ),
        html.Div("Claims to check", className="field-label"),
        dcc.RadioItems(id="chk-source", options=[{"label": "Loaded dataset", "value": "main"},
                                                 {"label": "Synthetic demo set (defects added on purpose)", "value": "demo"}],
                       value="main", inline=True, inputStyle={"marginLeft": "14px", "marginRight": "5px"}),
        html.Div(id="chk-demo-note"),
        html.Div(id="chk-kpis"),
        dcc.Graph(id="chk-level-chart", config=GRAPH_CONFIG),
        html.Div("Filter by", className="field-label"),
        dcc.RadioItems(id="chk-dim", options=CHK_DIMS, value="Needs review only", inline=True,
                       inputStyle={"marginLeft": "14px", "marginRight": "5px"}),
        dcc.Dropdown(id="chk-values", multi=True, placeholder="Choose one or more (leave empty to show everything)",
                     style={"marginTop": "8px"}),
        html.Div(id="chk-banner"),
        make_table("chk-table", columns, level_styles("risk_level")),
        html.Div(id="chk-note", className="caption"),
        html.Button("Download this view as CSV", id="chk-download-view-btn", className="btn secondary"),
        html.Button("Download all analyzed claims as CSV", id="chk-download-all-btn", className="btn secondary"),
        dcc.Download(id="chk-download"),
        html.Div(id="chk-detail"),
    ])


def filter_checker(source: str, dim: str, values: list[str] | None) -> pd.DataFrame:
    df = ANALYZED[source].copy()
    df["row_id"] = df.index
    if dim == "Needs review only":
        df = df[df["needs_review"]]
    elif dim == "Risk level" and values:
        df = df[df["risk_level"].isin(values)]
    elif dim == "Payer" and values:
        df = df[df["insurance_provider"].isin(values)]
    elif dim == "Rule" and values:
        both = df["confirmed_issues"].fillna("") + ";" + df["risk_indicators"].fillna("")
        df = df[both.str.contains("|".join(values), regex=True)]
    return df.sort_values(["risk_score", "billed_amount"], ascending=False)


@app.callback(Output("chk-values", "options"), Output("chk-values", "value"), Output("chk-values", "style"),
              Input("chk-dim", "value"), Input("chk-source", "value"))
def chk_options(dim, source):
    df = ANALYZED[source]
    if dim == "Risk level":
        opts = [{"label": x, "value": x} for x in RISK_LEVELS[1:]]
    elif dim == "Payer":
        opts = [{"label": x, "value": x} for x in sorted(df["insurance_provider"].dropna().unique())]
    elif dim == "Rule":
        opts = [{"label": r.label, "value": c} for c, r in RULES.items()]
    else:
        return [], [], {"display": "none"}
    return opts, [], {"marginTop": "8px"}


@app.callback(Output("chk-table", "data"), Output("chk-table", "selected_rows"), Output("chk-kpis", "children"),
              Output("chk-level-chart", "figure"), Output("chk-demo-note", "children"), Output("chk-banner", "children"),
              Output("chk-note", "children"),
              Input("chk-source", "value"), Input("chk-dim", "value"), Input("chk-values", "value"),
              State("status-chk", "data"), State("theme-store", "data"))
def chk_table(source, dim, values, store, theme):
    analyzed = ANALYZED[source]
    summ = claim_check_summary(analyzed)
    kpis = html.Div([
        kpi_row([("Claims checked", f"{summ['total_claims']:,}"), ("Total billed", money(summ["total_billed"])),
                 (f"Flagged for review ({summ['flagged_pct']:.1f}%)", f"{summ['flagged_claims']:,}"),
                 ("Value to review (est.)", money(summ["billed_value_requiring_review"]))]),
        html.Div(f"{summ['confirmed_issue_claims']:,} claims have a confirmed data issue and "
                 f"{summ['indicator_only_claims']:,} have a risk indicator only. The dollar figure is the billed amount on "
                 "flagged claims - an estimate of value needing a person's attention, not a predicted or guaranteed loss.",
                 className="caption")])
    fig = pc.level_fig(analyzed["risk_level"].value_counts().to_dict(), dark=(theme == "dark"))
    demo = (html.Div(f"These {len(DEMO_CLAIMS)} insurance claims are synthetic and contain deliberate problems so you can see "
                     f"each rule fire. {DEMO_INFO['self_pay_excluded']} self-pay row was excluded.", className="cw-demo")
            if source == "demo" else None)
    banner = html.Div("No claims in this set were flagged for review by the rules.", className="cw-ok") if summ["flagged_claims"] == 0 else None
    view = apply_status(filter_checker(source, dim, values), store)
    note = f"Showing the top {MAX_ROWS:,} of {len(view):,} rows. The download has all rows." if len(view) > MAX_ROWS else (
        "" if len(view) else "No claims match this filter.")
    return records(view.head(MAX_ROWS), CHK_COLS), [], kpis, fig, demo, banner, note


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


@app.callback(Output("chk-download", "data"), Input("chk-download-view-btn", "n_clicks"),
              Input("chk-download-all-btn", "n_clicks"), State("chk-source", "value"), State("chk-dim", "value"),
              State("chk-values", "value"), State("status-chk", "data"), prevent_initial_call=True)
def chk_download(_v, _a, source, dim, values, store):
    if ctx.triggered_id == "chk-download-all-btn":
        df = filter_checker(source, "All claims", None)
        name = "claimwise_all_analyzed_claims.csv"
    else:
        df = filter_checker(source, dim, values)
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
    # Card A: active alerts summary
    card_a = html.Div([
        html.Div("Active alerts", style={"fontWeight": "700", "fontSize": "15px",
                                          "color": "var(--text-primary)", "marginBottom": "4px"}),
        html.Div("Alerts that would be included in the next email based on your current trigger settings.",
                 className="caption", style={"marginBottom": "12px"}),
        html.Div(id="n-alerts"),
    ], className="card")

    # Card B: alert rules
    card_b = html.Div([
        html.Div("Alert rules", style={"fontWeight": "700", "fontSize": "15px",
                                        "color": "var(--text-primary)", "marginBottom": "4px"}),
        html.Div("Choose which conditions trigger an email notification.",
                 className="caption", style={"marginBottom": "14px"}),
        dcc.Checklist(
            id="n-events",
            value=["hp", "ov", "pu", "ck"],
            options=[
                {"label": "High-priority open denials", "value": "hp"},
                {"label": "Open denied value above a limit", "value": "ov"},
                {"label": "Payer rule update affects open denials", "value": "pu"},
                {"label": "Claims flagged by the claim checker", "value": "ck"},
            ],
            labelStyle={"display": "flex", "alignItems": "center", "gap": "8px",
                        "padding": "8px 0", "borderBottom": "1px solid var(--border)",
                        "cursor": "pointer", "fontSize": "14px", "color": "var(--text-primary)"},
            inputStyle={"width": "16px", "height": "16px", "cursor": "pointer", "accentColor": "var(--teal-600)"},
        ),
        html.Div([
            html.Div([
                html.Div("Min high-priority denials to trigger", className="field-label",
                         style={"marginTop": "14px", "marginBottom": "4px"}),
                dcc.Input(id="n-hp-min", type="number", min=1, value=1,
                          style={"width": "100px"}),
            ]),
            html.Div([
                html.Div("Open denied value limit ($)", className="field-label",
                         style={"marginTop": "14px", "marginBottom": "4px"}),
                dcc.Input(id="n-ov-limit", type="number", min=0, value=1_000_000,
                          style={"width": "140px"}),
            ]),
            html.Div([
                html.Div("Min flagged claims to trigger", className="field-label",
                         style={"marginTop": "14px", "marginBottom": "4px"}),
                dcc.Input(id="n-ck-min", type="number", min=1, value=1,
                          style={"width": "100px"}),
            ]),
        ]),
    ], className="card")

    # Card C: recipients + delivery
    smtp_banner = (
        html.Div(f"Email server configured ({smtp['host']}). Messages will be sent for real.",
                 className="cw-ok", style={"marginBottom": "12px"})
        if smtp else
        html.Div([html.B("Preview mode. "),
                  "No email server is configured — 'Send' saves the message as a preview file in output/outbox/. "
                  "See the README to connect an email server."],
                 className="cw-demo", style={"marginBottom": "12px"})
    )
    card_c = html.Div([
        html.Div("Recipients & delivery", style={"fontWeight": "700", "fontSize": "15px",
                                                   "color": "var(--text-primary)", "marginBottom": "4px"}),
        html.Div("Emails contain totals only — no patient names, IDs, or claim numbers.",
                 className="caption", style={"marginBottom": "12px"}),
        smtp_banner,
        html.Div("Administrator email addresses (separate with commas)", className="field-label",
                 style={"marginBottom": "4px"}),
        dcc.Textarea(id="n-recipients", placeholder="admin@hospital.org, billing.lead@hospital.org",
                     style={"width": "100%", "height": "64px", "boxSizing": "border-box"}),
        html.Div(id="n-warning", style={"marginTop": "6px"}),
        html.Details([
            html.Summary("Preview email",
                         style={"cursor": "pointer", "fontWeight": "600", "fontSize": "14px",
                                "color": "var(--teal-600)", "padding": "10px 0"}),
            html.Pre(id="n-preview",
                     style={"whiteSpace": "pre-wrap", "background": "var(--bg-subtle)",
                            "border": "1px solid var(--border)", "borderRadius": "8px",
                            "padding": "12px", "fontSize": "13px", "marginTop": "6px",
                            "color": "var(--text-primary)"}),
        ], style={"marginTop": "4px"}),
        html.Div([
            html.Button("Send notification email", id="n-send", className="btn", disabled=True),
            html.Div(id="n-result", style={"marginTop": "8px"}),
        ], style={"marginTop": "14px"}),
        html.Div("A real deployment needs privacy and compliance review before sending patient-related data.",
                 className="caption", style={"marginTop": "12px", "fontStyle": "italic"}),
    ], className="card")

    return html.Div([
        html.Div([
            html.H3("Notifications", style={"margin": "0 0 4px"}),
            html.Div("Configure email alerts that fire when claims data crosses the thresholds you set.",
                     className="caption"),
        ], style={"marginBottom": "20px"}),
        html.Div([card_a, card_b, card_c],
                 style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "flex-start"}),
    ])


def build_alerts(events, hp_min, ov_limit, ck_min, sim):
    events = events or []
    settings = NotificationSettings(
        high_priority_enabled="hp" in events, high_priority_min_count=int(hp_min or 1),
        open_value_enabled="ov" in events, open_value_limit=float(ov_limit or 0),
        payer_update_enabled="pu" in events, checker_enabled="ck" in events, checker_min_flagged=int(ck_min or 1))
    return evaluate_events(QUEUE, CHECKER_SUMMARY_MAIN, current_impact(sim), settings)


NOTIFY_INPUTS = [Input("n-recipients", "value"), Input("n-events", "value"), Input("n-hp-min", "value"),
                 Input("n-ov-limit", "value"), Input("n-ck-min", "value"), Input("sim-updates", "data")]


@app.callback(Output("n-alerts", "children"), Output("n-warning", "children"), Output("n-preview", "children"),
              Output("n-send", "disabled"), *NOTIFY_INPUTS)
def notify_preview(recipients, events, hp_min, ov_limit, ck_min, sim):
    alerts = build_alerts(events, hp_min, ov_limit, ck_min, sim)
    valid, invalid = parse_recipients(recipients or "")
    shown = ([html.Div([badge(a.severity), html.B("  " + a.title), html.Div(a.detail, className="caption")], style={"marginBottom": "8px"}) for a in alerts]
             or [html.Div("Nothing currently meets your trigger settings.", className="cw-ok")])
    warning = html.Div("These do not look like email addresses: " + ", ".join(invalid), className="cw-warn") if invalid else None
    if alerts and valid:
        msg = build_email(alerts, valid, sender=(smtp_settings_from_env() or {}).get("from", "claimwise@localhost"))
        return shown, warning, f"To: {msg['To']}\nSubject: {msg['Subject']}\n\n{msg.get_content()}", False
    return shown, warning, "Add at least one valid address and have at least one alert to see a preview.", True


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
            html.Td(label, style={"fontWeight": "600", "width": "42%",
                                  "padding": "5px 10px 5px 0", "verticalAlign": "top",
                                  "color": "#374F66", "fontSize": "13px"}),
            html.Td(_fmt_field(key, val), style={"padding": "5px 0",
                                                  "fontSize": "13px", "color": "#0F2340"}),
        ]))
    # Render any nested list (e.g. top_denial_reasons) as sub-rows
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
                html.Td(label, style={"fontWeight": "600", "width": "42%",
                                      "padding": "5px 10px 5px 0", "verticalAlign": "top",
                                      "color": "#374F66", "fontSize": "13px"}),
                html.Td(html.Ul([html.Li(r, style={"fontSize": "13px"}) for r in sub_rows]),
                        style={"padding": "5px 0"}),
            ]))
    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


def _trace_card(entry: dict) -> html.Div:
    """Render one tool-trace entry as a labeled table with a raw-JSON toggle."""
    tool_name = entry.get("tool", "unknown")
    tool_result = entry.get("result", {})
    raw_json = json.dumps(tool_result, indent=2, default=str)
    return html.Div([
        html.Div(tool_name, style={"fontWeight": "700", "fontSize": "13px",
                                   "color": "#0A6B63", "marginBottom": "8px",
                                   "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        _tool_table(tool_result),
        html.Details([
            html.Summary("View raw data",
                         style={"cursor": "pointer", "fontSize": "12px",
                                "color": "#536678", "marginTop": "8px"}),
            html.Pre(raw_json,
                     style={"fontSize": "11px", "background": "#F4F7FA",
                            "padding": "8px", "borderRadius": "6px",
                            "overflowX": "auto", "marginTop": "4px"}),
        ]),
    ], className="card", style={"marginBottom": "10px"})


def investigator_layout() -> html.Div:
    claim_ids = sorted(QUEUE["claim_id"].dropna().unique())
    return html.Div([
        page_header(
            "AI Investigator",
            "Gathers evidence from claim data using AI-assisted tool calls. Results require human review.",
            html.Div([html.B("Experimental \u2014 "),
                      "requires OPENAI_API_KEY. Never discloses PHI. Not a denial prediction."],
                     className="cw-demo"),
        ),
        html.Div("Select a claim from the appeal queue", className="field-label"),
        dcc.Dropdown(
            id="inv-claim-dropdown",
            options=[{"label": str(c), "value": str(c)} for c in claim_ids],
            placeholder="Select a claim from the appeal queue...",
            clearable=True,
        ),
        html.Div("Or type a claim ID directly", className="field-label",
                 style={"marginTop": "8px"}),
        dcc.Input(id="inv-claim-input", type="text", placeholder="Claim ID", debounce=True,
                  style={"width": "100%", "padding": "8px 12px", "borderRadius": "8px",
                         "border": "1px solid #DCE6EC", "fontSize": "14px"}),
        html.Button("Investigate", id="inv-run-btn", className="btn",
                    style={"marginTop": "12px"}),
        dcc.Loading(
            html.Div(id="inv-result"),
            type="circle",
            color="#0A6B63",
            overlay_style={"visibility": "visible"},
            custom_spinner=html.Div([
                html.Div(style={"width": "28px", "height": "28px",
                                "border": "3px solid #E0F2F0",
                                "borderTop": "3px solid #0A6B63",
                                "borderRadius": "50%",
                                "animation": "spin 0.8s linear infinite",
                                "margin": "0 auto 10px auto"}),
                html.Div("Investigating claim\u2026",
                         style={"color": "#374F66", "fontSize": "14px",
                                "fontWeight": "600"}),
            ], style={"textAlign": "center", "padding": "32px 0"}),
        ),
    ])


@app.callback(
    Output("inv-result", "children"),
    Input("inv-run-btn", "n_clicks"),
    State("inv-claim-dropdown", "value"),
    State("inv-claim-input", "value"),
    prevent_initial_call=True,
)
def run_investigation(n_clicks, dropdown_val, input_val):
    claim_id = (dropdown_val or input_val or "").strip()
    if not claim_id:
        return html.Div("Please select or enter a claim ID before running.",
                        className="cw-warn")
    from src.agent import investigate
    result = investigate(claim_id)
    status = result.get("status", "error")
    summary = result.get("summary", "")
    if status == "unavailable":
        return html.Div(summary, className="cw-warn")
    if status == "error":
        return html.Div(["Error: ", summary], className="cw-warn")
    trace = result.get("tool_trace", [])
    trace_cards = [_trace_card(e) for e in trace] if trace else [
        html.Div("No tools were called.", className="caption"),
    ]
    return html.Div([
        # Markdown renders headings, bullets, numbered lists from the agent
        dcc.Markdown(summary, className="cw-info",
                     style={"lineHeight": "1.65"}),
        html.Div(
            [html.B("Human review required \u2014 "),
             "this is AI-generated analysis, not a clinical or billing determination."],
            className="cw-notice",
            style={"marginTop": "8px"},
        ),
        html.Details([
            html.Summary(
                f"Tool trace \u2014 {len(trace)} tool call{'s' if len(trace) != 1 else ''}"
                " (expand to see evidence gathered)",
                style={"cursor": "pointer", "fontWeight": "600", "padding": "8px 0"},
            ),
            html.Div(trace_cards, style={"marginTop": "8px"}),
        ], className="card", style={"marginTop": "12px"}),
    ])


# ------------------------------------------------------------------------------ method
def method_layout() -> html.Div:
    bands = pd.DataFrame(
        [{"Level": "Clean", "Score": "0", "Meaning": "No rule fired"},
         {"Level": "Low", "Score": f"1-{LEVEL_BANDS[0][0] - 1}", "Meaning": "One or more minor items"},
         {"Level": "Medium", "Score": f"{LEVEL_BANDS[0][0]}-{LEVEL_BANDS[1][0] - 1}", "Meaning": "Meaningful issue or pattern"},
         {"Level": "High", "Score": f"{LEVEL_BANDS[1][0]}-{SCORE_CAP}", "Meaning": "Several issues or a serious one"}])
    rules = pd.DataFrame([{"Rule": r.label, "Type": "Confirmed" if r.kind == "confirmed" else "Indicator",
                           "Weight": r.weight, "What it checks": r.explanation} for r in RULES.values()])
    band_df = denial_rate_by_amount_band(MERGED).rename(
        columns={"billed_amount_band": "Billed amount band", "claims": "Claims", "denial_rate_pct": "Denial rate (%)"})
    return html.Div([
        page_header("Method & limits",
                    "How the risk engine and investigation tools work, and what they cannot determine."),
        html.H4("Where the cut-offs and weights come from"),
        html.Div([html.B("The risk levels, rule weights and review threshold are design choices made by our team for this prototype. "),
                  "They are not taken from an industry standard, a payer policy or a published model, and they have not been "
                  "validated against real outcomes. The rule categories (missing identifiers, invalid amounts, duplicate "
                  "submissions, status mismatches) are common kinds of claim checks; the numbers attached to them are our "
                  "judgment. A hospital should review and tune them with its own billing staff and data."], className="cw-warn"),
        simple_table(bands),
        dcc.Markdown(
            f"**Claim checker.** Each rule is either a *confirmed data issue* (a fact visible in the record) or a *risk indicator* "
            f"(a pattern associated with denials, **not** a prediction). The risk score is the sum of the weights of the rules that "
            f"fire, capped at {SCORE_CAP}. A claim needs review if it has any confirmed issue or a score of {REVIEW_SCORE_THRESHOLD} "
            f"or more. The high-amount threshold is {money(HIGH_AMOUNT_THRESHOLD)}."),
        simple_table(rules),
        dcc.Markdown(
            f"**Appeal & rework queue.** Uses denials that already happened. Priority (0-100, open items only) = denied amount "
            f"(up to {AMOUNT_MAX_POINTS} points, maximum at {money(AMOUNT_CAP)}) + appeal stage (30 if no appeal was filed, 20 if "
            f"pending) + age (up to {AGE_MAX_POINTS} points over {AGE_CAP_DAYS} days). These weights are also our own design choices. "
            "Payer appeal deadlines differ, so age is only a rough urgency signal.\n\n"
            "**Why there is no machine-learning denial score.** We tested whether fields known *before* a claim is submitted show any "
            "signal in this dataset. They do not (live numbers below), and the denial reason is only known *after* a denial, so using "
            "it would be data leakage. A model would not be honest here."),
        html.H4("Denial rate by billed-amount band (loaded data)"),
        simple_table(band_df),
        html.Div("Rates are nearly flat across bands, so billed amount does not separate denied from paid claims.", className="caption"),
        dcc.Markdown(
            "**Payer alerts and email.** The payer feed is synthetic and read from a local file. Email is preview-only unless an email "
            "server is configured. Neither is a live integration.\n\n"
            "**Limits.** Rules are generic examples, not any payer's actual policy. Data is synthetic. All dollar figures describe the "
            "loaded data only. Next-step guidance is general and must be verified against payer policy by a human."),
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

sidebar = html.Nav([
    html.Div([
        html.Div("ClaimWise AI", className="cw-nav-brand-text", id="cw-nav-brand-text"),
        html.Button(
            _icon(_CHEVRON_L_SVG),
            id="nav-collapse-btn",
            className="cw-nav-collapse-btn",
            title="Collapse sidebar",
            n_clicks=0,
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
        nav_item("method", "Method & limits", _METHOD_ICON),
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
    dcc.Store(id="nav-collapsed", storage_type="local", data=False),
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
    Output("nav-collapsed", "data"),
    Input("nav-collapse-btn", "n_clicks"),
    State("nav-collapsed", "data"),
    prevent_initial_call=True,
)
def toggle_nav_collapsed(n, current):
    return not bool(current)


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
