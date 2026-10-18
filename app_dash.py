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
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
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


def kpi(label: str, value: str) -> html.Div:
    return html.Div([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value")], className="kpi")


def kpi_row(items: list[tuple[str, str]]) -> html.Div:
    return html.Div([kpi(a, b) for a, b in items], className="kpi-row")


def badge(level: str) -> html.Span:
    return html.Span(level, className=f"badge {level.lower()}")


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
LEVEL_STYLE = {"High": ("#9A2A12", "#FDE8E4"), "Medium": ("#7A4B00", "#FEF1D6"), "Low": ("#1E4F80", "#E3EEF9"),
               "Clean": ("#1F6B45", "#E4F4EA"), "Closed": ("#4B5A69", "#EEF1F4")}


def level_styles(col: str) -> list[dict]:
    return [
        {"if": {"filter_query": f'{{{col}}} = "{lvl}"', "column_id": col},
         "color": fg, "backgroundColor": bg, "fontWeight": "700"}
        for lvl, (fg, bg) in LEVEL_STYLE.items()
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


# ------------------------------------------------------------------------------ app shell
app = Dash(__name__, title="ClaimWise AI", suppress_callback_exceptions=False)
app.index_string = app.index_string.replace(
    "{%favicon%}", '<link rel="icon" type="image/png" href="/assets/favicon.png">'
)
server = app.server


# ------------------------------------------------------------------------------ overview
def overview_layout() -> html.Div:
    ds, asum = denial_summary(MERGED), appeal_summary(QUEUE)
    by_payer, by_reason, monthly = denial_rate_by_payer(MERGED), denials_by_reason(MERGED), denied_by_month(MERGED)
    partial_note = (
        "Lighter bars are months only partly covered by the data, so they are not comparable to full months."
        if len(monthly) and monthly["is_partial"].any() else ""
    )
    return html.Div([
        html.H3("Denial history"),
        kpi_row([
            ("Insurance claims", f"{ds['insurance_claims']:,}"),
            ("Total billed", money(ds["total_billed"])),
            (f"Denied claims ({ds['denial_rate_pct']:.1f}% of claims)", f"{ds['denied_claims']:,}"),
            ("Denied amount", money(ds["denied_amount"])),
            ("Open denials to work", f"{asum['open_count']:,}"),
            ("Open denied value (est.)", money(asum["open_value"])),
            ("Recovered on appeal (history)", money(asum["recovered_value"])),
            ("Written off (history)", money(asum["written_off_value"])),
        ]),
        html.Div(
            f"Of {asum['recovered_count'] + asum['written_off_count']:,} appeals with a final result, "
            f"{asum['resolved_appeal_win_rate_pct']:.1f}% were approved. This describes this dataset only and is not a forecast. "
            "Open denied value is an estimate of what is still recoverable, not a guarantee.",
            className="caption"),
        html.Div([
            dcc.Graph(figure=pc.stage_fig(denied_dollars_by_stage(asum)), config=GRAPH_CONFIG),
            dcc.Graph(figure=pc.reasons_fig(by_reason), config=GRAPH_CONFIG),
            html.Div([dcc.Graph(figure=pc.payer_fig(by_payer, ds["denial_rate_pct"]), config=GRAPH_CONFIG),
                      html.Div("Dashed line = overall denial rate. Payers differ only slightly in this dataset.", className="caption")]),
            html.Div([dcc.Graph(figure=pc.monthly_fig(monthly), config=GRAPH_CONFIG) if len(monthly) else html.Div("No denial dates to chart."),
                      html.Div(partial_note, className="caption")]),
        ], className="chart-row"),
    ])


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
        html.H3("Appeal & rework queue"),
        html.Div("Denied claims ranked for human follow-up by amount, appeal stage and time since denial. "
                 "Click the circle at the left of a row to see details below. " + GUIDANCE_DISCLAIMER, className="caption"),
        html.Div("Filter by", className="field-label"),
        dcc.RadioItems(id="apl-dim", options=list(APL_DIMS), value="Denial stage", inline=True,
                       inputStyle={"marginLeft": "14px", "marginRight": "5px"}),
        html.Div("Values", className="field-label"),
        dcc.Dropdown(id="apl-values", multi=True, placeholder="Choose one or more (leave empty to show everything)"),
        html.Div(id="apl-kpis"),
        make_table("apl-table", columns, level_styles("priority_level")),
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
        html.H3("Claim checker (rules-based)"),
        html.Div("Finds confirmed data issues (facts in the record) and risk indicators (patterns linked to denials, "
                 "not predictions). No machine learning is used.", className="caption"),
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
              State("status-chk", "data"))
def chk_table(source, dim, values, store):
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
    fig = pc.level_fig(analyzed["risk_level"].value_counts().to_dict())
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
        html.H3("Payer rule-change alerts"),
        html.Div([html.B("Demo feed. "), "These updates are synthetic examples, not real payer notices, and they are read from a "
                  "local file, not in real time. In production this table would be filled from payer bulletins. Impact numbers "
                  "are real calculations on the loaded denial data."], className="cw-demo"),
        html.Div(id="alert-kpis"),
        make_table("alert-table", columns, level_styles("severity"), page_size=8, with_status=False),
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
    banner = (html.Div(f"Email server configured ({smtp['host']}). Messages will be sent for real.", className="cw-ok") if smtp else
              html.Div([html.B("Preview mode. "), "No email server is configured, so nothing will actually be sent. 'Send' saves the "
                        "message as a preview file in output/outbox/. See the README to connect an email server."], className="cw-demo"))
    return html.Div([
        html.H3("Email notifications"), banner,
        html.Div([
            html.Div([
                html.H4("Who gets notified"),
                html.Div("Administrator email addresses (separate with commas)", className="field-label"),
                dcc.Textarea(id="n-recipients", style={"width": "100%", "height": "70px"}, placeholder="admin@hospital.org, billing.lead@hospital.org"),
                html.H4("What triggers an email"),
                dcc.Checklist(id="n-events", value=["hp", "ov", "pu", "ck"], labelStyle={"display": "block", "margin": "6px 0"},
                              inputStyle={"marginRight": "8px"},
                              options=[{"label": "High-priority open denials", "value": "hp"},
                                       {"label": "Open denied value above a limit", "value": "ov"},
                                       {"label": "Payer rule update affects open denials", "value": "pu"},
                                       {"label": "Claims flagged by the claim checker", "value": "ck"}]),
                html.Div("Notify when at least this many high-priority denials", className="field-label"),
                dcc.Input(id="n-hp-min", type="number", min=1, value=1),
                html.Div("Open denied value limit ($)", className="field-label"),
                dcc.Input(id="n-ov-limit", type="number", min=0, value=1_000_000),
                html.Div("Notify when at least this many claims are flagged", className="field-label"),
                dcc.Input(id="n-ck-min", type="number", min=1, value=1),
            ]),
            html.Div([html.H4("Alerts that would be sent now"), html.Div(id="n-alerts")]),
        ], className="split"),
        html.Hr(),
        html.Div(id="n-warning"),
        html.Details([html.Summary("Preview the email", style={"cursor": "pointer", "fontWeight": "600"}),
                      html.Pre(id="n-preview", style={"whiteSpace": "pre-wrap", "background": "#fff", "border": "1px solid #DCE6EC",
                                                      "borderRadius": "10px", "padding": "12px"})]),
        html.Button("Send notification email", id="n-send", className="btn", disabled=True),
        html.Div(id="n-result"),
        html.Div("Emails contain totals only - no patient names, patient IDs or claim IDs. A real deployment still needs privacy and compliance review.", className="footer-note"),
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


# ------------------------------------------------------------------------------ AI investigator
def investigator_layout() -> html.Div:
    claim_ids = sorted(QUEUE["claim_id"].dropna().unique())
    return html.Div([
        html.H3("AI Investigator (experimental)"),
        html.Div([
            html.B("Experimental"),
            " — requires OPENAI_API_KEY. Results need human review.",
        ], className="cw-demo"),
        html.Div("Select a claim from the appeal queue", className="field-label"),
        dcc.Dropdown(
            id="inv-claim-dropdown",
            options=[{"label": str(c), "value": str(c)} for c in claim_ids],
            placeholder="Select a claim from the appeal queue...",
            clearable=True,
        ),
        html.Div("Or type a claim ID directly", className="field-label", style={"marginTop": "8px"}),
        dcc.Input(id="inv-claim-input", type="text", placeholder="Claim ID", debounce=True,
                  style={"width": "100%", "padding": "8px 12px", "borderRadius": "8px",
                         "border": "1px solid #DCE6EC", "fontSize": "14px"}),
        html.Button("Investigate", id="inv-run-btn", className="btn", style={"marginTop": "12px"}),
        dcc.Loading(html.Div(id="inv-result"), type="circle"),
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
        return html.Div("Please select or enter a claim ID before running.", className="cw-warn")
    from src.agent import investigate
    result = investigate(claim_id)
    status = result.get("status", "error")
    summary = result.get("summary", "")
    if status == "unavailable":
        return html.Div(summary, className="cw-warn")
    if status == "error":
        return html.Div(["Error: ", summary], className="cw-warn")
    trace = result.get("tool_trace", [])
    trace_items = []
    for entry in trace:
        tool_name = entry.get("tool", "unknown")
        tool_result = entry.get("result", {})
        trace_items.append(html.Div([
            html.Div(tool_name, style={"fontWeight": "600", "marginBottom": "4px"}),
            html.Pre(json.dumps(tool_result, indent=2, default=str),
                     style={"fontSize": "12px", "background": "#F7FAFB", "padding": "8px",
                            "borderRadius": "6px", "overflowX": "auto"}),
        ], className="card", style={"marginBottom": "8px"}))
    return html.Div([
        html.Div(summary, className="cw-info"),
        html.Div(
            "Human review required — this is AI-generated analysis, not a clinical determination.",
            className="cw-notice",
            style={"marginTop": "8px"},
        ),
        html.Details([
            html.Summary("Tool trace (expand to see evidence gathered)",
                         style={"cursor": "pointer", "fontWeight": "600", "padding": "6px 0"}),
            html.Div(trace_items if trace_items else html.Div("No tools were called.", className="caption")),
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
        html.H3("How it works, and what it does not do"),
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


# ------------------------------------------------------------------------------ layout
def tab(label: str, value: str, content) -> dcc.Tab:
    return dcc.Tab(label=label, value=value, className="tab", selected_className="tab--selected", children=[html.Div(content, style={"paddingTop": "10px"})])


app.layout = html.Div([
    dcc.Store(id="status-apl", storage_type="session", data={}),
    dcc.Store(id="status-chk", storage_type="session", data={}),
    dcc.Store(id="sim-updates", storage_type="session", data=[]),
    html.Div([
        html.Div(html.Img(src="/assets/favicon.png", style={"width": "34px", "height": "34px"}), className="cw-logo"),
        html.Div([html.Div("ClaimWise AI", className="cw-title"),
                  html.Div("Revenue cycle review assistant for hospital billing teams", className="cw-sub")]),
        html.Div("Synthetic demo data", className="cw-pill"),
    ], className="cw-header"),
    html.Div([html.B("Decision support only. "), "Results need human review and are not a substitute for payer policy verification. "
              "A flag is not a determination that a claim will be denied or paid, and “billed value requiring review” is an "
              "estimate, not a guaranteed financial loss."], className="cw-notice"),
    html.Div(f"{len(CLAIMS):,} insurance claims loaded · {CLAIMS_INFO['self_pay_excluded']:,} self-pay rows excluded (not insurance "
             f"claims) · data as of {AS_OF:%b %d, %Y} (latest denial date in the file)", className="caption"),
    dcc.Tabs(id="tabs", value="overview", children=[
        tab("Overview", "overview", overview_layout()),
        tab("Appeal & rework queue", "appeal", appeal_layout()),
        tab("Claim checker", "checker", checker_layout()),
        tab("Payer alerts", "payer", payer_layout()),
        tab("Notifications", "notify", notifications_layout()),
        tab("AI Investigator", "investigator", investigator_layout()),
        tab("Method & limits", "method", method_layout()),
    ]),
], className="page")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClaimWise AI dashboard (Dash)")
    parser.add_argument("--host", default="127.0.0.1", help="use 127.0.0.1 with an SSH tunnel (default)")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
