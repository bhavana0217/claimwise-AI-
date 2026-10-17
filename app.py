"""ClaimWise AI - Streamlit dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src import charts
from src.appeal_queue import (
    AMOUNT_CAP,
    AMOUNT_MAX_POINTS,
    AGE_CAP_DAYS,
    AGE_MAX_POINTS,
    CLOSED_STATUSES,
    GUIDANCE_DISCLAIMER,
    OPEN_STATUSES,
    STATUS_POINTS,
    build_appeal_queue,
)
from src.claims_processor import SchemaError, load_claims, load_denials, merge_claims_denials
from src.financial_analytics import (
    appeal_summary,
    claim_check_summary,
    denial_rate_by_amount_band,
    denial_rate_by_payer,
    denial_summary,
    denied_by_month,
    denied_dollars_by_stage,
    denials_by_reason,
)
from src.notifications import (
    NotificationSettings,
    build_email,
    evaluate_events,
    parse_recipients,
    send_email,
    smtp_settings_from_env,
)
from src.payer_alerts import SEVERITIES, UPDATE_COLUMNS, alert_impact, clean_updates, load_updates, new_update, suggested_action
from src.risk_engine import (
    HIGH_AMOUNT_THRESHOLD,
    LEVEL_BANDS,
    REVIEW_SCORE_THRESHOLD,
    RULES,
    SCORE_CAP,
    analyze_claims,
    breakdown_frame,
    score_if_corrected,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
CLAIMS_PATH = DATA_DIR / "claims_and_billing.csv"
DENIALS_PATH = DATA_DIR / "denials.csv"
SAMPLE_PATH = DATA_DIR / "sample_claims.csv"
UPDATES_PATH = DATA_DIR / "payer_rule_updates.csv"
OUTBOX_DIR = APP_DIR / "output" / "outbox"
REVIEW_STATES = ["Open", "In review", "Resolved"]
MAX_ROWS_SHOWN = 500
RISK_LEVELS = ["Clean", "Low", "Medium", "High"]

st.set_page_config(page_title="ClaimWise AI", page_icon=str(APP_DIR / "assets" / "favicon.png"), layout="wide")

CSS = """
<style>
.block-container { padding-top: 3.6rem; max-width: 1400px; }
.cw-header { display:flex; align-items:center; gap:16px; padding:16px 22px; border-radius:14px;
  background: linear-gradient(90deg, #12344D 0%, #0F766E 100%); color:#fff; margin-bottom:12px; }
.cw-logo { width:46px; height:46px; border-radius:11px; background:#fff; display:flex; align-items:center; justify-content:center; flex:none; }
.cw-title { font-size:1.55rem; font-weight:700; line-height:1.15; color:#fff; }
.cw-sub { font-size:.92rem; color:#D6E7EA; margin-top:2px; }
.cw-pill { margin-left:auto; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
  border-radius:999px; padding:4px 14px; font-size:.8rem; color:#fff; white-space:nowrap; }
.cw-notice { border-left:4px solid #B7791F; background:#FFF8EB; color:#5A4210; padding:10px 16px;
  border-radius:8px; font-size:.88rem; margin-bottom:6px; }
.cw-demo { border-left:4px solid #0F766E; background:#E8F4F2; color:#0B4F49; padding:10px 16px;
  border-radius:8px; font-size:.88rem; margin:4px 0 10px 0; }
[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #DCE6EC; border-radius:12px;
  padding:14px 18px; box-shadow:0 1px 2px rgba(18,52,77,.06); }
[data-testid="stMetricLabel"] p { color:#5B6B7B; font-size:.82rem; font-weight:600; }
[data-testid="stMetricValue"] { color:#12344D; }
button[data-baseweb="tab"] { font-weight:600; }
[data-testid="stExpander"] { background:#FFFFFF; border:1px solid #DCE6EC; border-radius:12px; }
[data-testid="stSidebar"] { border-right:1px solid #DCE6EC; }
h2, h3 { color:#12344D; }
.cw-badge { display:inline-block; padding:2px 11px; border-radius:999px; font-size:.8rem; font-weight:700; }
.cw-badge.high { background:#FDE8E4; color:#9A2A12; }
.cw-badge.medium { background:#FEF1D6; color:#7A4B00; }
.cw-badge.low { background:#E3EEF9; color:#1E4F80; }
.cw-badge.clean { background:#E4F4EA; color:#1F6B45; }
.cw-badge.closed { background:#EEF1F4; color:#4B5A69; }
.cw-id { font-size:1.15rem; font-weight:700; color:#12344D; }
.cw-meta { color:#5B6B7B; font-size:.9rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def money(x: float) -> str:
    return f"${x:,.0f}"


def esc(text: str) -> str:
    """Escape $ so markdown does not treat dollar amounts as math formulas."""
    return str(text).replace("$", "\\$")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def badge(level: str) -> str:
    return f'<span class="cw-badge {level.lower()}">{level}</span>'


@st.cache_data(show_spinner="Loading and analyzing data...")
def load_main(claims_bytes: bytes | None, denials_bytes: bytes | None):
    claims_src = io.BytesIO(claims_bytes) if claims_bytes else CLAIMS_PATH
    denials_src = io.BytesIO(denials_bytes) if denials_bytes else DENIALS_PATH
    claims, claims_info = load_claims(claims_src)
    denials, denials_info = load_denials(denials_src)
    merged = merge_claims_denials(claims, denials)
    queue = build_appeal_queue(merged)
    latest = merged["denial_date"].max()
    return {
        "claims": claims,
        "claims_info": claims_info,
        "denials_info": denials_info,
        "merged": merged,
        "queue": queue,
        "as_of": latest if pd.notna(latest) else pd.Timestamp.today().normalize(),
    }


@st.cache_data(show_spinner="Checking claims...")
def check_claims(claims: pd.DataFrame) -> pd.DataFrame:
    return analyze_claims(claims)


@st.cache_data
def load_sample():
    return load_claims(SAMPLE_PATH)


def default_updates() -> pd.DataFrame:
    try:
        return load_updates(UPDATES_PATH)
    except FileNotFoundError:
        return clean_updates(pd.DataFrame(columns=UPDATE_COLUMNS))


def status_key(prefix: str, claim_id) -> str:
    return f"{prefix}:{claim_id}"


def with_review_status(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    store = st.session_state["review_status"]
    # The selectbox's own session value is updated before the script reruns, so read it first;
    # otherwise the table and CSV would lag one interaction behind the dropdown.
    out["review_status"] = [
        st.session_state.get(f"sel_{status_key(prefix, c)}", store.get(status_key(prefix, c), "Open"))
        for c in out["claim_id"]
    ]
    return out


def status_editor(prefix: str, claim_id) -> None:
    store = st.session_state["review_status"]
    key = status_key(prefix, claim_id)
    current = store.get(key, "Open")
    choice = st.selectbox(
        "Reviewer status",
        REVIEW_STATES,
        index=REVIEW_STATES.index(current),
        key=f"sel_{key}",
        help=(
            "Your team's own progress marker for this claim: Open, In review, or Resolved. "
            "It shows in the table and in the CSV download. It is kept only for this browser "
            "session, and it does not change any claim data or contact the payer."
        ),
    )
    store[key] = choice


def table(display: pd.DataFrame, key: str, config: dict, height: int = 380) -> int | None:
    """Show a selectable table. Returns the clicked row's position, or None."""
    event = st.dataframe(
        display,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config=config,
        key=key,
        height=height,
    )
    rows = event.selection.rows if event is not None else []
    if rows and rows[0] < len(display):
        return int(rows[0])
    return None


def pick_filter(df: pd.DataFrame, label_to_col: dict, key: str, default_dim: str, default_values=None):
    """One filter at a time: choose what to filter by, then choose values. Returns filtered df."""
    dims = list(label_to_col)
    dim = st.radio("Filter by", dims, horizontal=True, key=f"{key}_dim", index=dims.index(default_dim))
    spec = label_to_col[dim]
    if spec is None:
        return df, dim
    col, options = spec
    picks = st.multiselect(
        dim,
        options,
        default=default_values if dim == default_dim and default_values else None,
        key=f"{key}_{dim}",
        placeholder="Choose one or more (leave empty to show everything)",
    )
    if picks:
        df = df[df[col].isin(picks)]
    return df, dim


MONEY_FMT = "dollar"

# ----------------------------------------------------------------------------- state
st.session_state.setdefault("review_status", {})
st.session_state.setdefault("payer_updates", default_updates())

# ----------------------------------------------------------------------------- header
st.markdown(
    '<div class="cw-header"><div class="cw-logo">'
    '<svg viewBox="0 0 24 24" width="28" height="28"><path d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6z" fill="#0F766E"/></svg>'
    '</div><div><div class="cw-title">ClaimWise AI</div>'
    '<div class="cw-sub">Revenue cycle review assistant for hospital billing teams</div></div>'
    '<div class="cw-pill">Synthetic demo data</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="cw-notice"><strong>Decision support only.</strong> Results need human review and are '
    "not a substitute for payer policy verification. A flag is not a determination that a claim will "
    "be denied or paid, and &ldquo;billed value requiring review&rdquo; is an estimate, not a "
    "guaranteed financial loss.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- sidebar
st.sidebar.markdown("### Data")
use_upload = st.sidebar.checkbox("Upload my own files instead", value=False)
claims_upload = denials_upload = None
if use_upload:
    claims_upload = st.sidebar.file_uploader("Claims CSV", type="csv", key="claims_up")
    denials_upload = st.sidebar.file_uploader("Denials CSV", type="csv", key="denials_up")
    if claims_upload is None or denials_upload is None:
        st.info("Upload both a claims CSV and a denials CSV in the sidebar to begin.")
        st.stop()

try:
    data = load_main(
        claims_upload.getvalue() if claims_upload else None,
        denials_upload.getvalue() if denials_upload else None,
    )
except SchemaError as exc:
    st.error(str(exc))
    st.stop()
except FileNotFoundError:
    st.error(
        "Could not find data/claims_and_billing.csv and data/denials.csv. Copy the two datathon "
        "files into the data/ folder, or tick 'Upload my own files' in the sidebar."
    )
    st.stop()

merged, queue, as_of = data["merged"], data["queue"], data["as_of"]
st.sidebar.success(
    f"{len(data['claims']):,} insurance claims loaded\n\n"
    f"{data['claims_info']['self_pay_excluded']:,} self-pay rows excluded (not insurance claims)"
)
st.sidebar.caption(f"Data as of {as_of:%b %d, %Y} (latest denial date in the file)")

tab_overview, tab_appeal, tab_check, tab_payer, tab_notify, tab_method = st.tabs(
    ["Overview", "Appeal & rework queue", "Claim checker", "Payer alerts", "Notifications", "Method & limits"]
)


# ----------------------------------------------------------------------------- overview
def render_overview() -> None:
    ds = denial_summary(merged)
    asum = appeal_summary(queue)
    st.subheader("Denial history")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Insurance claims", f"{ds['insurance_claims']:,}")
    c2.metric("Total billed", money(ds["total_billed"]))
    c3.metric(f"Denied claims ({ds['denial_rate_pct']:.1f}% of claims)", f"{ds['denied_claims']:,}")
    c4.metric("Denied amount", money(ds["denied_amount"]))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Open denials to work", f"{asum['open_count']:,}")
    c6.metric("Open denied value (est.)", money(asum["open_value"]),
              help="Denied dollars on denials that still need action. An estimate of what is still recoverable, not a guarantee.")
    c7.metric("Recovered on appeal (history)", money(asum["recovered_value"]))
    c8.metric("Written off (history)", money(asum["written_off_value"]))
    st.caption(
        f"Of {asum['recovered_count'] + asum['written_off_count']:,} appeals with a final result, "
        f"{asum['resolved_appeal_win_rate_pct']:.1f}% were approved. This describes this dataset only and is not a forecast."
    )

    by_payer = denial_rate_by_payer(merged)
    by_reason = denials_by_reason(merged)
    monthly = denied_by_month(merged)

    left, right = st.columns(2)
    with left:
        st.altair_chart(charts.stage_chart(denied_dollars_by_stage(asum)))
    with right:
        st.altair_chart(charts.reasons_chart(by_reason))
    left, right = st.columns(2)
    with left:
        st.altair_chart(charts.payer_chart(by_payer, ds["denial_rate_pct"]))
        st.caption("Dashed line = overall denial rate. Payers differ only slightly in this dataset.")
    with right:
        if monthly.empty:
            st.info("No denial dates available to chart.")
        else:
            st.altair_chart(charts.monthly_chart(monthly))
            if monthly["is_partial"].any():
                st.caption("Lighter bars are months only partly covered by the data, so they are not comparable to full months.")


# ----------------------------------------------------------------------------- appeal queue
def render_appeal() -> None:
    st.subheader("Appeal & rework queue")
    st.caption(
        "Denied claims ranked for human follow-up by amount, appeal stage and time since denial. "
        "Click a row to see details below. " + GUIDANCE_DISCLAIMER
    )
    options = {
        "Denial stage": ("work_status", OPEN_STATUSES + CLOSED_STATUSES),
        "Payer": ("insurance_provider", sorted(queue["insurance_provider"].dropna().unique())),
        "Action category": ("action_category", sorted(queue["action_category"].unique())),
    }
    view, _ = pick_filter(queue, options, "apl", "Denial stage", OPEN_STATUSES)
    view = with_review_status(view, "apl")

    if view.empty:
        st.info("No denied claims match this filter.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Claims in view", f"{len(view):,}")
    m2.metric("Denied value in view", money(view["denied_amount"].fillna(0).sum()))
    m3.metric("High priority", f"{(view['priority_level'] == 'High').sum():,}")

    cols = ["claim_id", "insurance_provider", "denied_amount", "reason_label", "work_status",
            "priority_score", "priority_level", "review_status"]
    config = {
        "claim_id": "Claim ID",
        "insurance_provider": "Payer",
        "denied_amount": st.column_config.NumberColumn("Denied amount", format=MONEY_FMT),
        "reason_label": "Denial reason",
        "work_status": "Denial stage",
        "priority_score": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%.0f"),
        "priority_level": "Level",
        "action_category": "Action",
        "review_status": "Reviewer status",
    }
    display = view[cols].head(MAX_ROWS_SHOWN)
    pos = table(display, "apl_table", config)
    if len(view) > MAX_ROWS_SHOWN:
        st.caption(f"Showing the top {MAX_ROWS_SHOWN} of {len(view):,} rows. The download has all rows.")
    st.download_button("Download this view as CSV", csv_bytes(view), "claimwise_appeal_queue.csv", "text/csv")

    if pos is None:
        st.caption("Select a row in the table to inspect that claim.")
        return
    row = view.iloc[pos]
    with st.expander(f"Claim details - {row['claim_id']}", expanded=True):
        h1, h2 = st.columns([3, 1])
        h1.markdown(
            f'<div class="cw-id">{row["claim_id"]}</div><div class="cw-meta">{row["insurance_provider"]} &middot; '
            f'{row["reason_label"]} ({row["denial_reason_code"]}) &middot; denied {money(row["denied_amount"])}</div>',
            unsafe_allow_html=True,
        )
        h2.markdown(badge(row["priority_level"] if row["is_open"] else "Closed"), unsafe_allow_html=True)
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Why this priority**")
            if row["is_open"]:
                factors = pd.DataFrame(
                    [
                        {"Factor": f"Denied amount (maxes out at {money(AMOUNT_CAP)})", "Points": row["amount_points"], "Max": AMOUNT_MAX_POINTS},
                        {"Factor": f"Appeal stage: {row['work_status'].replace('Open - ', '')}", "Points": row["status_points"], "Max": max(STATUS_POINTS.values())},
                        {"Factor": f"Time since denial: {row['days_since_denial']} days (maxes out at {AGE_CAP_DAYS})", "Points": row["age_points"], "Max": AGE_MAX_POINTS},
                    ]
                )
                st.dataframe(factors, hide_index=True, column_config={"Points": st.column_config.NumberColumn(format="%.0f")})
                st.caption(f"Total priority {row['priority_score']:.0f} of 100.")
            else:
                st.write(esc(row["explanation"]))
        with right:
            st.markdown("**How to fix it**")
            st.info(f"**{row['action_category']}:** {row['recommended_next_step']}")
            status_editor("apl", row["claim_id"])


# ----------------------------------------------------------------------------- claim checker
def render_checker() -> None:
    st.subheader("Claim checker (rules-based)")
    st.caption(
        "Finds confirmed data issues (facts in the record) and risk indicators (patterns linked to "
        "denials, not predictions). No machine learning is used."
    )
    source = st.radio(
        "Claims to check",
        ["Loaded dataset", "Synthetic demo set (defects added on purpose)"],
        horizontal=True,
    )
    if source.startswith("Synthetic"):
        demo_claims, demo_info = load_sample()
        to_check = demo_claims
        st.markdown(
            f'<div class="cw-demo">These {len(demo_claims)} insurance claims are synthetic and contain deliberate '
            f"problems so you can see each rule fire. {demo_info['self_pay_excluded']} self-pay row was excluded.</div>",
            unsafe_allow_html=True,
        )
    else:
        to_check = data["claims"]

    analyzed = check_claims(to_check)
    summ = claim_check_summary(analyzed)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Claims checked", f"{summ['total_claims']:,}")
    k2.metric("Total billed", money(summ["total_billed"]))
    k3.metric(f"Flagged for review ({summ['flagged_pct']:.1f}%)", f"{summ['flagged_claims']:,}")
    k4.metric("Value to review (est.)", money(summ["billed_value_requiring_review"]),
              help="The billed amount on flagged claims. It is an estimate of value that needs a person's attention, not a predicted or guaranteed loss.")
    st.caption(
        f"{summ['confirmed_issue_claims']:,} claims have a confirmed data issue and "
        f"{summ['indicator_only_claims']:,} have a risk indicator only."
    )
    if summ["flagged_claims"] == 0:
        st.success("No claims in this set were flagged for review by the rules.")

    st.altair_chart(
        _level_chart(analyzed),
    )

    options = {
        "Needs review only": None,
        "Risk level": ("risk_level", RISK_LEVELS[1:]),
        "Payer": ("insurance_provider", sorted(analyzed["insurance_provider"].dropna().unique())),
        "All claims": None,
    }
    view = analyzed
    dims = list(options)
    dim = st.radio("Filter by", dims[:3] + ["Rule", "All claims"], horizontal=True, key="chk_dim")
    if dim == "Needs review only":
        view = view[view["needs_review"]]
    elif dim in ("Risk level", "Payer"):
        col, opts = options[dim]
        picks = st.multiselect(dim, opts, key=f"chk_{dim}", placeholder="Choose one or more (leave empty to show everything)")
        if picks:
            view = view[view[col].isin(picks)]
    elif dim == "Rule":
        picks = st.multiselect("Rule", list(RULES), key="chk_rule", format_func=lambda c: RULES[c].label,
                               placeholder="Choose one or more (leave empty to show everything)")
        if picks:
            both = view["confirmed_issues"].fillna("") + ";" + view["risk_indicators"].fillna("")
            view = view[both.str.contains("|".join(picks), regex=True)]

    view = view.sort_values(["risk_score", "billed_amount"], ascending=False)
    view = with_review_status(view, "chk")

    if view.empty:
        st.info("No claims match this filter.")
        return

    cols = ["claim_id", "insurance_provider", "billed_amount", "risk_score", "risk_level",
            "finding_type", "confirmed_issues", "risk_indicators", "review_status"]
    config = {
        "claim_id": "Claim ID",
        "insurance_provider": "Payer",
        "billed_amount": st.column_config.NumberColumn("Billed", format=MONEY_FMT),
        "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d"),
        "risk_level": "Level",
        "finding_type": "Finding",
        "confirmed_issues": "Confirmed issues",
        "risk_indicators": "Risk indicators",
        "review_status": "Reviewer status",
    }
    display = view[cols].head(MAX_ROWS_SHOWN)
    pos = table(display, "chk_table", config)
    if len(view) > MAX_ROWS_SHOWN:
        st.caption(f"Showing the top {MAX_ROWS_SHOWN} of {len(view):,} rows. The download has all rows.")

    d1, d2 = st.columns(2)
    d1.download_button("Download this view as CSV", csv_bytes(view.drop(columns=["claim_status_norm"])),
                       "claimwise_checked_claims.csv", "text/csv")
    d2.download_button("Download all analyzed claims as CSV",
                       csv_bytes(with_review_status(analyzed, "chk").drop(columns=["claim_status_norm"])),
                       "claimwise_all_analyzed_claims.csv", "text/csv")

    if pos is None:
        st.caption("Select a row in the table to inspect that claim.")
        return
    row = view.iloc[pos]
    cid = row["claim_id"] if pd.notna(row["claim_id"]) else "(no claim ID)"
    with st.expander(f"Claim details - {cid}", expanded=True):
        h1, h2 = st.columns([3, 1])
        payer = row["insurance_provider"] if pd.notna(row["insurance_provider"]) else "no payer recorded"
        amount = money(row["billed_amount"]) if pd.notna(row["billed_amount"]) else "no amount"
        h1.markdown(
            f'<div class="cw-id">{cid}</div><div class="cw-meta">{payer} &middot; billed {amount} &middot; '
            f'status {row["claim_status"] if pd.notna(row["claim_status"]) else "unknown"}</div>',
            unsafe_allow_html=True,
        )
        h2.markdown(badge(row["risk_level"]), unsafe_allow_html=True)

        after = score_if_corrected(row["confirmed_issues"], row["risk_indicators"])
        s1, s2 = st.columns(2)
        s1.metric("Risk score now", f"{row['risk_score']} / {SCORE_CAP}")
        s2.metric("Score if confirmed issues are corrected", f"{after} / {SCORE_CAP}",
                  f"{after - row['risk_score']:+d}" if row["risk_score"] else None, delta_color="inverse")

        breakdown = breakdown_frame(row["confirmed_issues"], row["risk_indicators"])
        if breakdown.empty:
            st.success("No rule-based issues found on this claim.")
        else:
            st.markdown("**Why this score, and how to fix it**")
            st.dataframe(
                breakdown, hide_index=True,
                column_config={
                    "Points": st.column_config.NumberColumn(width="small"),
                    "How to fix": st.column_config.TextColumn(width="large"),
                },
            )
            raw = int(breakdown["Points"].sum())
            note = f"Points add up to {raw}" + (f", capped at {SCORE_CAP}." if raw > SCORE_CAP else ".")
            st.caption(
                note + " Risk indicators are patterns linked to denials, not predictions. They stay in the score "
                "until a person checks the claim and decides they do not apply."
            )
        r1, r2 = st.columns([3, 1])
        r1.info(f"**Recommended next steps:** {row['recommended_next_steps']}")
        with r2:
            if pd.notna(row["claim_id"]):
                status_editor("chk", row["claim_id"])
            else:
                st.caption("Reviewer status is not tracked for claims without a claim ID.")


def _level_chart(analyzed: pd.DataFrame):
    import altair as alt

    counts = analyzed["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0).reset_index()
    counts.columns = ["level", "claims"]
    colors = {"Clean": "#2F855A", "Low": "#2B6CB0", "Medium": "#D69E2E", "High": "#C05621"}
    chart = (
        alt.Chart(counts, title="Claims by risk level")
        .mark_bar(cornerRadiusEnd=4, size=22)
        .encode(
            y=alt.Y("level:N", sort=RISK_LEVELS, title=None),
            x=alt.X("claims:Q", title="Claims"),
            color=alt.Color("level:N", scale=alt.Scale(domain=RISK_LEVELS, range=[colors[k] for k in RISK_LEVELS]), legend=None),
            tooltip=[alt.Tooltip("level:N", title="Level"), alt.Tooltip("claims:Q", title="Claims", format=",")],
        )
        .properties(height=alt.Step(36))
    )
    return chart.configure_view(strokeWidth=0).configure_title(color=charts.NAVY, anchor="start", fontSize=15)


# ----------------------------------------------------------------------------- payer alerts
def render_payer_alerts() -> None:
    st.subheader("Payer rule-change alerts")
    st.markdown(
        '<div class="cw-demo"><strong>Demo feed.</strong> These updates are synthetic examples, not real payer notices, '
        "and they are read from a local file, not in real time. In production this table would be filled from payer "
        "bulletins. Impact numbers below are real calculations on the loaded denial data.</div>",
        unsafe_allow_html=True,
    )
    updates = st.session_state["payer_updates"]
    impact = alert_impact(updates, queue, as_of)

    m1, m2, m3 = st.columns(3)
    m1.metric("Updates in feed", f"{len(impact):,}")
    m2.metric("Updates touching open denials", f"{(impact['open_denials_affected'] > 0).sum():,}" if len(impact) else "0")
    m3.metric("Open denied value touched", money(impact["open_value_affected"].sum()) if len(impact) else "$0",
              help="Open denials from the affected payer with the related denial codes. An update can overlap with others, so do not add this to other totals.")

    if impact.empty:
        st.info("There are no payer updates yet. Add one below.")
    else:
        cols = ["severity", "payer", "topic", "effective_date", "status", "open_denials_affected",
                "open_value_affected", "historical_denials_same_codes"]
        config = {
            "severity": "Severity", "payer": "Payer", "topic": "Topic",
            "effective_date": st.column_config.DateColumn("Effective", format="MMM D, YYYY"),
            "status": "Status",
            "open_denials_affected": st.column_config.NumberColumn("Open denials affected", format="%d"),
            "open_value_affected": st.column_config.NumberColumn("Open value affected", format=MONEY_FMT),
            "historical_denials_same_codes": st.column_config.NumberColumn("Past denials, same payer and codes", format="%d"),
        }
        pos = table(impact[cols], "alert_table", config, height=300)
        if pos is None:
            st.caption("Select a row to see the details and a suggested action.")
        else:
            row = impact.iloc[pos]
            with st.expander(f"{row['payer']}: {row['topic']}", expanded=True):
                st.markdown(f'{badge(row["severity"])} &nbsp; **{row["status"]}** &middot; effective {row["effective_date"]:%b %d, %Y} '
                            f'&middot; related codes {row["related_codes"]} &middot; source: {row["source"]}', unsafe_allow_html=True)
                st.write(row["summary"])
                st.info(f"**Suggested action:** {suggested_action(row)}")

    with st.expander("Simulate a new payer update arriving"):
        codes_available = sorted(queue["denial_reason_code"].dropna().unique())
        payers_available = sorted(queue["insurance_provider"].dropna().unique())
        with st.form("sim_update", clear_on_submit=True):
            f1, f2 = st.columns(2)
            payer = f1.selectbox("Payer", payers_available)
            severity = f2.selectbox("Severity", SEVERITIES, index=1)
            topic = st.text_input("Topic", placeholder="For example: Prior authorization")
            summary = st.text_area("What changed (your own words)", placeholder="Short description of the update")
            f3, f4 = st.columns(2)
            effective = f3.date_input("Effective date", value=(as_of + pd.Timedelta(days=30)).date())
            codes = f4.multiselect("Related denial codes", codes_available)
            submitted = st.form_submit_button("Add simulated update")
        if submitted:
            if not topic.strip() or not codes:
                st.warning("Please enter a topic and choose at least one related denial code.")
            else:
                n = len(st.session_state["payer_updates"]) + 1
                row = new_update(payer, topic, summary or "(no description)", effective, severity, codes, f"SIM-{n:02d}")
                st.session_state["payer_updates"] = pd.concat([st.session_state["payer_updates"], row], ignore_index=True)
                st.rerun()
        if st.button("Reset to the demo feed"):
            st.session_state["payer_updates"] = default_updates()
            st.rerun()


# ----------------------------------------------------------------------------- notifications
def render_notifications() -> None:
    st.subheader("Email notifications")
    smtp = smtp_settings_from_env()
    if smtp is None:
        st.markdown(
            '<div class="cw-demo"><strong>Preview mode.</strong> No email server is configured, so nothing will actually '
            "be sent. &ldquo;Send&rdquo; saves the message as a preview file in <code>output/outbox/</code>. "
            "See the README to connect an email server.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success(f"Email server configured ({smtp['host']}). Messages will be sent for real.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Who gets notified**")
        recipients_text = st.text_area("Administrator email addresses", placeholder="admin@hospital.org, billing.lead@hospital.org",
                                       help="Separate addresses with commas.")
        st.markdown("**What triggers an email**")
        s = NotificationSettings()
        s.high_priority_enabled = st.checkbox("High-priority open denials", value=True)
        s.high_priority_min_count = st.number_input("Notify when at least this many", 1, 10_000, 1, key="n_hp", disabled=not s.high_priority_enabled)
        s.open_value_enabled = st.checkbox("Open denied value above a limit", value=True)
        s.open_value_limit = st.number_input("Limit ($)", 0.0, 1e9, 1_000_000.0, 50_000.0, key="n_ov", disabled=not s.open_value_enabled)
        s.payer_update_enabled = st.checkbox("Payer rule update affects open denials", value=True)
        s.checker_enabled = st.checkbox("Claims flagged by the claim checker", value=True)
        s.checker_min_flagged = st.number_input("Notify when at least this many flagged", 1, 100_000, 1, key="n_ck", disabled=not s.checker_enabled)

    impact = alert_impact(st.session_state["payer_updates"], queue, as_of)
    checker = claim_check_summary(check_claims(data["claims"]))
    alerts = evaluate_events(queue, checker, impact, s)
    valid, invalid = parse_recipients(recipients_text)

    with right:
        st.markdown(f"**Alerts that would be sent now ({len(alerts)})**")
        if not alerts:
            st.success("Nothing currently meets your trigger settings.")
        for a in alerts:
            st.markdown(f"{badge(a.severity)} &nbsp; **{a.title}**", unsafe_allow_html=True)
            st.caption(esc(a.detail))

    st.divider()
    if invalid:
        st.warning("These do not look like email addresses: " + ", ".join(invalid))
    if alerts and valid:
        msg = build_email(alerts, valid, sender=(smtp or {}).get("from", "claimwise@localhost"))
        with st.expander("Preview the email", expanded=False):
            st.text(f"To: {msg['To']}\nSubject: {msg['Subject']}\n\n{msg.get_content()}")
        if st.button("Send notification email", type="primary"):
            result = send_email(msg, OUTBOX_DIR, smtp)
            saved = ""
            if result.saved_to:
                try:
                    saved = f" Saved to {Path(result.saved_to).relative_to(APP_DIR)}"
                except ValueError:
                    saved = f" Saved to {result.saved_to}"
            (st.success if result.sent else st.warning)(result.message + saved)
    else:
        st.button("Send notification email", type="primary", disabled=True,
                  help="Add at least one valid address and have at least one alert to enable sending.")
    st.caption("Emails contain totals only - no patient names, patient IDs or claim IDs. A real deployment still needs privacy and compliance review.")


# ----------------------------------------------------------------------------- method
def render_method() -> None:
    st.subheader("How it works, and what it does not do")
    st.markdown("#### Where the cut-offs and weights come from")
    st.warning(
        "The risk levels, rule weights and review threshold are **design choices made by our team for this "
        "prototype**. They are not taken from an industry standard, a payer policy or a published model, and they "
        "have not been validated against real outcomes. The rule *categories* (missing identifiers, invalid amounts, "
        "duplicate submissions, status mismatches) are common kinds of claim checks; the numbers attached to them are "
        "our judgment. A hospital should review and tune them with its own billing staff and data."
    )
    bands = pd.DataFrame(
        [{"Level": "Clean", "Score": "0", "Meaning": "No rule fired"}]
        + [
            {"Level": lbl, "Score": f"{lo}-{hi - 1}" if hi <= SCORE_CAP else f"{lo}-{SCORE_CAP}", "Meaning": m}
            for (lo, hi, lbl), m in zip(
                [(1, LEVEL_BANDS[0][0], LEVEL_BANDS[0][1]), (LEVEL_BANDS[0][0], LEVEL_BANDS[1][0], LEVEL_BANDS[1][1]),
                 (LEVEL_BANDS[1][0], LEVEL_BANDS[2][0], LEVEL_BANDS[2][1])],
                ["One or more minor items", "Meaningful issue or pattern", "Several issues or a serious one"],
            )
        ]
    )
    st.dataframe(bands, hide_index=True)
    st.markdown(
        f"""
**Claim checker.** Each rule below is either a *confirmed data issue* (a fact visible in the record) or a
*risk indicator* (a pattern associated with denials, **not** a prediction). The risk score is the sum of the weights
of the rules that fire, capped at {SCORE_CAP}. A claim needs review if it has any confirmed issue or a score of
{REVIEW_SCORE_THRESHOLD} or more. The high-amount threshold is {money(HIGH_AMOUNT_THRESHOLD)}.
"""
    )
    rules_table = pd.DataFrame(
        [{"Rule": r.label, "Type": "Confirmed" if r.kind == "confirmed" else "Indicator", "Weight": r.weight, "What it checks": r.explanation}
         for r in RULES.values()]
    )
    st.dataframe(rules_table, hide_index=True)
    st.markdown(
        f"""
**Appeal & rework queue.** Uses denials that already happened. Priority (0-100, open items only) = denied amount
(up to {AMOUNT_MAX_POINTS} points, reaching the maximum at {money(AMOUNT_CAP)}) + appeal stage (30 if no appeal was filed,
20 if pending) + age (up to {AGE_MAX_POINTS} points over {AGE_CAP_DAYS} days). These weights are also our own design
choices. Payer appeal deadlines differ, so age is only a rough urgency signal.

**Why there is no machine-learning denial score.** We tested whether fields known *before* a claim is submitted show
any signal in this dataset. They do not (live numbers below), and the denial reason is only known *after* a denial, so
using it would be data leakage. A model would not be honest here.
"""
    )
    st.markdown("**Denial rate by billed-amount band (loaded data)**")
    st.dataframe(
        denial_rate_by_amount_band(merged),
        hide_index=True,
        column_config={
            "billed_amount_band": "Billed amount band",
            "claims": st.column_config.NumberColumn("Claims", format="%d"),
            "denial_rate_pct": st.column_config.NumberColumn("Denial rate (%)", format="%.2f"),
        },
    )
    st.caption("Rates are nearly flat across bands, so billed amount does not separate denied from paid claims.")
    st.markdown(
        """
**Payer alerts and email.** The payer feed is synthetic and read from a local file. Email is preview-only unless an
email server is configured. Neither is a live integration.

**Limits.** Rules are generic examples, not any payer's actual policy. Data is synthetic. All dollar figures describe
the loaded data only. Next-step guidance is general and must be verified against payer policy by a human.
"""
    )


with tab_overview:
    render_overview()
with tab_appeal:
    render_appeal()
with tab_check:
    render_checker()
with tab_payer:
    render_payer_alerts()
with tab_notify:
    render_notifications()
with tab_method:
    render_method()
