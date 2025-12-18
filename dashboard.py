
import os
import re
import math
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="MAO Task Tracker Dashboard", layout="wide")

# ------------------------------------------------------------
# Minimal CSS polish (soft background + comfy width)
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
      background: linear-gradient(180deg, #F7FAFF 0%, #FFFFFF 80%);
    }
    .block-container {
      padding-top: 1.2rem;
      padding-bottom: 2rem;
      max-width: 1200px;
    }
    .badge {
      display:inline-block; padding:2px 8px; border-radius:12px;
      background:#EDF2FF; color:#1a3fa0; font-size:0.8rem; margin-left:6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Header: logo + title
# ------------------------------------------------------------
HERE = Path(__file__).parent
LOGO = HERE / "static" / "lpl-logo-blue.png"

col_logo, col_title = st.columns([1, 6], gap="small")
with col_logo:
    if LOGO.exists():
        try:
            st.image(str(LOGO), width=140)
        except Exception:
            st.empty()
    else:
        st.empty()

with col_title:
    st.title("MAO Task Tracker Dashboard")
    st.caption("LPL Financial — Operations")

# ------------------------------------------------------------
# Env & DB (normalize postgres:// → postgresql:// if needed)
# ------------------------------------------------------------
DB_URL = os.getenv("DATABASE_URL")
if DB_URL and DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

ADMIN_PIN = os.getenv("ADMIN_PIN", "000000")

engine = create_engine(DB_URL, pool_pre_ping=True) if DB_URL else None

# ------------------------------------------------------------
# Constants & helpers
# ------------------------------------------------------------
REQUIRED_COLS = [
    "Task ID", "Task Description", "Task Type",
    "Team Member", "Day", "Duration", "Volume",
]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_TO_OFFSET = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4}

def escape_md(text: object) -> str:
    """Escape Markdown special characters for safe rendering in task list."""
    return re.sub(r'([*_`])', r'\\\1', str(text)) if pd.notna(text) else ""

def has_required_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return (len(missing) == 0, missing)

def parse_week_end_from_filename(name: str):
    """Find YYYY-MM-DD in filename and return date (assumed week-ending, Friday)."""
    m = re.search(r"\d{4}-\d{2}-\d{2}", name)
    if not m:
        return None
    return pd.to_datetime(m.group(0)).date()

def parse_hh_dot_mm(s: str) -> int | None:
    """
    HH.MM where .MM are minutes (00-59). Examples:
    '1.30' -> 90 min, '0.45' -> 45 min, '2.00' -> 120 min.
    If .MM > 59, fall back to decimal hours (e.g., 1.5 -> 90 min).
    """
    s = str(s).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        if "." in s:
            hh, mm = s.split(".")
            h = int(hh)
            m = int(mm)
            if m <= 59:
                return h*60 + m
            # treat as decimal hours
            return int(round(float(s)*60))
        else:
            # pure integer -> assume minutes
            val = int(s)
            return val
    return None

def parse_duration_any(text) -> int:
    """Accept HH.MM, HH:MM, '2h', '90m', plain numbers."""
    if text is None or (isinstance(text, float) and math.isnan(text)): return 0
    s = str(text).strip().lower()
    # HH.MM first
    hhmm_dot = parse_hh_dot_mm(s)
    if hhmm_dot is not None: return hhmm_dot
    # HH:MM
    m_colon = re.match(r"^(\d+):(\d{1,2})$", s)
    if m_colon:
        h = int(m_colon.group(1)); m = int(m_colon.group(2)); return h*60 + m
    # xh ym
    h = sum(int(x) for x in re.findall(r"(\d+)\s*h", s)) if "h" in s else 0
    m = sum(int(x) for x in re.findall(r"(\d+)\s*m", s)) if "m" in s else 0
    if h or m: return h*60 + m
    # plain number -> minutes
    try:
        val = float(s)
        return int(round(val))
    except:
        return 0

def parse_volume(text) -> tuple[int, str | None]:
    """Parse '7 accounts', '50 emails' -> (7, 'accounts'/'emails'); blank -> (0, None)."""
    if pd.isna(text): return 0, None
    s = str(text).strip().lower()
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    val = float(m.group(1)) if m else 0
    words = re.findall(r"[a-z]+", s)
    label = words[-1] if words else None
    return int(round(val)), label

def compute_dates_from_week_end(df: pd.DataFrame, week_end: pd.Timestamp) -> pd.DataFrame:
    """Compute calendar 'date', 'week_start', 'month_start', 'quarter_start'."""
    week_start = pd.to_datetime(week_end) - pd.Timedelta(days=4)  # Monday start if week_end is Friday
    df["date"] = df["Day"].map(lambda d: (week_start + pd.Timedelta(days=DAY_TO_OFFSET.get(str(d), 0))).date())
    df["week_start"] = week_start.date()
    s = pd.to_datetime(df["date"])
    df["month_start"]   = s.dt.to_period("M").dt.to_timestamp().dt.date
    df["quarter_start"] = s.dt.to_period("Q").dt.start_time.dt.date
    return df

@st.cache_data(ttl=300)
def read_all():
    """Read all persisted rows (ops_tasks)."""
    if engine is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_table("ops_tasks", con=engine)
    except Exception:
        return pd.DataFrame()

def append_rows(df_out: pd.DataFrame):
    """Persist rows; create table on first append."""
    if engine is None:
        st.error("DATABASE_URL missing; cannot persist.")
        st.stop()
    df_out.to_sql("ops_tasks", con=engine, if_exists="append", index=False)

# ------------------------------------------------------------
# Tabs: Dashboard + Admin Upload
# ------------------------------------------------------------
tab_dash, tab_upload = st.tabs(["📊 Dashboard", "🔐 Admin Upload"])

# ------------------------------------------------------------
# Admin Upload (PIN)
# ------------------------------------------------------------
with tab_upload:
    st.subheader("Admin Upload")
    pin = st.text_input("Enter Admin PIN", type="password")
    if pin == ADMIN_PIN:
        st.success("Admin mode enabled")
        uploaded = st.file_uploader("Upload Combined Excel File (.xlsx)", type=["xlsx"])
        source_label = st.text_input("Optional: Source label (e.g., 'Week ending 2025-12-19')")
        if uploaded is not None:
            try:
                raw = pd.read_excel(uploaded, engine="openpyxl")
                ok, missing = has_required_columns(raw)
                if not ok:
                    st.error("Missing required columns: " + ", ".join(missing))
                    st.stop()

                # Normalize fields
                raw["Task Type"] = raw["Task Type"].astype(str).str.strip().str.title()
                raw["Day"] = raw["Day"].astype(str).str.strip().str.title()
                raw["Team Member"] = raw["Team Member"].astype(str).str.strip()

                # Parse duration & volume
                raw["duration_minutes"] = raw["Duration"].apply(parse_duration_any) if "Duration" in raw.columns else 0
                vpairs = raw["Volume"].apply(parse_volume) if "Volume" in raw.columns else [(0, None)]*len(raw)
                raw["volume_value"] = [p[0] for p in vpairs]
                raw["volume_label"] = [p[1] for p in vpairs]

                # Compute dates from file name (Friday week-ending → Monday start)
                week_end = parse_week_end_from_filename(uploaded.name)
                if not week_end:
                    st.warning("Could not parse week-ending (YYYY-MM-DD) from the file name. Please include it.")
                    st.stop()

                df = raw.rename(columns={
                    "Team Member": "member",
                    "Task Description": "task_name",
                    "Task Type": "task_type"
                })
                df = compute_dates_from_week_end(df, week_end)
                df["is_coverage"] = df["task_type"].eq("Coverage")
                df["task_count"] = 1
                if source_label:
                    df["source_file"] = source_label

                # Persist normalized fields
                out_cols = [
                    "date","week_start","month_start","quarter_start",
                    "member","task_name","task_type","is_coverage",
                    "task_count","duration_minutes","volume_value","volume_label","source_file"
                ]
                for c in out_cols:
                    if c not in df.columns:
                        df[c] = None
                df_out = df[out_cols]
                st.write("Preview:", df_out.head(10))

                append_rows(df_out)
                st.success(f"Inserted {len(df_out)} rows.")
                st.cache_data.clear()

            except Exception as e:
                st.error(f"Upload failed: {e}")
    else:
        st.info("Enter the Admin PIN to enable uploads.")

# ------------------------------------------------------------
# Dashboard (Team/Individual + Weekly/Monthly/Quarterly)
# ------------------------------------------------------------
with tab_dash:
    st.subheader("Performance Views")

    data = read_all()
    if data.empty:
        st.warning("No data yet. Use Admin Upload to add your first weekly file.")
        st.stop()

    # Filters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        period = st.selectbox("Period", ["Weekly","Monthly","Quarterly"])
    with c2:
        view = st.selectbox("View", ["Team","Individual"])
    with c3:
        member_sel = None
        if view == "Individual":
            member_sel = st.selectbox("Member", sorted(data["member"].dropna().unique()))
    with c4:
        only_completed = st.checkbox("Show only completed periods", value=False)

    # Coverage behavior
    include_coverage_default = (period == "Weekly")
    include_coverage = True if view == "Individual" else st.checkbox("Include Coverage (Team)", value=include_coverage_default)

    # Period key & optional picker
    period_key = {"Weekly":"week_start","Monthly":"month_start","Quarterly":"quarter_start"}[period]
    d = data.copy()
    if view == "Individual" and member_sel:
        d = d[d["member"] == member_sel]
    if not include_coverage:
        d = d[~d["is_coverage"]]

    # Completed period mask
    today = pd.Timestamp.today().date()
    def completed_mask(dframe, key):
        if key == "month_start":
            current_month_start = pd.Timestamp(today).to_period("M").start_time.date()
            return dframe[key] < current_month_start
        elif key == "quarter_start":
            current_q_start = pd.Timestamp(today).to_period("Q").start_time.date()
            return dframe[key] < current_q_start
        else:
            ts = pd.Timestamp(today)
            current_week_start = (ts - pd.Timedelta(days=ts.dayofweek)).date()
            return dframe[key] < current_week_start

    if only_completed:
        d = d[completed_mask(d, period_key)]

    available_periods = sorted(pd.Series(d[period_key].dropna().unique()).tolist())
    pick_specific = st.checkbox("Pick a specific period", value=False)
    chosen_period = None
    if pick_specific and available_periods:
        chosen_period = st.selectbox("Select period", available_periods, index=len(available_periods)-1)
        d = d[d[period_key] == chosen_period]

    # Aggregations
    agg = d.groupby(period_key).agg({
        "task_count": "sum",
        "duration_minutes": "sum",
        "volume_value": "sum"
    }).reset_index()
    agg["avg_duration_per_task"] = (agg["duration_minutes"] / agg["task_count"]).replace([pd.NA, float("inf")], 0)

    # Period label
    label = "All periods"
    if chosen_period:
        if period == "Weekly":
            label = f"Week of {pd.to_datetime(chosen_period).date()}"
        elif period == "Monthly":
            label = f"Month: {pd.to_datetime(chosen_period).strftime('%B %Y')}"
        else:
            qdt = pd.to_datetime(chosen_period)
            q = (qdt.month-1)//3 + 1
            label = f"Quarter: Q{q} {qdt.year}"

    st.markdown(f"**{view} — {period} — {label}**")
    if period in ["Monthly","Quarterly"] and not only_completed and not pick_specific:
        st.caption("Showing current period as MTD/QTD until the period completes.")

    # KPI cards
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Tasks", int(agg["task_count"].sum()))
    k2.metric("Total Duration (hrs)", round(agg["duration_minutes"].sum()/60, 1))
    k3.metric("Avg Duration per Task (min)", round(agg["avg_duration_per_task"].mean(), 1))

    # Totals table
    st.write("Totals per period")
    st.dataframe(agg, use_container_width=True)

    # Split by task type
    split = d.groupby([period_key, "task_type"]).agg({"task_count":"sum"}).reset_index()
    st.write("Split by task type (Production vs Coverage)")
    st.dataframe(split.pivot(index=period_key, columns="task_type", values="task_count").fillna(0), use_container_width=True)

    # Charts
    cA, cB = st.columns(2)
    with cA:
        st.bar_chart(agg.set_index(period_key)["task_count"], height=280, use_container_width=True)
    with cB:
        st.line_chart(agg.set_index(period_key)["avg_duration_per_task"], height=280, use_container_width=True)

    st.divider()

    # --------------------------------------------------------
    # Weekly → Individual layout (preserve your original sections)
    # --------------------------------------------------------
    if period == "Weekly" and view == "Individual" and member_sel:
        st.markdown("### ✅ Task Summary for the Week")
        wsel = d  # already filtered by member & possibly specific week
        unique_tasks = wsel.drop_duplicates(subset=["task_name"])
        if unique_tasks.empty:
            st.info("No tasks for this team member.")
        else:
            for _, row in unique_tasks[["task_name"]].iterrows():
                desc = escape_md(row["task_name"]) if pd.notna(row["task_name"]) else "(No Description)"
                st.markdown(f"- **{desc}**")

        st.markdown("### 📋 Raw Data (Production Tasks Only)")
        prod = wsel[wsel["task_type"] == "Production"]
        if not prod.empty:
            display = prod[["task_name","task_type","date","duration_minutes","volume_value","volume_label"]].copy()
            display["task_type"] = display["task_type"].str.capitalize()
            display = display.rename(columns={
                "task_name":"Task",
                "date":"Date",
                "duration_minutes":"Duration (min)",
                "volume_value":"Volume",
                "volume_label":"Volume Label",
            })
            st.dataframe(display, use_container_width=True)
            st.download_button(
                label="Download Filtered Data (Production)",
                data=display.to_csv(index=False),
                file_name=f"{member_sel}_ProductionTasks.csv",
                mime="text/csv"
            )
        else:
            st.write("No production tasks.")

        st.markdown("### 🛡 Coverage Tasks")
        coverage = (
            wsel[wsel["task_type"] == "Coverage"]
            .drop_duplicates(subset=["task_name"])[["task_name"]]
            .rename(columns={"task_name":"Task"})
        )
        if not coverage.empty:
            st.dataframe(coverage, use_container_width=True)
            st.download_button(
                label="Download Coverage Task Names",
                data=coverage.to_csv(index=False),
                file_name=f"{member_sel}_CoverageTasks.csv",
                mime="text/csv"
            )
        else:
            st.write("No coverage tasks.")

        st.markdown("### ⚙ Production Summary")
        if not prod.empty:
            prod["Day"] = pd.to_datetime(prod["date"]).dt.day_name()
            summary = (
                prod.groupby("Day")
                .agg(
                    Production_Tasks=("task_count", "sum"),
                    Total_Duration=("duration_minutes", "sum")
                )
                .reset_index()
            )
            summary["Day"] = pd            summary["Day"] = pd.Categorical(summary["Day"], categories=DAY_ORDER, ordered=True)
            summary = summary.sort_values("Day")
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No production summary available for this member.")
