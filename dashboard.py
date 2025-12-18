
# dashboard.py
# MAO Task Tracker Dashboard (Streamlit)
# Final: Monthly/Quarterly views show Task IDs; Weekly → Individual shows descriptions.
# Adds Individual task filters, Team highlights (duration & top-by-volume), clean matrices, MonthName YYYY labels.
# No charts. Includes helper bug fixes.

import os
import re
import math
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
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
DAY_TO_OFFSET = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}

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
    HH.MM where .MM are minutes (00-59).
    '1.30' -> 90 min, '0.45' -> 45 min, '2.00' -> 120 min.
    If .MM > 59, treat as decimal hours (e.g., 1.5 -> 90 min).
    """
    s = str(s).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        if "." in s:
            hh, mm = s.split(".")
            h = int(hh)
            m = int(mm)
            if m <= 59:
                return h * 60 + m
            return int(round(float(s) * 60))  # decimal hours
        else:
            return int(s)  # pure integer -> minutes
    return None

def parse_duration_any(text) -> int:
    """Accept HH.MM, HH:MM, '2h', '90m', plain numbers."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return 0
    s = str(text).strip().lower()
    # HH.MM first
    hhmm_dot = parse_hh_dot_mm(s)
    if hhmm_dot is not None:
        return hhmm_dot
    # HH:MM
    m_colon = re.match(r"^(\d+):(\d{1,2})$", s)
    if m_colon:
        h = int(m_colon.group(1))
        m = int(m_colon.group(2))
        return h * 60 + m
    # xh ym
    h = sum(int(x) for x in re.findall(r"(\d+)\s*h", s)) if "h" in s else 0
    m = sum(int(x) for x in re.findall(r"(\d+)\s*m", s)) if "m" in s else 0
    if h or m:
        return h * 60 + m
    # plain number -> minutes
    try:
        return int(round(float(s)))
    except:
        return 0

def parse_volume(text) -> tuple[int, str | None]:
    """Parse '7 accounts', '50 emails' -> (7, 'accounts'/'emails'); blank -> (0, None)."""
    if pd.isna(text):
        return 0, None
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
    df["month_start"] = s.dt.to_period("M").dt.to_timestamp().dt.date
    df["quarter_start"] = s.dt.to_period("Q").dt.start_time.dt.date
    return df

# ------------------------------------------------------------
# Monthly/Quarterly task-level helpers (sums only; no charts)
# ------------------------------------------------------------
def ensure_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    # Derive anchors if missing
    if "month_start" not in out.columns and "date" in out.columns:
        out["month_start"] = out["date"].dt.to_period("M").dt.to_timestamp().dt.date
    if "quarter_start" not in out.columns and "date" in out.columns:
        out["quarter_start"] = out["date"].dt.to_period("Q").dt.start_time.dt.date
    # Numeric safety
    out["duration_minutes"] = pd.to_numeric(out.get("duration_minutes", 0), errors="coerce").fillna(0)
    out["volume_value"] = pd.to_numeric(out.get("volume_value", 0), errors="coerce").fillna(0)
    return out

def _latest_and_prev_month(df: pd.DataFrame, member: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    base = df[df["member"] == member]
    months = sorted(pd.to_datetime(base["month_start"], errors="coerce").dropna().unique())
    if len(months) < 2:
        return (months[-1] if months else None), None
    return months[-1], months[-2]

def build_individual_monthly_comparison(df_all: pd.DataFrame, member: str,
                                        selected_month: pd.Timestamp | None = None) -> pd.DataFrame:
    """Task-ID level: current month vs previous month (sums only)."""
    df = ensure_period_columns(df_all)
    if selected_month is None:
        curr_m, prev_m = _latest_and_prev_month(df, member)
    else:
        curr_m = pd.to_datetime(selected_month)
        prev_m = curr_m - pd.offsets.MonthBegin(1)
    if curr_m is None or prev_m is None:
        return pd.DataFrame(columns=[
            "Task ID", "Month", "Duration (min)", "Volume",
            "Prev Month", "Prev Duration (min)", "Prev Volume", "Δ Duration", "Δ Volume"
        ])

    base = df[df["member"] == member].copy()
    agg_cols = ["task_id"]

    cur = (base[pd.to_datetime(base["month_start"]) == curr_m]
           .groupby(agg_cols)
           .agg(cur_duration=("duration_minutes", "sum"),
                cur_volume=("volume_value", "sum"))
           .reset_index())

    prev = (base[pd.to_datetime(base["month_start"]) == prev_m]
            .groupby(agg_cols)
            .agg(prev_duration=("duration_minutes", "sum"),
                 prev_volume=("volume_value", "sum"))
            .reset_index())

    merged = cur.merge(prev, on=agg_cols, how="outer").fillna(0)
    merged["Δ Duration"] = merged["cur_duration"] - merged["prev_duration"]
    merged["Δ Volume"] = merged["cur_volume"] - merged["prev_volume"]

    merged["Task ID"] = merged["task_id"]
    merged["Month"] = pd.to_datetime(curr_m).strftime("%B %Y")
    merged["Prev Month"] = pd.to_datetime(prev_m).strftime("%B %Y")

    final = merged[["Task ID", "Month", "cur_duration", "cur_volume",
                    "Prev Month", "prev_duration", "prev_volume",
                    "Δ Duration", "Δ Volume"]].rename(columns={
        "cur_duration": "Duration (min)", "cur_volume": "Volume",
        "prev_duration": "Prev Duration (min)", "prev_volume": "Prev Volume",
    }).sort_values(by="Duration (min)", ascending=False)
    return final

def build_individual_quarterly_breakdown(df_all: pd.DataFrame, member: str,
                                         quarter_start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Task-ID level: month-by-month sums within a quarter (sums only)."""
    df = ensure_period_columns(df_all)
    base = df[df["member"] == member].copy()
    quarters = sorted(pd.to_datetime(base["quarter_start"], errors="coerce").dropna().unique())
    q = pd.to_datetime(quarter_start) if quarter_start is not None else (quarters[-1] if quarters else None)
    if q is None:
        return pd.DataFrame(columns=[
            "Task ID", "M1 Duration", "M1 Volume", "M2 Duration", "M2 Volume", "M3 Duration", "M3 Volume",
            "Quarter Duration", "Quarter Volume"
        ])

    m1 = q
    m2 = q + pd.offsets.MonthBegin(1)
    m3 = q + pd.offsets.MonthBegin(2)

    def agg_month(m):
        return (base[pd.to_datetime(base["month_start"]) == m]
                .groupby(["task_id"])
                .agg(duration=("duration_minutes", "sum"),
                     volume=("volume_value", "sum"))
                .reset_index())

    df_m1 = agg_month(m1).rename(columns={"duration": "M1 Duration", "volume": "M1 Volume"})
    df_m2 = agg_month(m2).rename(columns={"duration": "M2 Duration", "volume": "M2 Volume"})
    df_m3 = agg_month(m3).rename(columns={"duration": "M3 Duration", "volume": "M3 Volume"})

    merged = df_m1.merge(df_m2, on="task_id", how="outer").merge(df_m3, on="task_id", how="outer").fillna(0)
    merged["Quarter Duration"] = merged[["M1 Duration", "M2 Duration", "M3 Duration"]].sum(axis=1)
    merged["Quarter Volume"] = merged[["M1 Volume", "M2 Volume", "M3 Volume"]].sum(axis=1)

    merged["Task ID"] = merged["task_id"]
    final = merged[["Task ID", "M1 Duration", "M1 Volume", "M2 Duration", "M2 Volume", "M3 Duration", "M3 Volume",
                    "Quarter Duration", "Quarter Volume"]].sort_values(by="Quarter Duration", ascending=False)
    return final

def _month_lbl(x) -> str:
    # Full month name + year (e.g., January 2026)
    return pd.to_datetime(x).strftime("%B %Y")

# ---------- Team monthly/quarterly overview helpers ----------
def build_team_monthly_overview(df_all: pd.DataFrame, include_coverage: bool = True, top_n: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two compact matrices:
      - Duration (min) by month (rows=Task ID, cols=MonthName YYYY)
      - Volume by month (rows=Task ID, cols=MonthName YYYY)
    Sorted by total duration (descending) and limited to top_n rows.
    """
    df = ensure_period_columns(df_all)
    if not include_coverage:
        df = df[df["is_coverage"] == False]

    grp = (df.groupby(["task_id", "month_start"])
           .agg(duration=("duration_minutes", "sum"),
                volume=("volume_value", "sum"))
           .reset_index())
    if grp.empty:
        return pd.DataFrame(), pd.DataFrame()

    grp["month_ts"] = pd.to_datetime(grp["month_start"])
    grp["month_label"] = grp["month_ts"].apply(_month_lbl)

    ordered_labels = [_month_lbl(ts) for ts in sorted(grp["month_ts"].unique())]

    dur = grp.pivot(index="task_id", columns="month_label", values="duration").fillna(0)
    vol = grp.pivot(index="task_id", columns="month_label", values="volume").fillna(0)
    dur = dur.reindex(columns=ordered_labels, fill_value=0)
    vol = vol.reindex(columns=ordered_labels, fill_value=0)

    dur["__total__"] = dur.sum(axis=1)
    dur = dur.sort_values("__total__", ascending=False).drop(columns="__total__")
    vol = vol.loc[dur.index]

    if top_n:
        dur = dur.head(top_n)
        vol = vol.loc[dur.index]

    dur.index.name = "Task ID"
    vol.index.name = "Task ID"
    return dur, vol

def build_team_quarterly_overview(df_all: pd.DataFrame, include_coverage: bool = True,
                                  quarter_selected: pd.Timestamp | None = None, top_n: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two matrices for the selected quarter:
      - Duration (min) by month
      - Volume by month
    Rows limited to top_n Task IDs by quarter total duration.
    """
    df = ensure_period_columns(df_all)
    if not include_coverage:
        df = df[df["is_coverage"] == False]

    quarters = sorted(pd.to_datetime(df["quarter_start"], errors="coerce").dropna().unique())
    q = pd.to_datetime(quarter_selected) if quarter_selected is not None else (quarters[-1] if quarters else None)
    if q is None:
        return pd.DataFrame(), pd.DataFrame()

    m1, m2, m3 = q, (q + pd.offsets.MonthBegin(1)), (q + pd.offsets.MonthBegin(2))
    mask = pd.to_datetime(df["month_start"]).isin([m1, m2, m3])
    base = df[mask]
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    grp = (base.groupby(["task_id", "month_start"])
           .agg(duration=("duration_minutes", "sum"),
                volume=("volume_value", "sum"))
           .reset_index())
    grp["month_ts"] = pd.to_datetime(grp["month_start"])
    grp["month_label"] = grp["month_ts"].apply(_month_lbl)

    ordered_labels = [_month_lbl(ts) for ts in sorted(grp["month_ts"].unique())]
    dur = grp.pivot(index="task_id", columns="month_label", values="duration").fillna(0)
    vol = grp.pivot(index="task_id", columns="month_label", values="volume").fillna(0)

    dur["Quarter Total"] = dur.sum(axis=1)
    dur = dur.sort_values("Quarter Total", ascending=False)
    vol = vol.loc[dur.index]

    if top_n:
        dur = dur.head(top_n)
        vol = vol.loc[dur.index]

    dur = dur.reindex(columns=ordered_labels + ([c for c in dur.columns if c not in ordered_labels]))
    vol = vol.reindex(columns=ordered_labels + ([c for c in vol.columns if c not in ordered_labels]))

    dur.index.name = "Task ID"
    vol.index.name = "Task ID"
    return dur, vol

# ---------- Team Highlights helpers ----------
def build_team_over_threshold(df_all: pd.DataFrame, period_key: str,
                              threshold_minutes: int = 60, include_coverage: bool = True) -> pd.DataFrame:
    """
    Team view: list Task IDs with sums over a duration threshold in the selected period(s).
    """
    df = ensure_period_columns(df_all)
    if not include_coverage:
        df = df[df["is_coverage"] == False]

    grp = (df.groupby([period_key, "task_id"])
           .agg(duration=("duration_minutes", "sum"),
                volume=("volume_value", "sum"),
                members=("member", "nunique"))
           .reset_index())

    over = grp[grp["duration"] > threshold_minutes].copy()
    over = over.rename(columns={
        period_key: "Period Start",
        "task_id": "Task ID",
        "duration": "Total Duration (min)",
        "volume": "Total Volume",
        "members": "Members involved"
    }).sort_values(by="Total Duration (min)", ascending=False)

    return over[["Task ID", "Period Start", "Total Duration (min)", "Total Volume", "Members involved"]]

def build_team_top_volume(df_all: pd.DataFrame, period_key: str, top_n: int = 10, include_coverage: bool = True) -> pd.DataFrame:
    """
    Return Top-N Task IDs by total volume (with total duration & member count).
    """
    df = ensure_period_columns(df_all)
    if not include_coverage:
        df = df[df["is_coverage"] == False]

    grp = (df.groupby([period_key, "task_id"])
           .agg(total_volume=("volume_value", "sum"),
                total_duration=("duration_minutes", "sum"),
                members=("member", "nunique"))
           .reset_index())

    if grp.empty:
        return pd.DataFrame(columns=["Task ID", "Period Start", "Total Volume", "Total Duration (min)", "Members involved"])

    top = (grp.sort_values("total_volume", ascending=False)
              .head(top_n)
              .rename(columns={
                  period_key: "Period Start",
                  "task_id": "Task ID",
                  "total_volume": "Total Volume",
                  "total_duration": "Total Duration (min)",
                  "members": "Members involved"
              }))

    return top[["Task ID", "Period Start", "Total Volume", "Total Duration (min)", "Members involved"]]

# ------------------------------------------------------------
# Data I/O
# ------------------------------------------------------------
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
                raw["Task ID"] = raw["Task ID"].astype(str).str.strip()
                raw["Task Description"] = raw["Task Description"].astype(str).str.strip()

                # Parse duration & volume
                raw["duration_minutes"] = raw["Duration"].apply(parse_duration_any) if "Duration" in raw.columns else 0
                vpairs = raw["Volume"].apply(parse_volume) if "Volume" in raw.columns else [(0, None)] * len(raw)
                raw["volume_value"] = [p[0] for p in vpairs]
                raw["volume_label"] = [p[1] for p in vpairs]

                # Compute dates from file name (Friday week-ending → Monday start)
                week_end = parse_week_end_from_filename(uploaded.name)
                if not week_end:
                    st.warning("Could not parse week-ending (YYYY-MM-DD) from the file name. Please include it.")
                    st.stop()

                df = raw.rename(columns={
                    "Task ID": "task_id",
                    "Task Description": "task_name",
                    "Team Member": "member",
                    "Task Type": "task_type"
                })
                df = compute_dates_from_week_end(df, week_end)
                df["is_coverage"] = df["task_type"].eq("Coverage")
                df["task_count"] = 1
                if source_label:
                    df["source_file"] = source_label

                # Persist normalized fields (keep both task_id and task_name)
                out_cols = [
                    "date", "week_start", "month_start", "quarter_start",
                    "member", "task_id", "task_name", "task_type", "is_coverage",
                    "task_count", "duration_minutes", "volume_value", "volume_label", "source_file"
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
        period = st.selectbox("Period", ["Weekly", "Monthly", "Quarterly"])
    with c2:
        view = st.selectbox("View", ["Team", "Individual"])
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
    period_key = {"Weekly": "week_start", "Monthly": "month_start", "Quarterly": "quarter_start"}[period]
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
        chosen_period = st.selectbox("Select period", available_periods, index=len(available_periods) - 1)
        d = d[d[period_key] == chosen_period]

    # Aggregations (for KPIs/totals when shown)
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
            q = (qdt.month - 1) // 3 + 1
            label = f"Quarter: Q{q} {qdt.year}"

    st.markdown(f"**{view} — {period} — {label}**")
    if period in ["Monthly", "Quarterly"] and not only_completed and not pick_specific:
        st.caption("Showing current period as MTD/QTD until the period completes.")

    # --- KPI + totals shown only for Weekly OR Individual views ---
    show_kpis_and_totals = (period == "Weekly") or (view == "Individual")

    if show_kpis_and_totals:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Tasks", int(agg["task_count"].sum()))
        k2.metric("Total Duration (hrs)", round(agg["duration_minutes"].sum() / 60, 1))
        k3.metric("Avg Duration per Task (min)", round(agg["avg_duration_per_task"].mean(), 1))

        st.write("Totals per period")
        st.dataframe(agg, use_container_width=True)

        split = d.groupby([period_key, "task_type"]).agg({"task_count": "sum"}).reset_index()
        st.write("Split by task type (Production vs Coverage)")
        st.dataframe(
            split.pivot(index=period_key, columns="task_type", values="task_count").fillna(0),
            use_container_width=True
        )

    # --- No charts anywhere (removed) ---
    st.divider()

    # =========================
    # Individual — Monthly view (Task IDs + filter)
    # =========================
    if view == "Individual" and period == "Monthly" and member_sel:
        st.markdown("### 📅 Monthly Task Comparison (Sums Only, by Task ID)")
        selected_month = pd.to_datetime(chosen_period) if (pick_specific and chosen_period) else None
        comp = build_individual_monthly_comparison(data, member_sel, selected_month)

        if comp.empty:
            st.info("Not enough monthly data to compare for this member.")
        else:
            task_ids = sorted(comp["Task ID"].dropna().unique().tolist())
            sel = st.multiselect("Filter Task IDs", task_ids, default=task_ids)
            filtered = comp[comp["Task ID"].isin(sel)] if sel else comp.iloc[0:0]
            st.dataframe(filtered, use_container_width=True)
            st.download_button(
                "Download (CSV)",
                data=filtered.to_csv(index=False),
                file_name=f"{member_sel}_Monthly_TaskID_Comparison.csv",
                mime="text/csv"
            )

    # ==========================
    # Individual — Quarterly view (Task IDs + filter)
    # ==========================
    if view == "Individual" and period == "Quarterly" and member_sel:
        st.markdown("### 🗓 Quarterly Task Breakdown (Sums Only, by Task ID)")
        selected_quarter = pd.to_datetime(chosen_period) if (pick_specific and chosen_period) else None
        qb = build_individual_quarterly_breakdown(data, member_sel, selected_quarter)

        if qb.empty:
            st.info("No quarterly data available for this member.")
        else:
            task_ids = sorted(qb["Task ID"].dropna().unique().tolist())
            sel = st.multiselect("Filter Task IDs", task_ids, default=task_ids, key="q_task_filter_ids")
            filtered = qb[qb["Task ID"].isin(sel)] if sel else qb.iloc[0:0]
            st.dataframe(filtered, use_container_width=True)
            st.download_button(
                "Download (CSV)",
                data=filtered.to_csv(index=False),
                file_name=f"{member_sel}_Quarterly_TaskID_Breakdown.csv",
                mime="text/csv"
            )

    # ==========================
    # Team — Monthly/Quarterly view (highlights + matrices)
    # ==========================
    if view == "Team" and period in ("Monthly", "Quarterly"):
        # ---- Highlights: Tasks over duration threshold (respect Coverage toggle & chosen period) ----
        st.markdown("### ⏱ Highlights — Tasks over duration threshold (by Task ID)")
        threshold = st.slider("Minimum total duration (minutes)", 30, 480, 60, 15)
        tbl = build_team_over_threshold(d, period_key, threshold_minutes=threshold, include_coverage=include_coverage)
        if tbl.empty:
            st.success("No tasks exceeded the threshold for the selected period(s).")
        else:
            st.dataframe(tbl, use_container_width=True)
            st.download_button(
                "Download Duration Highlights (CSV)",
                data=tbl.to_csv(index=False),
                file_name=f"Team_{period}_TaskID_Over_{threshold}min.csv",
                mime="text/csv"
            )

        # ---- Highlights — Top Task IDs by volume ----
        st.markdown("### 📦 Highlights — Top Task IDs by volume")
        top_n_vol = st.slider("Show top N Task IDs by volume", 5, 30, 10, 1, key="topn_vol_ids")
        top_vol_tbl = build_team_top_volume(d, period_key, top_n=top_n_vol, include_coverage=include_coverage)
        if top_vol_tbl.empty:
            st.info("No volume highlights for the current selection.")
        else:
            st.dataframe(top_vol_tbl, use_container_width=True)
            st.download_button(
                "Download Volume Highlights (CSV)",
                data=top_vol_tbl.to_csv(index=False),
                file_name=f"Team_{period}_Top_{top_n_vol}_TaskID_Volume.csv",
                mime="text/csv"
            )

        st.divider()

        # ---- Team Overview: Month-to-month matrices (clean tables; full dataset, respects coverage) ----
        st.markdown("### 🧭 Team Overview — Month-to-month (Sums Only, by Task ID)")
        top_n = st.slider("Show top N Task IDs by total duration", 10, 100, 25, 5)

        if period == "Monthly":
            dur_mat, vol_mat = build_team_monthly_overview(
                data if include_coverage else data[data["is_coverage"] == False],
                include_coverage=include_coverage,
                top_n=top_n
            )
        else:
            selected_quarter = pd.to_datetime(chosen_period) if (pick_specific and chosen_period) else None
            dur_mat, vol_mat = build_team_quarterly_overview(
                data if include_coverage else data[data["is_coverage"] == False],
                include_coverage=include_coverage,
                quarter_selected=selected_quarter,
                top_n=top_n
            )

        if dur_mat.empty and vol_mat.empty:
            st.info("No month-to-month data available for the current selection.")
        else:
            cA, cB = st.columns(2)
            with cA:
                st.caption("Duration (min) by month")
                st.dataframe(dur_mat, use_container_width=True)
                st.download_button(
                    "Download Duration Matrix (CSV)",
                    data=dur_mat.to_csv(),
                    file_name=f"Team_{period}_TaskID_Duration_Matrix.csv",
                    mime="text/csv"
                )
            with cB:
                st.caption("Volume by month")
                st.dataframe(vol_mat, use_container_width=True)
                st.download_button(
                label="Download Coverage Tasks",
                data=coverage.to_csv(index=False),
                file_name=f"{member_sel}_CoverageTasks.csv",
                mime="text/csv"
            )

 # Day-by-day production summary
        st.markdown("### ⚙ Production Summary")
        if not prod.empty:
            prod["Day"] = pd.to_datetime(prod["date"]).dt.day_name()
            summary = (
                prod.groupby("Day")
                .agg(
                    Production_Tasks=("task_count", "sum"),
                    Total_Duration=("duration_minutes", "sum"),
                    Total_Volume=("volume_value", "sum")
                )
                .reset_index()
            )
            summary["Day"] = pd.Categorical(summary["Day"], categories=DAY_ORDER, ordered=True)
            summary = summary.sort_values("Day")
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No production summary available for this member.")
