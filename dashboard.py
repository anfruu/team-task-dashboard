# dashboard.py
# MAO Workflow Tracker Dashboard (Streamlit)
# - Weekly / Monthly / Quarterly / Admin Upload tabs (layout preserved)
# - Volume is NUMERIC ONLY (INTEGER everywhere). Any non-numeric volume becomes 0 on upload.
# - Weekly adds: Task Summary rename, Coverage day column, Production (Primary) rename + day filter,
#   and Production (Backup) section.
# - Monthly + Quarterly: totals + comparisons (month-over-month / quarter-over-quarter) without changing upload logic.
# - Monthly + Quarterly tables: KPI by task for Production (Primary), Production (Backup), and Coverage
#   with deltas vs prior period. (Descriptions removed in Monthly/Quarterly only.)

import os
import re
from io import BytesIO
from typing import List

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# -----------------------------
# App config (do not change title/layout)
# -----------------------------
st.set_page_config(page_title="MAO Workflow Tracker Dashboard", layout="wide")
st.title("MAO Workflow Tracker Dashboard")
st.caption("LPL Financial – Operations")

# -----------------------------
# DB / Engine
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if not DATABASE_URL:
    st.error(
        "DATABASE_URL is not set. Add it in Render Environment variables.\n\n"
        "Example: postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME?sslmode=require"
    )
    st.stop()

ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
IS_SQLITE = ENGINE.dialect.name == "sqlite"

# -----------------------------
# Helpers
# -----------------------------
def _clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def duration_to_seconds(val) -> int:
    """Accepts hh:mm:ss, mm:ss, numeric seconds, or blanks; returns int seconds."""
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if not s:
        return 0

    # If it's numeric-ish, treat as seconds
    try:
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return max(0, int(float(s)))
    except Exception:
        pass

    # Time format
    parts = s.split(":")
    try:
        parts = [int(float(p)) for p in parts]
    except Exception:
        return 0

    if len(parts) == 3:
        h, m, sec = parts
        return max(0, h * 3600 + m * 60 + sec)
    if len(parts) == 2:
        m, sec = parts
        return max(0, m * 60 + sec)

    return 0


def seconds_to_hhmmss(seconds: int) -> str:
    try:
        seconds = int(seconds)
    except Exception:
        seconds = 0
    seconds = max(0, seconds)
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def seconds_to_hours(seconds: int) -> float:
    try:
        return float(int(seconds)) / 3600.0
    except Exception:
        return 0.0


def fmt_hours(x: float) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "0.00"


def fmt_int(x: int) -> str:
    try:
        return f"{int(x):,d}"
    except Exception:
        return "0"


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def parse_week_ending(series: pd.Series) -> pd.Series:
    """
    week_ending in DB is stored as TEXT like '2026-02-06'.
    Parse it at query-time (do NOT change upload behavior).
    """
    return pd.to_datetime(series.astype(str).str.strip(), format="%Y-%m-%d", errors="coerce")


def period_delta_str(curr: float, prev: float, is_int: bool = False) -> str:
    """Delta string for st.metric."""
    if prev is None:
        return ""
    if is_int:
        try:
            return f"{int(curr) - int(prev):+d}"
        except Exception:
            return ""
    try:
        return f"{float(curr) - float(prev):+,.2f}"
    except Exception:
        return ""


def style_delta_df(df: pd.DataFrame, delta_cols: List[str]) -> pd.io.formats.style.Styler if False else object:
    """
    Return a Styler with red/green formatting for delta columns.
    IMPORTANT: No runtime pandas Styler type annotations (prevents Render crash).
    """
    if df is None or df.empty:
        return df

    def color(v):
        try:
            v = float(v)
        except Exception:
            return ""
        if v > 0:
            return "color: green;"
        if v < 0:
            return "color: red;"
        return ""

    # If pandas Styler is unavailable for some reason, just return df
    try:
        sty = df.style
        for c in delta_cols:
            if c in df.columns:
                sty = sty.applymap(color, subset=[c])
        return sty
    except Exception:
        return df


# -----------------------------
# Normalization (upload stays the same)
# -----------------------------
def normalize_tasks_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected from combined Tasks sheet:
      Task ID, Team Member, Task Description, Task Type, Role Type,
      Duration Seconds, Duration Minutes, Duration Hours, Volume, Day, Week Ending
    Canonical:
      task_id, team_member, task_description, task_type, role_type,
      duration_seconds, volume, day, week_ending
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "task_id",
                "team_member",
                "task_description",
                "task_type",
                "role_type",
                "duration_seconds",
                "volume",
                "day",
                "week_ending",
            ]
        )

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    colmap = {
        "Task ID": "task_id",
        "Team Member": "team_member",
        "Task Description": "task_description",
        "Task Type": "task_type",
        "Role Type": "role_type",
        "Duration Seconds": "duration_seconds",
        "Duration Minutes": "duration_minutes",
        "Duration Hours": "duration_hours",
        "Duration": "duration_raw",  # legacy
        "Volume": "volume",
        "Day": "day",
        "Week Ending": "week_ending",
    }

    out = pd.DataFrame()
    for src, dst in colmap.items():
        if src in df.columns:
            out[dst] = df[src]

    # Build duration_seconds if missing
    if "duration_seconds" not in out.columns and "duration_raw" in out.columns:
        out["duration_seconds"] = out["duration_raw"].apply(duration_to_seconds)

    if "duration_seconds" not in out.columns:
        sec = 0
        if "duration_minutes" in out.columns:
            sec = pd.to_numeric(out["duration_minutes"], errors="coerce").fillna(0) * 60
        if "duration_hours" in out.columns:
            sec = sec + pd.to_numeric(out["duration_hours"], errors="coerce").fillna(0) * 3600
        out["duration_seconds"] = sec

    # Clean strings
    for c in ["task_id", "team_member", "task_description", "task_type", "role_type", "day", "week_ending"]:
        if c in out.columns:
            out[c] = out[c].apply(_clean_str)

    # Volume: NUMERIC ONLY
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
    else:
        out["volume"] = 0

    # Ensure required columns exist
    for c in [
        "task_id",
        "team_member",
        "task_description",
        "task_type",
        "role_type",
        "duration_seconds",
        "volume",
        "day",
        "week_ending",
    ]:
        if c not in out.columns:
            out[c] = "" if c in ["task_id", "team_member", "task_description", "task_type", "role_type", "day", "week_ending"] else 0

    out["duration_seconds"] = pd.to_numeric(out["duration_seconds"], errors="coerce").fillna(0).astype(int)

    # Drop blanks
    out = out.dropna(subset=["task_id"]).copy()
    out = out[out["task_id"].astype(str).str.strip() != ""].copy()

    return out


def normalize_projects_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected from combined Projects sheet:
      Project Name, Owner, Start Date, End Date, Status, Days Active, Notes, Week Ending
    Canonical:
      project_name, owner, start_date, end_date, status, days_active, notes, week_ending
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["project_name", "owner", "start_date", "end_date", "status", "days_active", "notes", "week_ending"]
        )

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    colmap = {
        "Project Name": "project_name",
        "Owner": "owner",
        "Start Date": "start_date",
        "End Date": "end_date",
        "Status": "status",
        "Days Active": "days_active",
        "Notes": "notes",
        "Week Ending": "week_ending",
    }

    out = pd.DataFrame()
    for src, dst in colmap.items():
        if src in df.columns:
            out[dst] = df[src]

    for c in ["project_name", "owner", "status", "notes", "week_ending"]:
        if c in out.columns:
            out[c] = out[c].apply(_clean_str)

    # Dates: keep as string
    for c in ["start_date", "end_date"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: "" if pd.isna(x) else str(x))

    if "days_active" in out.columns:
        out["days_active"] = pd.to_numeric(out["days_active"], errors="coerce").fillna(0).astype(int)
    else:
        out["days_active"] = 0

    out = out.dropna(subset=["project_name"]).copy()
    out = out[out["project_name"].astype(str).str.strip() != ""].copy()

    return out


# -----------------------------
# DB init
# -----------------------------
def init_db():
    with ENGINE.begin() as conn:
        if IS_SQLITE:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                      task_id TEXT,
                      team_member TEXT,
                      task_description TEXT,
                      task_type TEXT,
                      role_type TEXT,
                      duration_seconds INTEGER,
                      volume INTEGER,
                      day TEXT,
                      week_ending TEXT,
                      uploaded_at TEXT
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS projects (
                      project_name TEXT,
                      owner TEXT,
                      start_date TEXT,
                      end_date TEXT,
                      status TEXT,
                      days_active INTEGER,
                      notes TEXT,
                      week_ending TEXT,
                      uploaded_at TEXT
                    )
                    """
                )
            )
        else:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                      task_id TEXT,
                      team_member TEXT,
                      task_description TEXT,
                      task_type TEXT,
                      role_type TEXT,
                      duration_seconds INTEGER,
                      volume INTEGER,
                      day TEXT,
                      week_ending TEXT,
                      uploaded_at TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS projects (
                      project_name TEXT,
                      owner TEXT,
                      start_date TEXT,
                      end_date TEXT,
                      status TEXT,
                      days_active INTEGER,
                      notes TEXT,
                      week_ending TEXT,
                      uploaded_at TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
            )
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tasks_member_week ON tasks(team_member, week_ending)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_projects_owner_week ON projects(owner, week_ending)"))

init_db()

# -----------------------------
# DB fetch helpers
# -----------------------------
def list_team_members() -> List[str]:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT team_member
                FROM tasks
                WHERE team_member IS NOT NULL AND team_member <> ''
                ORDER BY team_member
                """
            )
        ).fetchall()
    return [r[0] for r in rows]


def list_weeks_for_member(member: str) -> List[str]:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT week_ending
                FROM tasks
                WHERE team_member = :m
                  AND week_ending IS NOT NULL AND week_ending <> ''
                ORDER BY week_ending DESC
                """
            ),
            {"m": member},
        ).fetchall()
    return [r[0] for r in rows]


def fetch_week_tasks(member: str, week_ending: str) -> pd.DataFrame:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT task_id, team_member, task_description, task_type, role_type,
                       duration_seconds, volume, day, week_ending
                FROM tasks
                WHERE team_member = :m AND week_ending = :w
                """
            ),
            {"m": member, "w": week_ending},
        ).fetchall()

    df = pd.DataFrame(
        rows,
        columns=[
            "task_id",
            "team_member",
            "task_description",
            "task_type",
            "role_type",
            "duration_seconds",
            "volume",
            "day",
            "week_ending",
        ],
    )
    if not df.empty:
        df["duration_hhmmss"] = df["duration_seconds"].apply(seconds_to_hhmmss)
    return df


def fetch_week_projects(owner: str, week_ending: str) -> pd.DataFrame:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT project_name, owner, start_date, end_date, status, days_active, notes, week_ending
                FROM projects
                WHERE owner = :o AND week_ending = :w
                """
            ),
            {"o": owner, "w": week_ending},
        ).fetchall()

    return pd.DataFrame(
        rows,
        columns=["project_name", "owner", "start_date", "end_date", "status", "days_active", "notes", "week_ending"],
    )


def fetch_all_tasks(member: str) -> pd.DataFrame:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT task_id, team_member, task_description, task_type, role_type,
                       duration_seconds, volume, day, week_ending
                FROM tasks
                WHERE team_member = :m
                """
            ),
            {"m": member},
        ).fetchall()

    return pd.DataFrame(
        rows,
        columns=[
            "task_id",
            "team_member",
            "task_description",
            "task_type",
            "role_type",
            "duration_seconds",
            "volume",
            "day",
            "week_ending",
        ],
    )


def fetch_all_projects(owner: str) -> pd.DataFrame:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT project_name, owner, start_date, end_date, status, days_active, notes, week_ending
                FROM projects
                WHERE owner = :o
                """
            ),
            {"o": owner},
        ).fetchall()

    return pd.DataFrame(
        rows,
        columns=["project_name", "owner", "start_date", "end_date", "status", "days_active", "notes", "week_ending"],
    )


# -----------------------------
# Upload / overwrite helpers (unchanged behavior)
# -----------------------------
def delete_week_data(week_ending: str):
    with ENGINE.begin() as conn:
        conn.execute(text("DELETE FROM tasks WHERE week_ending = :w"), {"w": week_ending})
        conn.execute(text("DELETE FROM projects WHERE week_ending = :w"), {"w": week_ending})


def insert_tasks(df: pd.DataFrame):
    if df.empty:
        return
    with ENGINE.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO tasks (task_id, team_member, task_description, task_type, role_type,
                                       duration_seconds, volume, day, week_ending, uploaded_at)
                    VALUES (:task_id, :team_member, :task_description, :task_type, :role_type,
                            :duration_seconds, :volume, :day, :week_ending, :uploaded_at)
                    """
                    if IS_SQLITE
                    else
                    """
                    INSERT INTO tasks (task_id, team_member, task_description, task_type, role_type,
                                       duration_seconds, volume, day, week_ending)
                    VALUES (:task_id, :team_member, :task_description, :task_type, :role_type,
                            :duration_seconds, :volume, :day, :week_ending)
                    """
                ),
                {
                    "task_id": r["task_id"],
                    "team_member": r["team_member"],
                    "task_description": r["task_description"],
                    "task_type": r["task_type"],
                    "role_type": r["role_type"],
                    "duration_seconds": int(r["duration_seconds"]),
                    "volume": int(r["volume"]),
                    "day": r["day"],
                    "week_ending": r["week_ending"],
                    "uploaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                },
            )


def insert_projects(df: pd.DataFrame):
    if df.empty:
        return
    with ENGINE.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO projects (project_name, owner, start_date, end_date, status,
                                          days_active, notes, week_ending, uploaded_at)
                    VALUES (:project_name, :owner, :start_date, :end_date, :status,
                            :days_active, :notes, :week_ending, :uploaded_at)
                    """
                    if IS_SQLITE
                    else
                    """
                    INSERT INTO projects (project_name, owner, start_date, end_date, status,
                                          days_active, notes, week_ending)
                    VALUES (:project_name, :owner, :start_date, :end_date, :status,
                            :days_active, :notes, :week_ending)
                    """
                ),
                {
                    "project_name": r["project_name"],
                    "owner": r["owner"],
                    "start_date": r.get("start_date", ""),
                    "end_date": r.get("end_date", ""),
                    "status": r.get("status", ""),
                    "days_active": int(r.get("days_active", 0)),
                    "notes": r.get("notes", ""),
                    "week_ending": r.get("week_ending", ""),
                    "uploaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                },
            )


# -----------------------------
# Metrics builders (Monthly / Quarterly)
# -----------------------------
def prep_tasks_for_rollups(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["task_type_norm"] = out["task_type"].astype(str).str.strip().str.lower()
    out["role_type_norm"] = out["role_type"].astype(str).str.strip().str.lower()
    out["day"] = out["day"].astype(str).str.strip()
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
    out["duration_seconds"] = pd.to_numeric(out["duration_seconds"], errors="coerce").fillna(0).astype(int)

    out["week_date"] = parse_week_ending(out["week_ending"])
    out = out[~out["week_date"].isna()].copy()

    out["month"] = out["week_date"].dt.to_period("M").astype(str)      # "YYYY-MM"
    out["quarter"] = out["week_date"].dt.to_period("Q").astype(str)    # "YYYYQ#"
    return out


def prep_projects_for_rollups(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["week_date"] = parse_week_ending(out["week_ending"])
    out = out[~out["week_date"].isna()].copy()
    out["month"] = out["week_date"].dt.to_period("M").astype(str)
    out["quarter"] = out["week_date"].dt.to_period("Q").astype(str)
    out["status_norm"] = out["status"].astype(str).str.strip()
    out["project_name"] = out["project_name"].astype(str).str.strip()
    return out


def compute_period_metrics(tasks_df: pd.DataFrame, proj_df: pd.DataFrame) -> dict:
    """Stable totals for the period (used by Monthly/Quarterly top metrics)."""
    if tasks_df is None or tasks_df.empty:
        tasks_df = pd.DataFrame(columns=["task_type_norm", "role_type_norm", "duration_seconds", "volume", "task_id"])

    prod_primary = tasks_df[(tasks_df["task_type_norm"] == "production") & (tasks_df["role_type_norm"] == "primary")].copy()
    prod_backup = tasks_df[(tasks_df["task_type_norm"] == "production") & (tasks_df["role_type_norm"] == "backup")].copy()
    coverage = tasks_df[(tasks_df["task_type_norm"] == "coverage")].copy()
    production_all = tasks_df[(tasks_df["task_type_norm"] == "production")].copy()

    prod_primary_hours = prod_primary["duration_seconds"].sum() / 3600.0
    prod_backup_hours = prod_backup["duration_seconds"].sum() / 3600.0
    coverage_volume = int(coverage["volume"].sum()) if not coverage.empty else 0
    production_volume = int(production_all["volume"].sum()) if not production_all.empty else 0

    proj_status = pd.DataFrame(columns=["status", "count"])
    if proj_df is not None and not proj_df.empty:
        proj_status = (
            proj_df.groupby("status_norm", as_index=False)["project_name"]
            .count()
            .rename(columns={"status_norm": "status", "project_name": "count"})
            .sort_values(["count", "status"], ascending=[False, True])
        )

    return {
        "prod_primary_hours": prod_primary_hours,
        "prod_backup_hours": prod_backup_hours,
        "coverage_volume": coverage_volume,
        "production_volume": production_volume,
        "proj_status": proj_status,
    }


def kpi_by_task(tasks_df: pd.DataFrame, filter_mask: pd.Series, kind: str) -> pd.DataFrame:
    """
    Build KPI table by task_id for a given mask.
    kind:
      - "production" => hours + volume
      - "coverage"   => volume only
    Returns columns:
      production: task_id, primary_hours or backup_hours (set by caller), production_volume
      coverage: task_id, coverage_volume
    """
    if tasks_df is None or tasks_df.empty:
        return pd.DataFrame()

    df = tasks_df[filter_mask].copy()
    if df.empty:
        return pd.DataFrame()

    if kind == "coverage":
        out = (
            df.groupby("task_id", as_index=False)["volume"]
            .sum()
            .rename(columns={"volume": "Coverage Volume"})
            .sort_values(["Coverage Volume", "task_id"], ascending=[False, True])
        )
        out["Coverage Volume"] = out["Coverage Volume"].astype(int)
        return out

    # production
    df["hours"] = df["duration_seconds"].apply(seconds_to_hours)
    out = (
        df.groupby("task_id", as_index=False)
        .agg(hours=("hours", "sum"), volume=("volume", "sum"))
        .sort_values(["hours", "task_id"], ascending=[False, True])
    )
    out["hours"] = out["hours"].round(2)
    out["volume"] = out["volume"].astype(int)
    out = out.rename(columns={"hours": "Hours", "volume": "Production Volume"})
    return out


def add_deltas(curr_df: pd.DataFrame, prev_df: pd.DataFrame, value_cols: List[str], key_col: str = "task_id") -> pd.DataFrame:
    """
    Merge curr + prev and add Δ columns for each value col.
    Output keeps curr values, adds Δ columns:
      e.g. "Hours" + "Δ Hours", "Production Volume" + "Δ Production Volume"
    """
    if curr_df is None or curr_df.empty:
        return pd.DataFrame()

    curr = curr_df.copy()
    if prev_df is None or prev_df.empty:
        # still include Δ columns as 0
        for c in value_cols:
            curr[f"Δ {c}"] = 0
        return curr

    prev = prev_df.copy()
    merged = curr.merge(prev[[key_col] + value_cols], on=key_col, how="left", suffixes=("", "_prev"))
    for c in value_cols:
        prevc = f"{c}_prev"
        if prevc not in merged.columns:
            merged[f"Δ {c}"] = 0
        else:
            merged[f"Δ {c}"] = merged[c] - merged[prevc].fillna(0)

    # Drop *_prev columns
    drop_cols = [f"{c}_prev" for c in value_cols if f"{c}_prev" in merged.columns]
    merged = merged.drop(columns=drop_cols)
    return merged


# -----------------------------
# UI: Tabs (do not change order/labels)
# -----------------------------
tabs = st.tabs(["Weekly", "Monthly", "Quarterly", "Admin Upload"])

# =============================
# Weekly Tab (UNCHANGED)
# =============================
with tabs[0]:
    st.markdown("## Weekly View")

    members = list_team_members()
    if not members:
        st.info("No data yet. Please upload a combined workbook in Admin Upload.")
    else:
        sel_member = st.selectbox("Team Member", members, index=0)

        weeks = list_weeks_for_member(sel_member)
        if not weeks:
            st.info("No weeks found for this team member yet.")
        else:
            sel_week = st.selectbox("Week Ending", weeks, index=0)

            df = fetch_week_tasks(sel_member, sel_week)
            dfp = fetch_week_projects(sel_member, sel_week)

            # Standardize for comparisons
            if not df.empty:
                df["task_type_norm"] = df["task_type"].str.strip().str.lower()
                df["role_type_norm"] = df["role_type"].str.strip().str.lower()
                df["day"] = df["day"].astype(str).str.strip()
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

            # 2) Task Summary
            st.markdown("### Task Summary")

            if df.empty:
                st.info("No task entries for this week.")
            else:
                df_prod = df[df["task_type_norm"] == "production"].copy()

                if df_prod.empty:
                    st.info("No production entries for this week.")
                else:
                    summary = (
                        df_prod[["task_id", "task_description"]]
                        .drop_duplicates()
                        .sort_values(["task_id"])
                    )
                    st.dataframe(summary, use_container_width=True, hide_index=True)

            # 3) Coverage
            st.markdown("### Coverage")

            if df.empty:
                st.info("No coverage entries for this week.")
            else:
                df_cov = df[df["task_type_norm"] == "coverage"].copy()
                if df_cov.empty:
                    st.info("No coverage entries for this week.")
                else:
                    cov_view = df_cov[["task_id", "day", "volume"]].copy()
                    cov_view["day_sort"] = cov_view["day"].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                    cov_view = cov_view.sort_values(["task_id", "day_sort"]).drop(columns=["day_sort"])
                    st.dataframe(cov_view, use_container_width=True, hide_index=True)

            # 4) Production (Primary)
            st.markdown("### Production (Primary)")

            if df.empty:
                st.info("No production (primary) entries for this week.")
            else:
                df_primary = df[(df["task_type_norm"] == "production") & (df["role_type_norm"] == "primary")].copy()

                if df_primary.empty:
                    st.info("No production (primary) entries for this week.")
                else:
                    days_present = [d for d in DAY_ORDER if d in set(df_primary["day"].astype(str))]
                    day_choice = st.selectbox("Filter by Day (optional)", ["All"] + days_present, index=0)

                    view = df_primary[["task_id", "duration_hhmmss", "volume", "day"]].copy()
                    if day_choice != "All":
                        view = view[view["day"] == day_choice].copy()

                    view["day_sort"] = view["day"].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                    view = view.sort_values(["day_sort", "task_id"]).drop(columns=["day_sort"])
                    view = view.rename(columns={"duration_hhmmss": "duration (hh:mm:ss)"})

                    st.dataframe(view, use_container_width=True, hide_index=True)

            # 5) Production (Backup)
            st.markdown("### Production (Backup)")

            if df.empty:
                st.info("No production (backup) entries for this week.")
            else:
                df_backup = df[(df["task_type_norm"] == "production") & (df["role_type_norm"] == "backup")].copy()

                if df_backup.empty:
                    st.info("No backup tasks this week.")
                else:
                    days_present_b = [d for d in DAY_ORDER if d in set(df_backup["day"].astype(str))]
                    day_choice_b = st.selectbox("Filter Backup by Day (optional)", ["All"] + days_present_b, index=0, key="backup_day_filter")

                    view_b = df_backup[["task_id", "duration_hhmmss", "volume", "day"]].copy()
                    if day_choice_b != "All":
                        view_b = view_b[view_b["day"] == day_choice_b].copy()

                    view_b["day_sort"] = view_b["day"].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                    view_b = view_b.sort_values(["day_sort", "task_id"]).drop(columns=["day_sort"])
                    view_b = view_b.rename(columns={"duration_hhmmss": "duration (hh:mm:ss)"})

                    st.dataframe(view_b, use_container_width=True, hide_index=True)

            # 6) Projects (Weekly)
            st.markdown("### Projects (Weekly)")

            if dfp.empty:
                st.info("No projects found for this owner/week.")
            else:
                proj_cols = ["project_name", "status", "days_active", "notes"]
                dfp_view = dfp[proj_cols].copy()
                st.dataframe(dfp_view, use_container_width=True, hide_index=True)

# =============================
# Monthly Tab
# =============================
with tabs[1]:
    st.subheader("Monthly View")

    members = list_team_members()
    if not members:
        st.info("No data yet. Please upload a combined workbook in Admin Upload.")
    else:
        sel_member_m = st.selectbox("Team Member", members, index=0, key="monthly_member")

        all_tasks = prep_tasks_for_rollups(fetch_all_tasks(sel_member_m))
        all_projects = prep_projects_for_rollups(fetch_all_projects(sel_member_m))

        if all_tasks.empty:
            st.info("No data yet for this team member.")
        else:
            months = sorted(all_tasks["month"].unique().tolist())
            if not months:
                st.info("No month values recognized yet.")
            else:
                sel_month = st.selectbox("Month", months, index=len(months) - 1, key="monthly_month")

                cur_tasks = all_tasks[all_tasks["month"] == sel_month].copy()
                cur_proj = all_projects[all_projects["month"] == sel_month].copy() if not all_projects.empty else pd.DataFrame()

                prev_month = None
                try:
                    cur_period = pd.Period(sel_month, freq="M")
                    prev_month = str(cur_period - 1)
                except Exception:
                    prev_month = None

                prev_tasks = all_tasks[all_tasks["month"] == prev_month].copy() if (prev_month in months) else pd.DataFrame()
                prev_proj = all_projects[all_projects["month"] == prev_month].copy() if (prev_month and (not all_projects.empty) and (prev_month in all_projects["month"].unique())) else pd.DataFrame()

                cur_tot = compute_period_metrics(cur_tasks, cur_proj)
                prev_tot = compute_period_metrics(prev_tasks, prev_proj) if (prev_month and not prev_tasks.empty) else None

                st.markdown("### Monthly Totals & Comparisons")
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    delta = period_delta_str(cur_tot["prod_primary_hours"], prev_tot["prod_primary_hours"]) if prev_tot else ""
                    st.metric("Production (Primary) Hours", fmt_hours(cur_tot["prod_primary_hours"]), delta)

                with c2:
                    delta = period_delta_str(cur_tot["prod_backup_hours"], prev_tot["prod_backup_hours"]) if prev_tot else ""
                    st.metric("Production (Backup) Hours", fmt_hours(cur_tot["prod_backup_hours"]), delta)

                with c3:
                    delta = period_delta_str(cur_tot["coverage_volume"], prev_tot["coverage_volume"], is_int=True) if prev_tot else ""
                    st.metric("Coverage Volume", fmt_int(cur_tot["coverage_volume"]), delta)

                with c4:
                    delta = period_delta_str(cur_tot["production_volume"], prev_tot["production_volume"], is_int=True) if prev_tot else ""
                    st.metric("Production Volume (Total)", fmt_int(cur_tot["production_volume"]), delta)

                # ---------- KPI by Task: Production (Primary) ----------
                st.markdown("### Production Tasks (Monthly) — KPIs by Task (Primary)")

                mask_primary = (cur_tasks["task_type_norm"] == "production") & (cur_tasks["role_type_norm"] == "primary")
                mask_primary_prev = (prev_tasks["task_type_norm"] == "production") & (prev_tasks["role_type_norm"] == "primary") if not prev_tasks.empty else None

                cur_primary = kpi_by_task(cur_tasks, mask_primary, kind="production")
                prev_primary = kpi_by_task(prev_tasks, mask_primary_prev, kind="production") if (mask_primary_prev is not None) else pd.DataFrame()

                if cur_primary.empty:
                    st.info("No primary production tasks for this month.")
                else:
                    # add deltas and remove descriptions (none included here anyway)
                    cur_primary = add_deltas(cur_primary, prev_primary, value_cols=["Hours", "Production Volume"])
                    # rename for clarity
                    cur_primary = cur_primary.rename(columns={"Hours": "Primary Hours", "Δ Hours": "Δ Primary Hours"})
                    # style only delta columns
                    styled = style_delta_df(cur_primary, delta_cols=["Δ Primary Hours", "Δ Production Volume"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                # ---------- KPI by Task: Production (Backup) ----------
                st.markdown("### Production Tasks (Monthly) — KPIs by Task (Backup)")

                mask_backup = (cur_tasks["task_type_norm"] == "production") & (cur_tasks["role_type_norm"] == "backup")
                mask_backup_prev = (prev_tasks["task_type_norm"] == "production") & (prev_tasks["role_type_norm"] == "backup") if not prev_tasks.empty else None

                cur_backup = kpi_by_task(cur_tasks, mask_backup, kind="production")
                prev_backup = kpi_by_task(prev_tasks, mask_backup_prev, kind="production") if (mask_backup_prev is not None) else pd.DataFrame()

                if cur_backup.empty:
                    st.info("No backup tasks for this month.")
                else:
                    cur_backup = add_deltas(cur_backup, prev_backup, value_cols=["Hours", "Production Volume"])
                    cur_backup = cur_backup.rename(columns={"Hours": "Backup Hours", "Δ Hours": "Δ Backup Hours"})
                    styled_b = style_delta_df(cur_backup, delta_cols=["Δ Backup Hours", "Δ Production Volume"])
                    st.dataframe(styled_b, use_container_width=True, hide_index=True)

                # ---------- KPI by Task: Coverage ----------
                st.markdown("### Coverage Tasks (Monthly) — KPIs by Task")

                mask_cov = (cur_tasks["task_type_norm"] == "coverage")
                mask_cov_prev = (prev_tasks["task_type_norm"] == "coverage") if not prev_tasks.empty else None

                cur_cov = kpi_by_task(cur_tasks, mask_cov, kind="coverage")
                prev_cov = kpi_by_task(prev_tasks, mask_cov_prev, kind="coverage") if (mask_cov_prev is not None) else pd.DataFrame()

                if cur_cov.empty:
                    st.info("No coverage tasks for this month.")
                else:
                    cur_cov = add_deltas(cur_cov, prev_cov, value_cols=["Coverage Volume"])
                    styled_c = style_delta_df(cur_cov, delta_cols=["Δ Coverage Volume"])
                    st.dataframe(styled_c, use_container_width=True, hide_index=True)

                # Projects by status (unchanged)
                st.markdown("### Projects by Status (This Month)")
                if cur_tot["proj_status"].empty:
                    st.info("No projects found for this month.")
                else:
                    st.dataframe(cur_tot["proj_status"], use_container_width=True, hide_index=True)

# =============================
# Quarterly Tab
# =============================
with tabs[2]:
    st.subheader("Quarterly View")

    members = list_team_members()
    if not members:
        st.info("No data yet. Please upload a combined workbook in Admin Upload.")
    else:
        sel_member_q = st.selectbox("Team Member", members, index=0, key="quarterly_member")

        all_tasks = prep_tasks_for_rollups(fetch_all_tasks(sel_member_q))
        all_projects = prep_projects_for_rollups(fetch_all_projects(sel_member_q))

        if all_tasks.empty:
            st.info("No data yet for this team member.")
        else:
            quarters = sorted(all_tasks["quarter"].unique().tolist())
            if not quarters:
                st.info("No quarter values recognized yet.")
            else:
                sel_quarter = st.selectbox("Quarter", quarters, index=len(quarters) - 1, key="quarterly_quarter")

                cur_tasks = all_tasks[all_tasks["quarter"] == sel_quarter].copy()
                cur_proj = all_projects[all_projects["quarter"] == sel_quarter].copy() if not all_projects.empty else pd.DataFrame()

                prev_quarter = None
                try:
                    cur_period = pd.Period(sel_quarter, freq="Q")
                    prev_quarter = str(cur_period - 1)
                except Exception:
                    prev_quarter = None

                prev_tasks = all_tasks[all_tasks["quarter"] == prev_quarter].copy() if (prev_quarter in quarters) else pd.DataFrame()
                prev_proj = all_projects[all_projects["quarter"] == prev_quarter].copy() if (prev_quarter and (not all_projects.empty) and (prev_quarter in all_projects["quarter"].unique())) else pd.DataFrame()

                cur_tot = compute_period_metrics(cur_tasks, cur_proj)
                prev_tot = compute_period_metrics(prev_tasks, prev_proj) if (prev_quarter and not prev_tasks.empty) else None

                st.markdown("### Quarterly Totals & Comparisons")
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    delta = period_delta_str(cur_tot["prod_primary_hours"], prev_tot["prod_primary_hours"]) if prev_tot else ""
                    st.metric("Production (Primary) Hours", fmt_hours(cur_tot["prod_primary_hours"]), delta)

                with c2:
                    delta = period_delta_str(cur_tot["prod_backup_hours"], prev_tot["prod_backup_hours"]) if prev_tot else ""
                    st.metric("Production (Backup) Hours", fmt_hours(cur_tot["prod_backup_hours"]), delta)

                with c3:
                    delta = period_delta_str(cur_tot["coverage_volume"], prev_tot["coverage_volume"], is_int=True) if prev_tot else ""
                    st.metric("Coverage Volume", fmt_int(cur_tot["coverage_volume"]), delta)

                with c4:
                    delta = period_delta_str(cur_tot["production_volume"], prev_tot["production_volume"], is_int=True) if prev_tot else ""
                    st.metric("Production Volume (Total)", fmt_int(cur_tot["production_volume"]), delta)

                # KPI by task — Production (Primary)
                st.markdown("### Production Tasks (Quarterly) — KPIs by Task (Primary)")

                mask_primary = (cur_tasks["task_type_norm"] == "production") & (cur_tasks["role_type_norm"] == "primary")
                mask_primary_prev = (prev_tasks["task_type_norm"] == "production") & (prev_tasks["role_type_norm"] == "primary") if not prev_tasks.empty else None

                cur_primary = kpi_by_task(cur_tasks, mask_primary, kind="production")
                prev_primary = kpi_by_task(prev_tasks, mask_primary_prev, kind="production") if (mask_primary_prev is not None) else pd.DataFrame()

                if cur_primary.empty:
                    st.info("No primary production tasks for this quarter.")
                else:
                    cur_primary = add_deltas(cur_primary, prev_primary, value_cols=["Hours", "Production Volume"])
                    cur_primary = cur_primary.rename(columns={"Hours": "Primary Hours", "Δ Hours": "Δ Primary Hours"})
                    styled = style_delta_df(cur_primary, delta_cols=["Δ Primary Hours", "Δ Production Volume"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                # KPI by task — Production (Backup)
                st.markdown("### Production Tasks (Quarterly) — KPIs by Task (Backup)")

                mask_backup = (cur_tasks["task_type_norm"] == "production") & (cur_tasks["role_type_norm"] == "backup")
                mask_backup_prev = (prev_tasks["task_type_norm"] == "production") & (prev_tasks["role_type_norm"] == "backup") if not prev_tasks.empty else None

                cur_backup = kpi_by_task(cur_tasks, mask_backup, kind="production")
                prev_backup = kpi_by_task(prev_tasks, mask_backup_prev, kind="production") if (mask_backup_prev is not None) else pd.DataFrame()

                if cur_backup.empty:
                    st.info("No backup tasks for this quarter.")
                else:
                    cur_backup = add_deltas(cur_backup, prev_backup, value_cols=["Hours", "Production Volume"])
                    cur_backup = cur_backup.rename(columns={"Hours": "Backup Hours", "Δ Hours": "Δ Backup Hours"})
                    styled_b = style_delta_df(cur_backup, delta_cols=["Δ Backup Hours", "Δ Production Volume"])
                    st.dataframe(styled_b, use_container_width=True, hide_index=True)

                # KPI by task — Coverage
                st.markdown("### Coverage Tasks (Quarterly) — KPIs by Task")

                mask_cov = (cur_tasks["task_type_norm"] == "coverage")
                mask_cov_prev = (prev_tasks["task_type_norm"] == "coverage") if not prev_tasks.empty else None

                cur_cov = kpi_by_task(cur_tasks, mask_cov, kind="coverage")
                prev_cov = kpi_by_task(prev_tasks, mask_cov_prev, kind="coverage") if (mask_cov_prev is not None) else pd.DataFrame()

                if cur_cov.empty:
                    st.info("No coverage tasks for this quarter.")
                else:
                    cur_cov = add_deltas(cur_cov, prev_cov, value_cols=["Coverage Volume"])
                    styled_c = style_delta_df(cur_cov, delta_cols=["Δ Coverage Volume"])
                    st.dataframe(styled_c, use_container_width=True, hide_index=True)

                st.markdown("### Projects by Status (This Quarter)")
                if cur_tot["proj_status"].empty:
                    st.info("No projects found for this quarter.")
                else:
                    st.dataframe(cur_tot["proj_status"], use_container_width=True, hide_index=True)

# =============================
# Admin Upload Tab
# =============================
with tabs[3]:
    st.subheader("Admin Upload (Weekly Combined Workbook)")
    st.caption("Upload the combined weekly workbook. This will overwrite that week in the DB (safe re-upload).")

    uploaded = st.file_uploader(
        "Upload Combined_TaskTracker_YYYY-MM-DD.xlsx",
        type=["xlsx"],
        accept_multiple_files=False,
    )
    overwrite = st.checkbox("Overwrite week if it already exists", value=True)

    if uploaded is not None:
        try:
            data = BytesIO(uploaded.read())
            xls = pd.ExcelFile(data)

            if "Tasks" not in xls.sheet_names or "Projects" not in xls.sheet_names:
                st.error("Workbook must contain two sheets named exactly: 'Tasks' and 'Projects'.")
                st.stop()

            tasks_raw = pd.read_excel(xls, sheet_name="Tasks")
            projects_raw = pd.read_excel(xls, sheet_name="Projects")

            tasks_df = normalize_tasks_df(tasks_raw)
            projects_df = normalize_projects_df(projects_raw)

            week_vals = sorted(set([w for w in tasks_df["week_ending"].unique() if str(w).strip() != ""]))
            if not week_vals:
                st.error("Could not find 'Week Ending' values in Tasks sheet.")
                st.stop()

            week_ending = week_vals[0]

            st.write(f"Detected week ending: **{week_ending}**")
            st.write("Preview (Tasks):")
            st.dataframe(tasks_df.head(10), use_container_width=True, hide_index=True)
            st.write("Preview (Projects):")
            st.dataframe(projects_df.head(10), use_container_width=True, hide_index=True)

            if st.button("Upload to Database"):
                if overwrite:
                    delete_week_data(week_ending)

                insert_tasks(tasks_df)
                insert_projects(projects_df)

                st.success("Upload complete. You can switch to Weekly tab and verify.")
        except Exception as e:
            st.error(f"Upload failed: {e}")
