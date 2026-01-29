import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# -----------------------------
# Page / Layout (DO NOT TOUCH)
# -----------------------------
st.set_page_config(page_title="MAO Workflow Tracker Dashboard", layout="wide")
st.title("MAO Workflow Tracker Dashboard")
st.caption("LPL Financial – Operations")

# -----------------------------
# DB CONFIG
# -----------------------------
def get_engine():
    """
    Uses:
      - Render Postgres via DATABASE_URL
      - falls back to local sqlite dashboard.db
    """
    db_url = os.environ.get("DATABASE_URL", "").strip()

    if not db_url:
        return create_engine("sqlite:///dashboard.db", future=True)

    # Render commonly provides postgres://; SQLAlchemy wants postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    # Force SSL for Render
    connect_args = {"sslmode": "require"}
    return create_engine(db_url, future=True, connect_args=connect_args)

ENGINE = get_engine()

# -----------------------------
# HELPERS
# -----------------------------
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def clean_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    # Normalize weird whitespace
    s = re.sub(r"\s+", " ", s)
    return s

def parse_volume_num(vol) -> Optional[float]:
    """
    Accepts:
      - 222
      - "222 emails"
      - "16 accts"
      - "0 accounts"
      - "" / None -> None
    Returns numeric (float) or None if no number present.
    """
    if vol is None:
        return None
    s = str(vol).strip()
    if not s:
        return None

    # Grab first number (int or decimal)
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def ensure_day_sort(df: pd.DataFrame, day_col: str = "day") -> pd.DataFrame:
    if day_col not in df.columns:
        return df
    df = df.copy()
    df[day_col] = df[day_col].astype(str)
    df["day_sort"] = df[day_col].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
    df = df.sort_values(["day_sort"] + [c for c in df.columns if c not in ("day_sort",)])
    df = df.drop(columns=["day_sort"])
    return df

# -----------------------------
# NORMALIZATION (Excel -> canonical)
# -----------------------------
def normalize_tasks_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected from combined Tasks sheet:
      Task ID, Team Member, Task Description, Task Type, Role Type,
      Duration Seconds, Duration Minutes, Duration Hours, Volume, Day, Week Ending
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "task_id","team_member","task_description","task_type","role_type",
            "duration_seconds","volume_text","volume_num","day","week_ending"
        ])

    df = df.copy()
    df.columns = [clean_str(c) for c in df.columns]

    colmap = {
        "Task ID": "task_id",
        "Team Member": "team_member",
        "Task Description": "task_description",
        "Task Type": "task_type",
        "Role Type": "role_type",
        "Duration Seconds": "duration_seconds",
        "Volume": "volume_text",
        "Day": "day",
        "Week Ending": "week_ending",
    }

    out = pd.DataFrame()
    for src, dst in colmap.items():
        if src in df.columns:
            out[dst] = df[src]
        else:
            out[dst] = None

    # Clean strings
    for c in ["task_id","team_member","task_description","task_type","role_type","day","week_ending"]:
        out[c] = out[c].apply(clean_str)

    # Duration seconds numeric (keep missing as 0)
    out["duration_seconds"] = pd.to_numeric(out["duration_seconds"], errors="coerce").fillna(0).astype(int)

    # Volume: keep what user typed + extract numeric for metrics
    out["volume_text"] = out["volume_text"].apply(lambda x: clean_str(x))
    out["volume_num"] = out["volume_text"].apply(parse_volume_num)

    # Drop rows without task_id
    out = out.dropna(subset=["task_id"])
    out = out[out["task_id"].astype(str).str.strip() != ""]

    return out

def normalize_projects_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected from combined Projects sheet:
      Project Name, Owner, Start Date, End Date, Status, Days Active, Notes, Week Ending
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "project_name","owner","start_date","end_date","status","days_active","notes","week_ending"
        ])

    df = df.copy()
    df.columns = [clean_str(c) for c in df.columns]

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
        out[dst] = df[src] if src in df.columns else None

    for c in ["project_name","owner","status","notes","week_ending"]:
        out[c] = out[c].apply(clean_str)

    # Dates can stay as text; Days Active should be numeric
    out["start_date"] = out["start_date"].apply(lambda x: clean_str(x))
    out["end_date"] = out["end_date"].apply(lambda x: clean_str(x))
    out["days_active"] = pd.to_numeric(out["days_active"], errors="coerce").fillna(0).astype(int)

    out = out.dropna(subset=["project_name"])
    out = out[out["project_name"].astype(str).str.strip() != ""]

    return out

# -----------------------------
# DB INIT
# -----------------------------
def init_db():
    """
    Creates tables if they don't exist.
    Volume stored as:
      - volume_text TEXT (whatever user typed)
      - volume_num DOUBLE PRECISION (parsed number)
    """
    with ENGINE.begin() as conn:
        dialect = ENGINE.dialect.name.lower()

        if dialect == "sqlite":
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    team_member TEXT,
                    task_description TEXT,
                    task_type TEXT,
                    role_type TEXT,
                    duration_seconds INTEGER,
                    volume_text TEXT,
                    volume_num REAL,
                    day TEXT,
                    week_ending TEXT,
                    uploaded_at TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id BIGSERIAL PRIMARY KEY,
                    task_id TEXT,
                    team_member TEXT,
                    task_description TEXT,
                    task_type TEXT,
                    role_type TEXT,
                    duration_seconds INTEGER,
                    volume_text TEXT,
                    volume_num DOUBLE PRECISION,
                    day TEXT,
                    week_ending TEXT,
                    uploaded_at TIMESTAMP DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS projects (
                    id BIGSERIAL PRIMARY KEY,
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
            """))

init_db()

# -----------------------------
# QUERY HELPERS
# -----------------------------
def list_members() -> List[str]:
    with ENGINE.begin() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT team_member
            FROM tasks
            WHERE team_member IS NOT NULL AND team_member <> ''
            ORDER BY team_member
        """)).fetchall()
    return [r[0] for r in rows]

def list_weeks_for_member(member: str) -> List[str]:
    with ENGINE.begin() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT week_ending
            FROM tasks
            WHERE team_member = :m
              AND week_ending IS NOT NULL AND week_ending <> ''
            ORDER BY week_ending DESC
        """), {"m": member}).fetchall()
    return [r[0] for r in rows]

def fetch_week_tasks(member: str, week_ending: str) -> pd.DataFrame:
    with ENGINE.begin() as conn:
        rows = conn.execute(text("""
            SELECT task_id, team_member, task_description, task_type, role_type,
                   duration_seconds, volume_text, volume_num, day, week_ending
            FROM tasks
            WHERE team_member = :m AND week_ending = :w
        """), {"m": member, "w": week_ending}).fetchall()

    return pd.DataFrame(rows, columns=[
        "task_id","team_member","task_description","task_type","role_type",
        "duration_seconds","volume_text","volume_num","day","week_ending"
    ])

def fetch_week_projects(owner: str, week_ending: str) -> pd.DataFrame:
    with ENGINE.begin() as conn:
        rows = conn.execute(text("""
            SELECT project_name, owner, start_date, end_date, status, days_active, notes, week_ending
            FROM projects
            WHERE owner = :o AND week_ending = :w
        """), {"o": owner, "w": week_ending}).fetchall()

    return pd.DataFrame(rows, columns=[
        "project_name","owner","start_date","end_date","status","days_active","notes","week_ending"
    ])

# -----------------------------
# UI: Tabs (KEEP ORDER)
# -----------------------------
tabs = st.tabs(["Weekly", "Monthly", "Quarterly", "Admin Upload"])

# -----------------------------
# WEEKLY TAB
# -----------------------------
with tabs[0]:
    members = list_members()
    if not members:
        st.info("No data yet. Upload a combined workbook in Admin Upload.")
    else:
        sel_member = st.selectbox("Team Member", members, index=0)
        weeks = list_weeks_for_member(sel_member)
        if not weeks:
            st.info("No weeks found for this team member yet.")
        else:
            sel_week = st.selectbox("Week Ending", weeks, index=0)

            df = fetch_week_tasks(sel_member, sel_week)

            # Split:
            # - Coverage: task_type == Coverage
            # - Production: task_type == Production
            df["task_type"] = df["task_type"].astype(str)
            df["role_type"] = df["role_type"].astype(str)

            df_cov = df[df["task_type"].str.lower() == "coverage"].copy()
            df_prod = df[df["task_type"].str.lower() == "production"].copy()

            # =========================================================
            # 2) Task Summary (Unique Tasks) -> "Task Summary"
            #    ONLY Production tasks, unique per week
            # =========================================================
            st.markdown("### Task Summary")
            summary = (
                df_prod[["task_id", "task_description"]]
                .dropna(subset=["task_id"])
                .drop_duplicates()
                .sort_values(["task_id"])
            )
            # Leave description blank if missing
            summary["task_description"] = summary["task_description"].fillna("").astype(str)

            st.dataframe(summary, use_container_width=True, hide_index=True)

            # =========================================================
            # 3) Coverage (add Day column, keep layout)
            #    Columns: task_id, day, volume
            #    Volume: show parsed numeric so monthly/quarterly work cleanly
            # =========================================================
            st.markdown("### Coverage")
            if df_cov.empty:
                st.info("No coverage entries for this week.")
            else:
                cov_view = df_cov[["task_id", "day", "volume_num"]].copy()
                cov_view["task_id"] = cov_view["task_id"].fillna("").astype(str)
                cov_view["day"] = cov_view["day"].fillna("").astype(str)
                cov_view["volume"] = cov_view["volume_num"].fillna(0).astype(int)
                cov_view = cov_view.drop(columns=["volume_num"])

                cov_view = ensure_day_sort(cov_view, "day")
                st.dataframe(cov_view, use_container_width=True, hide_index=True)

            # =========================================================
            # 4) Raw data -> rename to Production (Primary) + Day filter
            #    Only Production + Primary
            # =========================================================
            st.markdown("### Production (Primary)")

            df_primary = df_prod[df_prod["role_type"].str.lower() == "primary"].copy()

            # Day filter
            day_options = ["All"] + [d for d in DAY_ORDER if d in set(df_primary["day"].astype(str))]
            sel_day_primary = st.selectbox("Filter by day (Primary)", day_options, index=0)

            if sel_day_primary != "All":
                df_primary = df_primary[df_primary["day"].astype(str) == sel_day_primary].copy()

            if df_primary.empty:
                st.info("No primary production tasks found for this selection.")
            else:
                raw = df_primary[["task_id", "duration_seconds", "volume_num", "day"]].copy()

                # Duration to hh:mm:ss
                raw["duration (hh:mm:ss)"] = raw["duration_seconds"].apply(
                    lambda s: f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"
                )
                raw["volume"] = raw["volume_num"].fillna(0).astype(int)

                raw = raw.drop(columns=["duration_seconds", "volume_num"])
                raw["day"] = raw["day"].astype(str)

                raw = ensure_day_sort(raw, "day")
                st.dataframe(raw, use_container_width=True, hide_index=True)

            # =========================================================
            # 5) Production (Backup) section (always show)
            #    Only Production + Backup
            # =========================================================
            st.markdown("### Production (Backup)")

            df_backup = df_prod[df_prod["role_type"].str.lower() == "backup"].copy()

            # Optional day filter (same behavior)
            day_options_b = ["All"] + [d for d in DAY_ORDER if d in set(df_backup["day"].astype(str))]
            sel_day_backup = st.selectbox("Filter by day (Backup)", day_options_b, index=0)

            if sel_day_backup != "All":
                df_backup = df_backup[df_backup["day"].astype(str) == sel_day_backup].copy()

            if df_backup.empty:
                st.info("No backup tasks this week.")
            else:
                raw_b = df_backup[["task_id", "duration_seconds", "volume_num", "day"]].copy()

                raw_b["duration (hh:mm:ss)"] = raw_b["duration_seconds"].apply(
                    lambda s: f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"
                )
                raw_b["volume"] = raw_b["volume_num"].fillna(0).astype(int)

                raw_b = raw_b.drop(columns=["duration_seconds", "volume_num"])
                raw_b["day"] = raw_b["day"].astype(str)

                raw_b = ensure_day_sort(raw_b, "day")
                st.dataframe(raw_b, use_container_width=True, hide_index=True)

            # =========================================================
            # 6) Projects (Weekly) (unchanged layout)
            # =========================================================
            st.markdown("### Projects (Weekly)")
            dfp = fetch_week_projects(sel_member, sel_week)
            if dfp.empty:
                st.info("No projects found for this owner/week.")
            else:
                proj_cols = ["project_name", "status", "days_active", "notes"]
                dfp_view = dfp[proj_cols].copy()
                st.dataframe(dfp_view, use_container_width=True, hide_index=True)

# -----------------------------
# MONTHLY TAB (placeholder - keep layout)
# -----------------------------
with tabs[1]:
    st.subheader("Monthly View")
    st.info("Next: month-to-month comparisons (hours, volume, top tasks, coverage volume, projects by status).")

# -----------------------------
# QUARTERLY TAB (placeholder - keep layout)
# -----------------------------
with tabs[2]:
    st.subheader("Quarterly View")
    st.info("Next: quarter-to-quarter comparisons (same metrics).")

# -----------------------------
# ADMIN UPLOAD TAB
# -----------------------------
with tabs[3]:
    st.subheader("Admin Upload (Weekly Combined Workbook)")
    st.write("Upload the combined weekly workbook. This will overwrite that week in the DB (safe re-upload).")

    uploaded = st.file_uploader("Upload Combined_TaskTracker_YYYY-MM-DD.xlsx", type=["xlsx"])

    overwrite = st.checkbox("Overwrite week if it already exists", value=True)

    if uploaded is not None:
        try:
            xls = pd.ExcelFile(uploaded)
            if "Tasks" not in xls.sheet_names or "Projects" not in xls.sheet_names:
                st.error("Workbook must contain sheets named 'Tasks' and 'Projects'.")
            else:
                df_tasks_raw = pd.read_excel(xls, "Tasks")
                df_projects_raw = pd.read_excel(xls, "Projects")

                df_tasks = normalize_tasks_df(df_tasks_raw)
                df_projects = normalize_projects_df(df_projects_raw)

                # Determine week_ending from the file content
                week_vals = sorted(set(df_tasks["week_ending"].dropna().astype(str)))
                week_ending = week_vals[0] if week_vals else ""

                if not week_ending:
                    st.error("Could not detect Week Ending from Tasks sheet. Make sure 'Week Ending' column is populated.")
                else:
                    with ENGINE.begin() as conn:
                        if overwrite:
                            conn.execute(text("DELETE FROM tasks WHERE week_ending = :w"), {"w": week_ending})
                            conn.execute(text("DELETE FROM projects WHERE week_ending = :w"), {"w": week_ending})

                        # Insert tasks
                        now_txt = datetime.now().isoformat(timespec="seconds")
                        for _, r in df_tasks.iterrows():
                            conn.execute(text("""
                                INSERT INTO tasks (
                                    task_id, team_member, task_description, task_type, role_type,
                                    duration_seconds, volume_text, volume_num, day, week_ending, uploaded_at
                                ) VALUES (
                                    :task_id, :team_member, :task_description, :task_type, :role_type,
                                    :duration_seconds, :volume_text, :volume_num, :day, :week_ending, :uploaded_at
                                )
                            """), {
                                "task_id": r["task_id"],
                                "team_member": r["team_member"],
                                "task_description": r["task_description"],
                                "task_type": r["task_type"],
                                "role_type": r["role_type"],
                                "duration_seconds": int(r["duration_seconds"]),
                                "volume_text": r["volume_text"],
                                "volume_num": None if pd.isna(r["volume_num"]) else float(r["volume_num"]),
                                "day": r["day"],
                                "week_ending": r["week_ending"],
                                "uploaded_at": now_txt,
                            })

                        # Insert projects
                        for _, r in df_projects.iterrows():
                            conn.execute(text("""
                                INSERT INTO projects (
                                    project_name, owner, start_date, end_date, status,
                                    days_active, notes, week_ending, uploaded_at
                                ) VALUES (
                                    :project_name, :owner, :start_date, :end_date, :status,
                                    :days_active, :notes, :week_ending, :uploaded_at
                                )
                            """), {
                                "project_name": r["project_name"],
                                "owner": r["owner"],
                                "start_date": r["start_date"],
                                "end_date": r["end_date"],
                                "status": r["status"],
                                "days_active": int(r["days_active"]),
                                "notes": r["notes"],
                                "week_ending": r["week_ending"],
                                "uploaded_at": now_txt,
                            })

                    st.success(f"Upload complete. Week Ending = {week_ending}.")
                    st.info("Go back to Weekly tab and refresh the page to see the updates.")

        except Exception as e:
            st.error(f"Upload failed: {e}")
