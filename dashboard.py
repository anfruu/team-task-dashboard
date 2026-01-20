import os
import io
import re
from datetime import datetime, date
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# -----------------------------
# Page config / header
# -----------------------------
st.set_page_config(page_title="MAO Workflow Tracker Dashboard", layout="wide")

st.markdown(
    """
    <div style="padding-top:10px; padding-bottom:6px;">
        <h1 style="margin-bottom:0;">MAO Workflow Tracker Dashboard</h1>
        <div style="color:#6b7280; margin-top:2px;">LPL Financial – Operations</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# DB (Postgres if DATABASE_URL set; else local SQLite)
# -----------------------------
def get_engine():
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url:
        # Render commonly uses postgres://; SQLAlchemy prefers postgresql://
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url, pool_pre_ping=True)
    # Fallback local DB (works locally and on Render without paid DB)
    return create_engine("sqlite:///mao_workflow.db", pool_pre_ping=True)

ENGINE = get_engine()

# -----------------------------
# Helpers: normalize / parse
# -----------------------------
def _clean_str(x):
    if pd.isna(x):
        return None
    return str(x).strip()

def parse_week_ending_from_filename(name: str) -> Optional[str]:
    """
    Tries to find YYYY-MM-DD in filename.
    """
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    return m.group(1) if m else None

def duration_to_seconds(val) -> Optional[int]:
    """
    Accepts:
      - numeric seconds
      - Excel time (fraction of a day)
      - strings like '0:00:01', '01:12:30', '1:02:03', etc.
      - datetime/timedelta
    Returns integer seconds or None.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return None

    # Pandas Timedelta
    if isinstance(val, pd.Timedelta):
        return int(val.total_seconds())

    # Python timedelta
    try:
        import datetime as _dt
        if isinstance(val, _dt.timedelta):
            return int(val.total_seconds())
    except Exception:
        pass

    # Excel sometimes stores duration as fraction of day
    if isinstance(val, (int, float)) and not pd.isna(val):
        # Heuristic: if it's small (<= 2), treat as fraction-of-day; otherwise seconds
        # 0.5 == 12 hours
        if 0 <= float(val) <= 2:
            return int(round(float(val) * 86400))
        return int(round(float(val)))

    s = str(val).strip()

    # If it's like "0 days 00:01:02"
    if "day" in s and ":" in s:
        try:
            td = pd.to_timedelta(s)
            return int(td.total_seconds())
        except Exception:
            pass

    # If it's HH:MM:SS or MM:SS
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [int(float(p)) for p in parts]
        except Exception:
            # try timedelta parse
            try:
                td = pd.to_timedelta(s)
                return int(td.total_seconds())
            except Exception:
                return None

        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh * 3600 + mm * 60 + ss)
        if len(parts) == 2:
            mm, ss = parts
            return int(mm * 60 + ss)

    # Pure numeric string
    try:
        return int(round(float(s)))
    except Exception:
        return None

def seconds_to_hhmmss(seconds: Optional[int]) -> str:
    if seconds is None or pd.isna(seconds):
        return ""
    try:
        seconds = int(seconds)
    except Exception:
        return ""
    if seconds < 0:
        # clamp; negative durations should not happen in your tracker
        seconds = 0
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def normalize_tasks_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected from combined Tasks sheet (your current combined has):
      Task ID, Team Member, Task Description, Task Type, Role Type,
      Duration Seconds, Duration Minutes, Duration Hours, Volume, Day, Week Ending

    We normalize to canonical columns:
      task_id, team_member, task_description, task_type, role_type,
      duration_seconds, volume, day, week_ending
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "task_id","team_member","task_description","task_type","role_type",
            "duration_seconds","volume","day","week_ending"
        ])

    # Standardize col names
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Map potential column variants
    colmap = {
        "Task ID": "task_id",
        "Team Member": "team_member",
        "Task Description": "task_description",
        "Task Type": "task_type",
        "Role Type": "role_type",
        "Duration Seconds": "duration_seconds",
        "Duration": "duration_raw",          # legacy
        "Volume": "volume",
        "Day": "day",
        "Week Ending": "week_ending",
    }

    # Build a new DF with what we have
    out = pd.DataFrame()
    for src, dst in colmap.items():
        if src in df.columns:
            out[dst] = df[src]

    # If we don't have Duration Seconds, try Duration
    if "duration_seconds" not in out.columns and "duration_raw" in out.columns:
        out["duration_seconds"] = out["duration_raw"].apply(duration_to_seconds)

    if "duration_seconds" in out.columns:
        out["duration_seconds"] = out["duration_seconds"].apply(duration_to_seconds)

    # Clean strings
    for c in ["task_id","team_member","task_description","task_type","role_type","day","week_ending"]:
        if c in out.columns:
            out[c] = out[c].apply(_clean_str)

    # Volume numeric
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)

    # Drop blanks
    if "task_id" in out.columns:
        out = out.dropna(subset=["task_id"])

    # Ensure required columns exist
    for c in ["task_id","team_member","task_description","task_type","role_type",
              "duration_seconds","volume","day","week_ending"]:
        if c not in out.columns:
            out[c] = None

    # Duration default 0 if missing
    out["duration_seconds"] = pd.to_numeric(out["duration_seconds"], errors="coerce").fillna(0).astype(int)

    return out

def normalize_projects_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    From combined Projects sheet (your screenshot shows):
      Project Name, Owner, Start Date, End Date, Status, Days Active, Notes, Week Ending

    Canonical:
      project_name, owner, start_date, end_date, status, days_active, notes, week_ending
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "project_name","owner","start_date","end_date","status","days_active","notes","week_ending"
        ])

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

    for c in ["project_name","owner","status","notes","week_ending"]:
        if c in out.columns:
            out[c] = out[c].apply(_clean_str)

    # Parse dates (keep as ISO strings)
    for dc in ["start_date","end_date"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce").dt.date.astype("string")

    if "days_active" in out.columns:
        out["days_active"] = pd.to_numeric(out["days_active"], errors="coerce").fillna(0).astype(int)

    # Drop blanks
    if "project_name" in out.columns:
        out = out.dropna(subset=["project_name"])

    for c in ["project_name","owner","start_date","end_date","status","days_active","notes","week_ending"]:
        if c not in out.columns:
            out[c] = None

    return out

# -----------------------------
# DB schema + upsert
# -----------------------------
def init_db():
    # Works for sqlite + postgres
    with ENGINE.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
        """)) if ENGINE.dialect.name == "sqlite" else None

        # Create tasks table (Postgres uses SERIAL/IDENTITY; easiest is generic)
        if ENGINE.dialect.name == "sqlite":
            conn.execute(text("""
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
            """))
            conn.execute(text("""
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
            """))
        else:
            conn.execute(text("""
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
            """))
            conn.execute(text("""
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
            """))

            # Optional indexes for speed
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tasks_member_week ON tasks (team_member, week_ending)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_projects_owner_week ON projects (owner, week_ending)"))

init_db()

def delete_week(conn, week_ending: str):
    conn.execute(text("DELETE FROM tasks WHERE week_ending = :w"), {"w": week_ending})
    conn.execute(text("DELETE FROM projects WHERE week_ending = :w"), {"w": week_ending})

def insert_tasks(conn, df_tasks: pd.DataFrame):
    if df_tasks.empty:
        return
    now = datetime.now().isoformat(timespec="seconds")
    df = df_tasks.copy()
    df["uploaded_at"] = now
    df.to_sql("tasks", conn, if_exists="append", index=False)

def insert_projects(conn, df_projects: pd.DataFrame):
    if df_projects.empty:
        return
    now = datetime.now().isoformat(timespec="seconds")
    df = df_projects.copy()
    df["uploaded_at"] = now
    df.to_sql("projects", conn, if_exists="append", index=False)

def list_team_members() -> list:
    with ENGINE.begin() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT team_member
            FROM tasks
            WHERE team_member IS NOT NULL AND team_member <> ''
            ORDER BY team_member
        """)).fetchall()
    return [r[0] for r in rows]

def list_weeks_for_member(member: str) -> list:
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
                   duration_seconds, volume, day, week_ending
            FROM tasks
            WHERE team_member = :m AND week_ending = :w
        """), {"m": member, "w": week_ending}).fetchall()
    return pd.DataFrame(rows, columns=[
        "task_id","team_member","task_description","task_type","role_type",
        "duration_seconds","volume","day","week_ending"
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
# UI: Tabs
# -----------------------------
tabs = st.tabs(["Weekly", "Monthly", "Quarterly", "Admin Upload"])

# =============================
# Admin Upload
# =============================
with tabs[3]:
    st.subheader("Admin Upload (Weekly Combined Workbook)")
    st.caption("Upload the combined weekly workbook. This will overwrite that week in the DB (safe re-upload).")

    uploaded = st.file_uploader("Upload Combined_TaskTracker_YYYY-MM-DD.xlsx", type=["xlsx"])

    colA, colB = st.columns([1, 2])
    with colA:
        overwrite_week = st.checkbox("Overwrite week if it already exists", value=True)
    with colB:
        st.info("Tip: If you had to run combine in 2 parts, just upload again when you get the remaining files. "
                "Overwrite ON makes the DB match the final combined workbook.")

    if uploaded:
        # Read excel
        file_bytes = uploaded.read()
        bio = io.BytesIO(file_bytes)

        try:
            xl = pd.ExcelFile(bio, engine="openpyxl")
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")
            st.stop()

        # Load sheets
        tasks_sheet_name = "Tasks" if "Tasks" in xl.sheet_names else xl.sheet_names[0]
        projects_sheet_name = "Projects" if "Projects" in xl.sheet_names else None

        df_tasks_raw = pd.read_excel(xl, sheet_name=tasks_sheet_name)
        df_projects_raw = pd.read_excel(xl, sheet_name=projects_sheet_name) if projects_sheet_name else pd.DataFrame()

        df_tasks = normalize_tasks_df(df_tasks_raw)
        df_projects = normalize_projects_df(df_projects_raw) if not df_projects_raw.empty else pd.DataFrame()

        # Determine week_ending
        week_from_file = None
        if "week_ending" in df_tasks.columns and df_tasks["week_ending"].notna().any():
            week_from_file = df_tasks["week_ending"].dropna().astype(str).iloc[0].strip()
        if not week_from_file:
            week_from_file = parse_week_ending_from_filename(uploaded.name)

        if not week_from_file:
            st.error("Could not determine Week Ending. Ensure the Tasks sheet has a 'Week Ending' column or the filename includes YYYY-MM-DD.")
            st.stop()

        # Force week_ending into all rows (tasks & projects)
        df_tasks["week_ending"] = week_from_file
        if not df_projects.empty:
            df_projects["week_ending"] = week_from_file

        st.write(f"Detected Week Ending: **{week_from_file}**")
        st.write("Preview (first 20 Tasks rows):")
        preview = df_tasks.copy()
        preview["duration_hhmmss"] = preview["duration_seconds"].apply(seconds_to_hhmmss)
        st.dataframe(preview.head(20), use_container_width=True)

        if st.button("Upload to Database", type="primary"):
            with ENGINE.begin() as conn:
                if overwrite_week:
                    delete_week(conn, week_from_file)
                insert_tasks(conn, df_tasks)
                if not df_projects.empty:
                    insert_projects(conn, df_projects)

            st.success(f"Uploaded week {week_from_file}. Weekly dashboard should now show this week.")

# =============================
# Weekly
# =============================
with tabs[0]:
    st.subheader("Weekly View")

    members = list_team_members()
    if not members:
        st.warning("No data yet. Go to Admin Upload and upload a combined weekly workbook.")
        st.stop()

    sel_member = st.selectbox("Team Member", members, index=0)
    weeks = list_weeks_for_member(sel_member)
    if not weeks:
        st.warning("No weeks found for this team member.")
        st.stop()

    sel_week = st.selectbox("Week Ending", weeks, index=0)

    dfw = fetch_week_tasks(sel_member, sel_week)
    if dfw.empty:
        st.warning("No tasks found for this selection.")
        st.stop()

    # Split coverage vs non-coverage
    dfw["task_type"] = dfw["task_type"].fillna("").astype(str)
    is_cov = dfw["task_type"].str.strip().str.lower().eq("coverage")
    df_cov = dfw[is_cov].copy()
    df_non = dfw[~is_cov].copy()

    # Ensure duration display
    df_non["duration_hhmmss"] = df_non["duration_seconds"].apply(seconds_to_hhmmss)

    # -------------------------
    # NON-NEGOTIABLE: Summary (unique Task ID + Task Description, once)
    # Excluding coverage
    # -------------------------
    st.markdown("### Weekly Summary (Unique Tasks)")
    summary = (
        df_non[["task_id", "task_description"]]
        .dropna(subset=["task_id"])
        .drop_duplicates()
        .sort_values(["task_id"])
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # -------------------------
    # Coverage section: below summary, above raw
    # Only need task_id + volume (if present)
    # -------------------------
    st.markdown("### Coverage")
    if df_cov.empty:
        st.info("No coverage entries for this week.")
    else:
        cov_view = df_cov[["task_id", "volume"]].copy()
        cov_view["task_id"] = cov_view["task_id"].fillna("").astype(str)
        cov_grouped = cov_view.groupby("task_id", as_index=False).agg({"volume": "sum"})
        # Only show volume if > 0 (your requirement: "if they put volume")
        cov_grouped["volume"] = cov_grouped["volume"].fillna(0).astype(int)
        st.dataframe(cov_grouped, use_container_width=True, hide_index=True)

    # -------------------------
    # Raw data (non coverage): Task ID, Duration hh:mm:ss, Volume, Day
    # -------------------------
    st.markdown("### Raw Weekly Data (Non-Coverage)")
    raw = df_non[["task_id", "duration_hhmmss", "volume", "day"]].copy()
    raw = raw.rename(columns={"duration_hhmmss": "duration (hh:mm:ss)"})
    # Keep a stable day ordering if needed
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    raw["day"] = raw["day"].astype(str)
    raw["day_sort"] = raw["day"].apply(lambda d: day_order.index(d) if d in day_order else 99)
    raw = raw.sort_values(["day_sort", "task_id"]).drop(columns=["day_sort"])
    st.dataframe(raw, use_container_width=True, hide_index=True)

    # -------------------------
    # Projects section (weekly)
    # -------------------------
    st.markdown("### Projects (Weekly)")
    dfp = fetch_week_projects(sel_member, sel_week)
    if dfp.empty:
        st.info("No projects found for this owner/week.")
    else:
        proj_cols = ["project_name", "status", "days_active", "notes"]
        dfp_view = dfp[proj_cols].copy()
        st.dataframe(dfp_view, use_container_width=True, hide_index=True)

# =============================
# Monthly + Quarterly placeholders (we’ll implement once DB is active / you approve metrics)
# =============================
with tabs[1]:
    st.subheader("Monthly View")
    st.info("Next: month-to-month comparisons for selected team member (hours, volume, top tasks, coverage volume, projects by status).")

with tabs[2]:
    st.subheader("Quarterly View")
    st.info("Next: quarter-to-quarter comparisons for selected team member (same metrics).")
