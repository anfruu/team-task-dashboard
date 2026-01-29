# dashboard.py
# MAO Workflow Tracker Dashboard (Streamlit)
# - Weekly / Monthly / Quarterly / Admin Upload tabs (layout preserved)
# - Volume is NUMERIC ONLY (INTEGER everywhere). Any non-numeric volume becomes 0 on upload.
# - Weekly adds: Task Summary rename, Coverage day column, Production (Primary) rename + day filter,
#   and Production (Backup) section.

import os
import re
from io import BytesIO
from typing import List, Optional

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


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def normalize_tasks_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected from combined Tasks sheet (your screenshot):
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
        # Try minutes/hours
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

    # Duration seconds numeric int
    out["duration_seconds"] = pd.to_numeric(out["duration_seconds"], errors="coerce").fillna(0).astype(int)

    # Drop blanks
    out = out.dropna(subset=["task_id"]).copy()
    out = out[out["task_id"].astype(str).str.strip() != ""].copy()

    return out


def normalize_projects_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected from combined Projects sheet (your screenshot):
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

    # Dates: keep as string (consistent with your current approach)
    for c in ["start_date", "end_date"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: "" if pd.isna(x) else str(x))

    # Days active
    if "days_active" in out.columns:
        out["days_active"] = pd.to_numeric(out["days_active"], errors="coerce").fillna(0).astype(int)
    else:
        out["days_active"] = 0

    # Drop blanks
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
            # Postgres
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

            # Optional indexes
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


# -----------------------------
# Upload / overwrite helpers
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
# UI: Tabs (do not change order/labels)
# -----------------------------
tabs = st.tabs(["Weekly", "Monthly", "Quarterly", "Admin Upload"])

# =============================
# Weekly Tab
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

            # -------------------------
            # 2) Task Summary (Production tasks, unique)
            # -------------------------
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
                    # Leave description blank if missing (already blank strings in your data)
                    st.dataframe(summary, use_container_width=True, hide_index=True)

            # -------------------------
            # 3) Coverage (add Day column)
            # -------------------------
            st.markdown("### Coverage")

            if df.empty:
                st.info("No coverage entries for this week.")
            else:
                df_cov = df[df["task_type_norm"] == "coverage"].copy()
                if df_cov.empty:
                    st.info("No coverage entries for this week.")
                else:
                    cov_view = df_cov[["task_id", "day", "volume"]].copy()
                    # Keep stable day ordering
                    cov_view["day_sort"] = cov_view["day"].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                    cov_view = cov_view.sort_values(["task_id", "day_sort"]).drop(columns=["day_sort"])
                    st.dataframe(cov_view, use_container_width=True, hide_index=True)

            # -------------------------
            # 4) Production (Primary) + day filter
            # -------------------------
            st.markdown("### Production (Primary)")

            if df.empty:
                st.info("No production (primary) entries for this week.")
            else:
                df_primary = df[(df["task_type_norm"] == "production") & (df["role_type_norm"] == "primary")].copy()

                if df_primary.empty:
                    st.info("No production (primary) entries for this week.")
                else:
                    # Day filter (simple + clean)
                    days_present = [d for d in DAY_ORDER if d in set(df_primary["day"].astype(str))]
                    day_choice = st.selectbox("Filter by Day (optional)", ["All"] + days_present, index=0)

                    view = df_primary[["task_id", "duration_hhmmss", "volume", "day"]].copy()
                    if day_choice != "All":
                        view = view[view["day"] == day_choice].copy()

                    view["day_sort"] = view["day"].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                    view = view.sort_values(["day_sort", "task_id"]).drop(columns=["day_sort"])
                    view = view.rename(columns={"duration_hhmmss": "duration (hh:mm:ss)"})

                    st.dataframe(view, use_container_width=True, hide_index=True)

            # -------------------------
            # 5) Production (Backup)
            # -------------------------
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

            # -------------------------
            # 6) Projects (Weekly) unchanged layout
            # -------------------------
            st.markdown("### Projects (Weekly)")

            if dfp.empty:
                st.info("No projects found for this owner/week.")
            else:
                proj_cols = ["project_name", "status", "days_active", "notes"]
                dfp_view = dfp[proj_cols].copy()
                st.dataframe(dfp_view, use_container_width=True, hide_index=True)

# =============================
# Monthly Tab (layout placeholder preserved)
# =============================
with tabs[1]:
    st.subheader("Monthly View")
    st.info("Next: month-to-month comparisons for selected team member (hours, volume, top tasks, coverage volume, projects by status).")

# =============================
# Quarterly Tab (layout placeholder preserved)
# =============================
with tabs[2]:
    st.subheader("Quarterly View")
    st.info("Next: quarter-to-quarter comparisons for selected team member (same metrics).")

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

            # Determine week_ending (expect present in file)
            week_vals = sorted(set([w for w in tasks_df["week_ending"].unique() if str(w).strip() != ""]))
            if not week_vals:
                st.error("Could not find 'Week Ending' values in Tasks sheet.")
                st.stop()

            # If multiple, take the most common / first (your files should be single-week)
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
