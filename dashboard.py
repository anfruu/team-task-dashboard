# dashboard.py
# MAO Workflow Tracker Dashboard
# Executive-ready version
# - MAS-style layout/colors
# - Postgres-backed
# - Weekly / Monthly Overview / Monthly / Quarterly / Admin Upload
# - Production uses raw Duration Seconds
# - Coverage duration is derived from remaining 8-hour day time
# - Supports full replace, month replace, and week replace uploads

import os
from io import BytesIO

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.express as px

# ---------------------------------
# App config
# ---------------------------------
st.set_page_config(page_title="MAO Workflow Tracker Dashboard", layout="wide")

# ---------------------------------
# Styling
# ---------------------------------
TEXT_COLOR = "#102033"
SUBTEXT_COLOR = "#556476"
BORDER = "#D9E2EC"
CARD_BG = "#F7FAFC"
PAGE_BG = "#F4F8FB"
SECTION_BG = "#FFFFFF"

PRIMARY = "#2F5D8C"
SECONDARY = "#4F8A8B"
ACCENT = "#7A6FA6"
WARM = "#C28B52"
SOFT_RED = "#B86A6A"
SLATE = "#60758A"

st.markdown(
    f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, {PAGE_BG} 0%, #EEF3F7 100%);
    }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1520px;
    }}

    html, body, [class*="css"] {{
        color: {TEXT_COLOR} !important;
        font-family: "Segoe UI", "Inter", sans-serif;
    }}

    h1 {{
        color: {TEXT_COLOR} !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.12rem !important;
    }}

    h2, h3, h4, h5, h6 {{
        color: {TEXT_COLOR} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}

    p, label, .stCaption {{
        color: {SUBTEXT_COLOR} !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        border-bottom: none;
        padding-bottom: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        background-color: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding-left: 18px;
        padding-right: 18px;
        color: {TEXT_COLOR} !important;
        font-weight: 700;
        box-shadow: 0 1px 2px rgba(16, 32, 51, 0.04);
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(180deg, #F7FBFF 0%, #EEF5FB 100%) !important;
        border-color: #C9D8E6 !important;
        color: {PRIMARY} !important;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, #FFFFFF 0%, {CARD_BG} 100%);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 3px 10px rgba(16, 32, 51, 0.04);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {SUBTEXT_COLOR} !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {TEXT_COLOR} !important;
        font-weight: 800 !important;
        font-size: 1.65rem !important;
    }}

    .stSelectbox label {{
        color: {TEXT_COLOR} !important;
        font-weight: 700 !important;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(16, 32, 51, 0.03);
    }}

    .section-shell {{
        background: {SECTION_BG};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 16px 18px;
        margin-top: 0.45rem;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(16, 32, 51, 0.04);
    }}

    .section-title {{
        color: {TEXT_COLOR};
        font-weight: 800;
        font-size: 1.05rem;
        margin-bottom: 0.18rem;
    }}

    .section-subtitle {{
        color: {SUBTEXT_COLOR};
        font-size: 0.92rem;
        margin-bottom: 0;
    }}
</style>
""",
    unsafe_allow_html=True,
)

st.title("MAO Workflow Tracker Dashboard")
st.caption("LPL Financial – Operations")

# ---------------------------------
# Constants / Helpers
# ---------------------------------
DAY_TO_OFFSET = {
    "Monday": 4,
    "Tuesday": 3,
    "Wednesday": 2,
    "Thursday": 1,
    "Friday": 0,
    "Saturday": -1,
    "Sunday": -2,
}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def section_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", " ").replace("\r", "").replace("\xa0", " ") for c in df.columns]
    return df


def _clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def fmt_int(x) -> str:
    try:
        return f"{int(x):,d}"
    except Exception:
        return "0"


def fmt_hours(x) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "0.00"


def seconds_to_hhmm(seconds) -> str:
    try:
        seconds = int(round(float(seconds)))
    except Exception:
        seconds = 0
    seconds = max(0, seconds)
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    return f"{hh:02d}:{mm:02d}"


def seconds_to_hours(seconds) -> float:
    try:
        return round(float(seconds) / 3600.0, 2)
    except Exception:
        return 0.0


def derive_work_date(week_ending: pd.Series, day_name: pd.Series) -> pd.Series:
    week_dt = pd.to_datetime(week_ending, errors="coerce")
    day_clean = day_name.astype(str).str.strip()
    offsets = day_clean.map(DAY_TO_OFFSET)
    return week_dt - pd.to_timedelta(offsets, unit="D")


def style_delta_df(df: pd.DataFrame, delta_cols: list[str]):
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

    try:
        sty = df.style
        for c in delta_cols:
            if c in df.columns:
                sty = sty.applymap(color, subset=[c])
        return sty
    except Exception:
        return df


def apply_layout(fig, height=360, show_legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=52, b=18),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color=TEXT_COLOR, size=13),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_COLOR, size=12),
            title=None,
        ),
        showlegend=show_legend,
    )
    fig.update_xaxes(
        title_font=dict(color=TEXT_COLOR, size=13),
        tickfont=dict(color=TEXT_COLOR),
        gridcolor="#E6EDF3",
        zeroline=False,
    )
    fig.update_yaxes(
        title_font=dict(color=TEXT_COLOR, size=13),
        tickfont=dict(color=TEXT_COLOR),
        gridcolor="#E6EDF3",
        zeroline=False,
    )
    return fig


def period_delta_str(curr, prev, is_int=False):
    if prev is None:
        return ""
    try:
        if is_int:
            return f"{int(curr) - int(prev):+d}"
        return f"{float(curr) - float(prev):+,.2f}"
    except Exception:
        return ""


def task_bucket(task_id: str) -> str:
    s = _clean_str(task_id).lower()
    if "mailbox" in s:
        return "Mailbox"
    if "dynamics" in s or "cases & tasks" in s or "cases" in s:
        return "Dynamics Cases & Tasks"
    return "Other"


# ---------------------------------
# DB / Engine
# ---------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if not DATABASE_URL:
    st.error(
        "DATABASE_URL is not set. Add it in Render Environment variables.\n\n"
        "Example: postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DBNAME?sslmode=require"
    )
    st.stop()

ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
IS_SQLITE = ENGINE.dialect.name == "sqlite"

# ---------------------------------
# DB init
# ---------------------------------
def init_db():
    with ENGINE.begin() as conn:
        if IS_SQLITE:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT,
                        team_member TEXT,
                        task_type TEXT,
                        role_type TEXT,
                        raw_duration_seconds INTEGER,
                        effective_duration_seconds INTEGER,
                        volume INTEGER,
                        day TEXT,
                        week_ending TEXT,
                        work_date TEXT,
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
                        task_type TEXT,
                        role_type TEXT,
                        raw_duration_seconds INTEGER,
                        effective_duration_seconds INTEGER,
                        volume INTEGER,
                        day TEXT,
                        week_ending TEXT,
                        work_date TEXT,
                        uploaded_at TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
            )
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS task_type TEXT"))
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS role_type TEXT"))
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS raw_duration_seconds INTEGER"))
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS effective_duration_seconds INTEGER"))
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS volume INTEGER"))
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS day TEXT"))
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS week_ending TEXT"))
            conn.execute(text("ALTER TABLE tasks ADD COLUMN IF NOT EXISTS work_date TEXT"))

            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tasks_member_week ON tasks(team_member, week_ending)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tasks_member_workdate ON tasks(team_member, work_date)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tasks_taskid ON tasks(task_id)"))


init_db()

# ---------------------------------
# Normalization / Coverage Logic
# ---------------------------------
EXPECTED_MAP = {
    "Task ID": "task_id",
    "Team Member": "team_member",
    "Task Type": "task_type",
    "Role Type": "role_type",
    "Duration Seconds": "raw_duration_seconds",
    "Volume": "volume",
    "Day": "day",
    "Week Ending": "week_ending",
}


def normalize_tasks_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(EXPECTED_MAP.values()))

    df = clean_cols(df)
    missing = [src for src in EXPECTED_MAP if src not in df.columns]
    if missing:
        return pd.DataFrame(columns=list(EXPECTED_MAP.values()))

    out = pd.DataFrame()
    for src, dst in EXPECTED_MAP.items():
        out[dst] = df[src]

    for c in ["task_id", "team_member", "task_type", "role_type", "day"]:
        out[c] = out[c].apply(_clean_str)

    out["raw_duration_seconds"] = pd.to_numeric(out["raw_duration_seconds"], errors="coerce").fillna(0).astype(int)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
    out["week_ending"] = pd.to_datetime(out["week_ending"], errors="coerce")

    out = out.dropna(subset=["week_ending"]).copy()
    out = out[out["task_id"].astype(str).str.strip() != ""].copy()
    out = out[out["team_member"].astype(str).str.strip() != ""].copy()

    out["task_type"] = out["task_type"].astype(str).str.strip().str.title()
    out["role_type"] = out["role_type"].astype(str).str.strip().str.title()
    out["day"] = out["day"].astype(str).str.strip().str.title()
    out["week_ending"] = out["week_ending"].dt.strftime("%Y-%m-%d")
    return out


def apply_coverage_logic(tasks_df: pd.DataFrame) -> pd.DataFrame:
    if tasks_df is None or tasks_df.empty:
        return tasks_df

    df = tasks_df.copy()
    df["week_ending_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
    df["work_date_dt"] = derive_work_date(df["week_ending_dt"], df["day"])
    df["work_date"] = df["work_date_dt"].dt.strftime("%Y-%m-%d")

    df["raw_duration_seconds"] = pd.to_numeric(df["raw_duration_seconds"], errors="coerce").fillna(0).astype(int)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    production_mask = df["task_type"].str.strip().str.lower() == "production"
    coverage_mask = df["task_type"].str.strip().str.lower() == "coverage"

    prod_daily = (
        df[production_mask]
        .groupby(["team_member", "work_date"], as_index=False)["raw_duration_seconds"]
        .sum()
        .rename(columns={"raw_duration_seconds": "production_seconds_total"})
    )

    cov_daily = (
        df[coverage_mask]
        .groupby(["team_member", "work_date"], as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "coverage_volume_total"})
    )

    daily = prod_daily.merge(cov_daily, on=["team_member", "work_date"], how="outer").fillna(0)
    daily["production_seconds_total"] = pd.to_numeric(daily["production_seconds_total"], errors="coerce").fillna(0)
    daily["coverage_volume_total"] = pd.to_numeric(daily["coverage_volume_total"], errors="coerce").fillna(0)

    daily["remaining_seconds"] = (8 * 3600) - daily["production_seconds_total"]
    daily["remaining_seconds"] = daily["remaining_seconds"].clip(lower=0)

    daily["coverage_seconds_per_unit"] = 0.0
    mask_cov_units = daily["coverage_volume_total"] > 0
    daily.loc[mask_cov_units, "coverage_seconds_per_unit"] = (
        daily.loc[mask_cov_units, "remaining_seconds"] / daily.loc[mask_cov_units, "coverage_volume_total"]
    )

    df = df.merge(
        daily[["team_member", "work_date", "coverage_seconds_per_unit"]],
        on=["team_member", "work_date"],
        how="left",
    )
    df["coverage_seconds_per_unit"] = df["coverage_seconds_per_unit"].fillna(0.0)

    df["effective_duration_seconds"] = df["raw_duration_seconds"]
    df.loc[coverage_mask, "effective_duration_seconds"] = (
        df.loc[coverage_mask, "coverage_seconds_per_unit"] * df.loc[coverage_mask, "volume"]
    ).round().astype(int)

    df["effective_duration_seconds"] = pd.to_numeric(df["effective_duration_seconds"], errors="coerce").fillna(0).astype(int)
    return df.drop(columns=["week_ending_dt", "work_date_dt", "coverage_seconds_per_unit"], errors="ignore")


def load_month_file(uploaded_file) -> pd.DataFrame:
    data = BytesIO(uploaded_file.read())
    xls = pd.ExcelFile(data)

    frames = []
    for sheet in xls.sheet_names:
        if sheet.strip().lower() == "unmatched task ids":
            continue
        raw = pd.read_excel(xls, sheet_name=sheet)
        norm = normalize_tasks_df(raw)
        if not norm.empty:
            frames.append(norm)

    if not frames:
        return pd.DataFrame(columns=list(EXPECTED_MAP.values()))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------
# DB helpers
# ---------------------------------
def replace_all_task_data(df: pd.DataFrame):
    with ENGINE.begin() as conn:
        conn.execute(text("DELETE FROM tasks"))
        if df.empty:
            return

        for _, r in df.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO tasks (
                        task_id, team_member, task_type, role_type,
                        raw_duration_seconds, effective_duration_seconds,
                        volume, day, week_ending, work_date, uploaded_at
                    )
                    VALUES (
                        :task_id, :team_member, :task_type, :role_type,
                        :raw_duration_seconds, :effective_duration_seconds,
                        :volume, :day, :week_ending, :work_date, :uploaded_at
                    )
                    """
                ),
                {
                    "task_id": r["task_id"],
                    "team_member": r["team_member"],
                    "task_type": r["task_type"],
                    "role_type": r["role_type"],
                    "raw_duration_seconds": int(r["raw_duration_seconds"]),
                    "effective_duration_seconds": int(r["effective_duration_seconds"]),
                    "volume": int(r["volume"]),
                    "day": r["day"],
                    "week_ending": r["week_ending"],
                    "work_date": r["work_date"],
                    "uploaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                },
            )


def replace_selected_months(df: pd.DataFrame):
    months = pd.to_datetime(df["week_ending"], errors="coerce").dt.to_period("M").astype(str).dropna().unique().tolist()

    with ENGINE.begin() as conn:
        for month in months:
            conn.execute(
                text(
                    """
                    DELETE FROM tasks
                    WHERE TO_CHAR(TO_DATE(week_ending, 'YYYY-MM-DD'), 'YYYY-MM') = :month
                    """
                    if not IS_SQLITE
                    else """
                    DELETE FROM tasks
                    WHERE substr(week_ending, 1, 7) = :month
                    """
                ),
                {"month": month},
            )

        for _, r in df.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO tasks (
                        task_id, team_member, task_type, role_type,
                        raw_duration_seconds, effective_duration_seconds,
                        volume, day, week_ending, work_date, uploaded_at
                    )
                    VALUES (
                        :task_id, :team_member, :task_type, :role_type,
                        :raw_duration_seconds, :effective_duration_seconds,
                        :volume, :day, :week_ending, :work_date, :uploaded_at
                    )
                    """
                ),
                {
                    "task_id": r["task_id"],
                    "team_member": r["team_member"],
                    "task_type": r["task_type"],
                    "role_type": r["role_type"],
                    "raw_duration_seconds": int(r["raw_duration_seconds"]),
                    "effective_duration_seconds": int(r["effective_duration_seconds"]),
                    "volume": int(r["volume"]),
                    "day": r["day"],
                    "week_ending": r["week_ending"],
                    "work_date": r["work_date"],
                    "uploaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                },
            )


def replace_selected_weeks(df: pd.DataFrame):
    weeks = df["week_ending"].dropna().astype(str).unique().tolist()

    with ENGINE.begin() as conn:
        for wk in weeks:
            conn.execute(text("DELETE FROM tasks WHERE week_ending = :w"), {"w": wk})

        for _, r in df.iterrows():
            conn.execute(
                text(
                    """
                    INSERT INTO tasks (
                        task_id, team_member, task_type, role_type,
                        raw_duration_seconds, effective_duration_seconds,
                        volume, day, week_ending, work_date, uploaded_at
                    )
                    VALUES (
                        :task_id, :team_member, :task_type, :role_type,
                        :raw_duration_seconds, :effective_duration_seconds,
                        :volume, :day, :week_ending, :work_date, :uploaded_at
                    )
                    """
                ),
                {
                    "task_id": r["task_id"],
                    "team_member": r["team_member"],
                    "task_type": r["task_type"],
                    "role_type": r["role_type"],
                    "raw_duration_seconds": int(r["raw_duration_seconds"]),
                    "effective_duration_seconds": int(r["effective_duration_seconds"]),
                    "volume": int(r["volume"]),
                    "day": r["day"],
                    "week_ending": r["week_ending"],
                    "work_date": r["work_date"],
                    "uploaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                },
            )


def fetch_all_tasks() -> pd.DataFrame:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                    task_id, team_member, task_type, role_type,
                    raw_duration_seconds, effective_duration_seconds,
                    volume, day, week_ending, work_date
                FROM tasks
                """
            )
        ).fetchall()

    return pd.DataFrame(
        rows,
        columns=[
            "task_id", "team_member", "task_type", "role_type",
            "raw_duration_seconds", "effective_duration_seconds",
            "volume", "day", "week_ending", "work_date"
        ],
    )


def list_team_members() -> list[str]:
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


def list_weeks_for_member(member: str) -> list[str]:
    with ENGINE.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT week_ending
                FROM tasks
                WHERE team_member = :m
                ORDER BY week_ending DESC
                """
            ),
            {"m": member},
        ).fetchall()
    return [r[0] for r in rows]


def db_row_count() -> int:
    with ENGINE.begin() as conn:
        val = conn.execute(text("SELECT COUNT(*) FROM tasks")).scalar()
    return int(val or 0)


# ---------------------------------
# Rollups
# ---------------------------------
def prep_for_rollups(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["week_date"] = pd.to_datetime(out["week_ending"], errors="coerce")
    out["work_date_dt"] = pd.to_datetime(out["work_date"], errors="coerce")
    out["task_type_norm"] = out["task_type"].astype(str).str.strip().str.lower()
    out["role_type_norm"] = out["role_type"].astype(str).str.strip().str.lower()
    out["month"] = out["week_date"].dt.to_period("M").astype(str)
    out["quarter"] = out["week_date"].dt.to_period("Q").astype(str)
    out["month_label"] = out["week_date"].dt.strftime("%b %Y")
    out["month_sort"] = out["week_date"].dt.to_period("M").astype(str)
    out["task_bucket"] = out["task_id"].apply(task_bucket)
    return out


def compute_top_metrics(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "prod_primary_seconds": 0,
            "prod_backup_seconds": 0,
            "coverage_seconds": 0,
            "production_volume": 0,
            "coverage_volume": 0,
        }

    prod_primary = df[(df["task_type_norm"] == "production") & (df["role_type_norm"] == "primary")]
    prod_backup = df[(df["task_type_norm"] == "production") & (df["role_type_norm"] == "backup")]
    coverage = df[df["task_type_norm"] == "coverage"]
    production_all = df[df["task_type_norm"] == "production"]

    return {
        "prod_primary_seconds": int(prod_primary["effective_duration_seconds"].sum()),
        "prod_backup_seconds": int(prod_backup["effective_duration_seconds"].sum()),
        "coverage_seconds": int(coverage["effective_duration_seconds"].sum()),
        "production_volume": int(production_all["volume"].sum()),
        "coverage_volume": int(coverage["volume"].sum()),
    }


def build_task_kpi_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("task_id", as_index=False)
        .agg(
            total_seconds=("effective_duration_seconds", "sum"),
            total_volume=("volume", "sum"),
        )
        .sort_values(["total_seconds", "task_id"], ascending=[False, True])
    )

    grouped["Total Duration (hh:mm)"] = grouped["total_seconds"].apply(seconds_to_hhmm)
    grouped["Volume"] = grouped["total_volume"].astype(int)

    return grouped.rename(columns={"task_id": "Task ID"})[["Task ID", "Total Duration (hh:mm)", "Volume"]]


def build_task_kpi_table_with_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("task_id", as_index=False)
        .agg(
            total_seconds=("effective_duration_seconds", "sum"),
            total_volume=("volume", "sum"),
        )
        .sort_values(["total_seconds", "task_id"], ascending=[False, True])
    )

    grouped["Total Duration (hh:mm)"] = grouped["total_seconds"].apply(seconds_to_hhmm)
    grouped["Hours"] = grouped["total_seconds"].apply(seconds_to_hours)
    grouped["Volume"] = grouped["total_volume"].astype(int)

    return grouped.rename(columns={"task_id": "Task ID"})[["Task ID", "Total Duration (hh:mm)", "Hours", "Volume"]]


def add_deltas(curr_df: pd.DataFrame, prev_df: pd.DataFrame, key_col: str = "Task ID") -> pd.DataFrame:
    if curr_df is None or curr_df.empty:
        return pd.DataFrame()

    curr = curr_df.copy()
    if prev_df is None or prev_df.empty:
        if "Hours" in curr.columns:
            curr["Δ Hours"] = 0.0
        curr["Δ Volume"] = 0
        return curr

    prev = prev_df.copy()
    keep_cols = [key_col]
    if "Hours" in prev.columns:
        keep_cols.append("Hours")
    if "Volume" in prev.columns:
        keep_cols.append("Volume")

    prev = prev[keep_cols].copy()
    if "Hours" in prev.columns:
        prev = prev.rename(columns={"Hours": "Hours_prev"})
    if "Volume" in prev.columns:
        prev = prev.rename(columns={"Volume": "Volume_prev"})

    merged = curr.merge(prev, on=key_col, how="left")

    if "Hours" in merged.columns:
        merged["Δ Hours"] = merged["Hours"] - merged.get("Hours_prev", 0).fillna(0)
    if "Volume" in merged.columns:
        merged["Δ Volume"] = merged["Volume"] - merged.get("Volume_prev", 0).fillna(0)

    drop_cols = [c for c in ["Hours_prev", "Volume_prev"] if c in merged.columns]
    return merged.drop(columns=drop_cols)


# ---------------------------------
# Tabs
# ---------------------------------
tabs = st.tabs(["Weekly", "Monthly Overview", "Monthly", "Quarterly", "Admin Upload"])

# =================================
# Weekly
# =================================
with tabs[0]:
    section_header(
        "Weekly View",
        "Weekly task performance with production and coverage durations derived from the cleaned monthly uploads."
    )

    members = list_team_members()
    if not members:
        st.info("No data loaded yet. Upload data in Admin Upload.")
    else:
        sel_member = st.selectbox("Team Member", members, index=0)
        weeks = list_weeks_for_member(sel_member)

        if not weeks:
            st.info("No weekly data found for this team member.")
        else:
            sel_week = st.selectbox("Week Ending", weeks, index=0)

            all_df = fetch_all_tasks()
            df = all_df[(all_df["team_member"] == sel_member) & (all_df["week_ending"] == sel_week)].copy()

            if df.empty:
                st.info("No entries found for this week.")
            else:
                df = prep_for_rollups(df)
                totals = compute_top_metrics(df)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Production (Primary) Hours", fmt_hours(seconds_to_hours(totals["prod_primary_seconds"])))
                m2.metric("Production (Backup) Hours", fmt_hours(seconds_to_hours(totals["prod_backup_seconds"])))
                m3.metric("Coverage Hours", fmt_hours(seconds_to_hours(totals["coverage_seconds"])))
                m4.metric("Coverage Volume", fmt_int(totals["coverage_volume"]))

                st.markdown("<br>", unsafe_allow_html=True)

                section_header("Production (Primary)", "Timed production work for the selected week.")
                primary = df[(df["task_type_norm"] == "production") & (df["role_type_norm"] == "primary")].copy()
                if primary.empty:
                    st.info("No primary production entries for this week.")
                else:
                    view = primary[["task_id", "day", "volume", "effective_duration_seconds"]].copy()
                    view["Duration (hh:mm)"] = view["effective_duration_seconds"].apply(seconds_to_hhmm)
                    view["day_sort"] = view["day"].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                    view = view.sort_values(["day_sort", "task_id"]).drop(columns=["day_sort", "effective_duration_seconds"])
                    view = view.rename(columns={"task_id": "Task ID", "day": "Day", "volume": "Volume"})
                    st.dataframe(view, use_container_width=True, hide_index=True)

                section_header("Production (Backup)", "Timed backup production work for the selected week.")
                backup = df[(df["task_type_norm"] == "production") & (df["role_type_norm"] == "backup")].copy()
                if backup.empty:
                    st.info("No backup production entries for this week.")
                else:
                    view = backup[["task_id", "day", "volume", "effective_duration_seconds"]].copy()
                    view["Duration (hh:mm)"] = view["effective_duration_seconds"].apply(seconds_to_hhmm)
                    view["day_sort"] = view["day"].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                    view = view.sort_values(["day_sort", "task_id"]).drop(columns=["day_sort", "effective_duration_seconds"])
                    view = view.rename(columns={"task_id": "Task ID", "day": "Day", "volume": "Volume"})
                    st.dataframe(view, use_container_width=True, hide_index=True)

                section_header("Coverage", "Coverage duration is allocated from the remaining time in the 8-hour day after production.")
                coverage = df[df["task_type_norm"] == "coverage"].copy()
                if coverage.empty:
                    st.info("No coverage entries for this week.")
                else:
                    view = coverage[["task_id", "day", "volume", "effective_duration_seconds"]].copy()
                    view["Allocated Duration (hh:mm)"] = view["effective_duration_seconds"].apply(seconds_to_hhmm)
                    view["day_sort"] = view["day"].apply(lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                    view = view.sort_values(["day_sort", "task_id"]).drop(columns=["day_sort", "effective_duration_seconds"])
                    view = view.rename(columns={"task_id": "Task ID", "day": "Day", "volume": "Volume"})
                    st.dataframe(view, use_container_width=True, hide_index=True)

                section_header("Weekly Task Totals", "Task-level weekly totals across all entries for the selected week.")
                weekly_totals = build_task_kpi_table(df)
                if weekly_totals.empty:
                    st.info("No weekly totals available.")
                else:
                    st.dataframe(weekly_totals, use_container_width=True, hide_index=True)

# =================================
# Monthly Overview
# =================================
with tabs[1]:
    section_header(
        "Monthly Overview",
        "High-level month-over-month trends, top task drivers, and production oversight."
    )

    members = list_team_members()
    if not members:
        st.info("No data loaded yet. Upload data in Admin Upload.")
    else:
        sel_member = st.selectbox("Team Member", members, index=0, key="overview_member")
        all_df = prep_for_rollups(fetch_all_tasks())
        member_df = all_df[all_df["team_member"] == sel_member].copy()

        if member_df.empty:
            st.info("No data for this team member.")
        else:
            months = sorted(member_df["month"].dropna().unique().tolist())
            sel_month = st.selectbox("Month", months, index=len(months) - 1, key="overview_month")
            cur_df = member_df[member_df["month"] == sel_month].copy()

            month_summary = (
                member_df.groupby(["month_sort", "month_label"], as_index=False)
                .agg(
                    production_seconds=("effective_duration_seconds", lambda s: int(member_df.loc[s.index][member_df.loc[s.index, "task_type_norm"] == "production"]["effective_duration_seconds"].sum())),
                    production_volume=("volume", lambda s: int(member_df.loc[s.index][member_df.loc[s.index, "task_type_norm"] == "production"]["volume"].sum())),
                    coverage_volume=("volume", lambda s: int(member_df.loc[s.index][member_df.loc[s.index, "task_type_norm"] == "coverage"]["volume"].sum())),
                )
                .sort_values("month_sort")
            )

            top = compute_top_metrics(cur_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Production Hours", fmt_hours(seconds_to_hours(top["prod_primary_seconds"] + top["prod_backup_seconds"])))
            c2.metric("Coverage Hours", fmt_hours(seconds_to_hours(top["coverage_seconds"])))
            c3.metric("Production Volume", fmt_int(top["production_volume"]))
            c4.metric("Coverage Volume", fmt_int(top["coverage_volume"]))

            st.markdown("<br>", unsafe_allow_html=True)

            if not month_summary.empty:
                fig = px.bar(
                    month_summary,
                    x="month_label",
                    y="production_volume",
                    text_auto=True,
                    title="Production Volume by Month",
                    color_discrete_sequence=[PRIMARY],
                )
                fig = apply_layout(fig, height=320, show_legend=False)
                fig.update_xaxes(title="")
                fig.update_yaxes(title="Volume")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            prod_df = cur_df[cur_df["task_type_norm"] == "production"].copy()

            left, right = st.columns(2)

            with left:
                section_header("Top Tasks by Volume", "Highest production task volume for the selected month.")
                if prod_df.empty:
                    st.info("No production data for this month.")
                else:
                    top_volume = (
                        prod_df.groupby("task_id", as_index=False)["volume"]
                        .sum()
                        .rename(columns={"task_id": "Task ID", "volume": "Volume"})
                        .sort_values(["Volume", "Task ID"], ascending=[False, True])
                        .head(10)
                    )
                    st.dataframe(top_volume, use_container_width=True, hide_index=True)

            with right:
                section_header("Longest Timed Production Tasks", "Production tasks with the highest total duration for the selected month.")
                if prod_df.empty:
                    st.info("No production data for this month.")
                else:
                    longest = (
                        prod_df.groupby("task_id", as_index=False)["effective_duration_seconds"]
                        .sum()
                        .rename(columns={"task_id": "Task ID", "effective_duration_seconds": "Total Seconds"})
                        .sort_values(["Total Seconds", "Task ID"], ascending=[False, True])
                        .head(10)
                    )
                    longest["Total Duration (hh:mm)"] = longest["Total Seconds"].apply(seconds_to_hhmm)
                    longest = longest.drop(columns=["Total Seconds"])
                    st.dataframe(longest, use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)

            bucket_left, bucket_right = st.columns(2)

            with bucket_left:
                section_header("Mailbox Tasks", "Mailbox production tasks for the selected month.")
                mailbox = prod_df[prod_df["task_bucket"] == "Mailbox"].copy()
                if mailbox.empty:
                    st.info("No mailbox tasks for this month.")
                else:
                    mailbox_tbl = (
                        mailbox.groupby("task_id", as_index=False)
                        .agg(volume=("volume", "sum"), seconds=("effective_duration_seconds", "sum"))
                        .rename(columns={"task_id": "Task ID", "volume": "Volume"})
                        .sort_values(["Volume", "Task ID"], ascending=[False, True])
                    )
                    mailbox_tbl["Total Duration (hh:mm)"] = mailbox_tbl["seconds"].apply(seconds_to_hhmm)
                    mailbox_tbl = mailbox_tbl.drop(columns=["seconds"])
                    st.dataframe(mailbox_tbl, use_container_width=True, hide_index=True)

            with bucket_right:
                section_header("Dynamics Cases & Tasks", "Dynamics-related production tasks for the selected month.")
                dyn = prod_df[prod_df["task_bucket"] == "Dynamics Cases & Tasks"].copy()
                if dyn.empty:
                    st.info("No dynamics cases/tasks for this month.")
                else:
                    dyn_tbl = (
                        dyn.groupby("task_id", as_index=False)
                        .agg(volume=("volume", "sum"), seconds=("effective_duration_seconds", "sum"))
                        .rename(columns={"task_id": "Task ID", "volume": "Volume"})
                        .sort_values(["Volume", "Task ID"], ascending=[False, True])
                    )
                    dyn_tbl["Total Duration (hh:mm)"] = dyn_tbl["seconds"].apply(seconds_to_hhmm)
                    dyn_tbl = dyn_tbl.drop(columns=["seconds"])
                    st.dataframe(dyn_tbl, use_container_width=True, hide_index=True)

# =================================
# Monthly
# =================================
with tabs[2]:
    section_header(
        "Monthly View",
        "Monthly task KPIs using standardized task names and allocated coverage duration."
    )

    members = list_team_members()
    if not members:
        st.info("No data loaded yet. Upload data in Admin Upload.")
    else:
        sel_member = st.selectbox("Team Member", members, index=0, key="monthly_member")
        all_df = prep_for_rollups(fetch_all_tasks())
        member_df = all_df[all_df["team_member"] == sel_member].copy()

        if member_df.empty:
            st.info("No data for this team member.")
        else:
            months = sorted(member_df["month"].dropna().unique().tolist())
            sel_month = st.selectbox("Month", months, index=len(months) - 1)
            cur_df = member_df[member_df["month"] == sel_month].copy()

            prev_month = None
            try:
                prev_month = str(pd.Period(sel_month, freq="M") - 1)
            except Exception:
                prev_month = None

            prev_df = member_df[member_df["month"] == prev_month].copy() if prev_month in months else pd.DataFrame()

            cur_totals = compute_top_metrics(cur_df)
            prev_totals = compute_top_metrics(prev_df) if not prev_df.empty else None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Production (Primary) Hours",
                fmt_hours(seconds_to_hours(cur_totals["prod_primary_seconds"])),
                period_delta_str(seconds_to_hours(cur_totals["prod_primary_seconds"]),
                                 seconds_to_hours(prev_totals["prod_primary_seconds"]) if prev_totals else None),
            )
            c2.metric(
                "Production (Backup) Hours",
                fmt_hours(seconds_to_hours(cur_totals["prod_backup_seconds"])),
                period_delta_str(seconds_to_hours(cur_totals["prod_backup_seconds"]),
                                 seconds_to_hours(prev_totals["prod_backup_seconds"]) if prev_totals else None),
            )
            c3.metric(
                "Coverage Hours",
                fmt_hours(seconds_to_hours(cur_totals["coverage_seconds"])),
                period_delta_str(seconds_to_hours(cur_totals["coverage_seconds"]),
                                 seconds_to_hours(prev_totals["coverage_seconds"]) if prev_totals else None),
            )
            c4.metric(
                "Coverage Volume",
                fmt_int(cur_totals["coverage_volume"]),
                period_delta_str(cur_totals["coverage_volume"],
                                 prev_totals["coverage_volume"] if prev_totals else None,
                                 is_int=True),
            )

            st.markdown("<br>", unsafe_allow_html=True)

            chart_df = (
                member_df.groupby(["month_sort", "month_label"], as_index=False)["volume"]
                .sum()
                .sort_values("month_sort")
            )
            if not chart_df.empty:
                fig = px.bar(
                    chart_df,
                    x="month_label",
                    y="volume",
                    text_auto=True,
                    title="Total Volume by Month",
                    color_discrete_sequence=[PRIMARY],
                )
                fig = apply_layout(fig, height=320, show_legend=False)
                fig.update_xaxes(title="")
                fig.update_yaxes(title="Volume")
                st.plotly_chart(fig, use_container_width=True)

            section_header("Production Tasks (Monthly) — Primary", "Task-level monthly totals and deltas vs prior month.")
            cur_primary = build_task_kpi_table_with_hours(cur_df[(cur_df["task_type_norm"] == "production") & (cur_df["role_type_norm"] == "primary")])
            prev_primary = build_task_kpi_table_with_hours(prev_df[(prev_df["task_type_norm"] == "production") & (prev_df["role_type_norm"] == "primary")]) if not prev_df.empty else pd.DataFrame()

            if cur_primary.empty:
                st.info("No primary production tasks for this month.")
            else:
                primary_tbl = add_deltas(cur_primary, prev_primary)
                st.dataframe(style_delta_df(primary_tbl, ["Δ Hours", "Δ Volume"]), use_container_width=True, hide_index=True)

            section_header("Production Tasks (Monthly) — Backup", "Backup production totals and deltas vs prior month.")
            cur_backup = build_task_kpi_table_with_hours(cur_df[(cur_df["task_type_norm"] == "production") & (cur_df["role_type_norm"] == "backup")])
            prev_backup = build_task_kpi_table_with_hours(prev_df[(prev_df["task_type_norm"] == "production") & (prev_df["role_type_norm"] == "backup")]) if not prev_df.empty else pd.DataFrame()

            if cur_backup.empty:
                st.info("No backup production tasks for this month.")
            else:
                backup_tbl = add_deltas(cur_backup, prev_backup)
                st.dataframe(style_delta_df(backup_tbl, ["Δ Hours", "Δ Volume"]), use_container_width=True, hide_index=True)

            section_header("Coverage Tasks (Monthly)", "Coverage task totals based on allocated remaining-day time.")
            cur_cov = build_task_kpi_table_with_hours(cur_df[cur_df["task_type_norm"] == "coverage"])
            prev_cov = build_task_kpi_table_with_hours(prev_df[prev_df["task_type_norm"] == "coverage"]) if not prev_df.empty else pd.DataFrame()

            if cur_cov.empty:
                st.info("No coverage tasks for this month.")
            else:
                cov_tbl = add_deltas(cur_cov, prev_cov)
                st.dataframe(style_delta_df(cov_tbl, ["Δ Hours", "Δ Volume"]), use_container_width=True, hide_index=True)

# =================================
# Quarterly
# =================================
with tabs[3]:
    section_header(
        "Quarterly View",
        "Quarterly rollups built from weekly data using Week Ending."
    )

    members = list_team_members()
    if not members:
        st.info("No data loaded yet. Upload data in Admin Upload.")
    else:
        sel_member = st.selectbox("Team Member", members, index=0, key="quarterly_member")
        all_df = prep_for_rollups(fetch_all_tasks())
        member_df = all_df[all_df["team_member"] == sel_member].copy()

        if member_df.empty:
            st.info("No data for this team member.")
        else:
            quarters = sorted(member_df["quarter"].dropna().unique().tolist())
            sel_quarter = st.selectbox("Quarter", quarters, index=len(quarters) - 1)

            cur_df = member_df[member_df["quarter"] == sel_quarter].copy()

            prev_quarter = None
            try:
                prev_quarter = str(pd.Period(sel_quarter, freq="Q") - 1)
            except Exception:
                prev_quarter = None

            prev_df = member_df[member_df["quarter"] == prev_quarter].copy() if prev_quarter in quarters else pd.DataFrame()

            cur_totals = compute_top_metrics(cur_df)
            prev_totals = compute_top_metrics(prev_df) if not prev_df.empty else None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Production (Primary) Hours",
                fmt_hours(seconds_to_hours(cur_totals["prod_primary_seconds"])),
                period_delta_str(seconds_to_hours(cur_totals["prod_primary_seconds"]),
                                 seconds_to_hours(prev_totals["prod_primary_seconds"]) if prev_totals else None),
            )
            c2.metric(
                "Production (Backup) Hours",
                fmt_hours(seconds_to_hours(cur_totals["prod_backup_seconds"])),
                period_delta_str(seconds_to_hours(cur_totals["prod_backup_seconds"]),
                                 seconds_to_hours(prev_totals["prod_backup_seconds"]) if prev_totals else None),
            )
            c3.metric(
                "Coverage Hours",
                fmt_hours(seconds_to_hours(cur_totals["coverage_seconds"])),
                period_delta_str(seconds_to_hours(cur_totals["coverage_seconds"]),
                                 seconds_to_hours(prev_totals["coverage_seconds"]) if prev_totals else None),
            )
            c4.metric(
                "Coverage Volume",
                fmt_int(cur_totals["coverage_volume"]),
                period_delta_str(cur_totals["coverage_volume"],
                                 prev_totals["coverage_volume"] if prev_totals else None,
                                 is_int=True),
            )

            st.markdown("<br>", unsafe_allow_html=True)

            q_chart = (
                member_df.groupby("quarter", as_index=False)["volume"]
                .sum()
                .sort_values("quarter")
            )
            if not q_chart.empty:
                fig = px.bar(
                    q_chart,
                    x="quarter",
                    y="volume",
                    text_auto=True,
                    title="Total Volume by Quarter",
                    color_discrete_sequence=[ACCENT],
                )
                fig = apply_layout(fig, height=320, show_legend=False)
                fig.update_xaxes(title="")
                fig.update_yaxes(title="Volume")
                st.plotly_chart(fig, use_container_width=True)

            section_header("Production Tasks (Quarterly) — Primary", "Task-level quarterly totals and deltas vs prior quarter.")
            cur_primary = build_task_kpi_table_with_hours(cur_df[(cur_df["task_type_norm"] == "production") & (cur_df["role_type_norm"] == "primary")])
            prev_primary = build_task_kpi_table_with_hours(prev_df[(prev_df["task_type_norm"] == "production") & (prev_df["role_type_norm"] == "primary")]) if not prev_df.empty else pd.DataFrame()

            if cur_primary.empty:
                st.info("No primary production tasks for this quarter.")
            else:
                primary_tbl = add_deltas(cur_primary, prev_primary)
                st.dataframe(style_delta_df(primary_tbl, ["Δ Hours", "Δ Volume"]), use_container_width=True, hide_index=True)

            section_header("Production Tasks (Quarterly) — Backup", "Backup production totals and deltas vs prior quarter.")
            cur_backup = build_task_kpi_table_with_hours(cur_df[(cur_df["task_type_norm"] == "production") & (cur_df["role_type_norm"] == "backup")])
            prev_backup = build_task_kpi_table_with_hours(prev_df[(prev_df["task_type_norm"] == "production") & (prev_df["role_type_norm"] == "backup")]) if not prev_df.empty else pd.DataFrame()

            if cur_backup.empty:
                st.info("No backup production tasks for this quarter.")
            else:
                backup_tbl = add_deltas(cur_backup, prev_backup)
                st.dataframe(style_delta_df(backup_tbl, ["Δ Hours", "Δ Volume"]), use_container_width=True, hide_index=True)

            section_header("Coverage Tasks (Quarterly)", "Coverage totals based on allocated remaining-day time.")
            cur_cov = build_task_kpi_table_with_hours(cur_df[cur_df["task_type_norm"] == "coverage"])
            prev_cov = build_task_kpi_table_with_hours(prev_df[prev_df["task_type_norm"] == "coverage"]) if not prev_df.empty else pd.DataFrame()

            if cur_cov.empty:
                st.info("No coverage tasks for this quarter.")
            else:
                cov_tbl = add_deltas(cur_cov, prev_cov)
                st.dataframe(style_delta_df(cov_tbl, ["Δ Hours", "Δ Volume"]), use_container_width=True, hide_index=True)

# =================================
# Admin Upload
# =================================
with tabs[4]:
    section_header(
        "Admin Upload",
        "Replace all data, selected month data, or selected week data from uploaded files."
    )

    current_rows = db_row_count()
    st.metric("Current Rows in Database", fmt_int(current_rows))

    upload_mode = st.selectbox(
        "Upload Mode",
        ["Full Replace", "Replace Selected Month(s)", "Replace Selected Week(s)"],
        index=0,
    )

    uploaded_files = st.file_uploader(
        "Upload clean workflow tracker files",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        all_frames = []
        file_summaries = []

        for f in uploaded_files:
            try:
                month_df = load_month_file(f)
                if not month_df.empty:
                    all_frames.append(month_df)
                    file_summaries.append({"File": f.name, "Rows Loaded": len(month_df)})
                else:
                    file_summaries.append({"File": f.name, "Rows Loaded": 0})
            except Exception as e:
                file_summaries.append({"File": f.name, "Rows Loaded": f"ERROR: {e}"})

        st.dataframe(pd.DataFrame(file_summaries), use_container_width=True, hide_index=True)

        if all_frames:
            preview_raw = pd.concat(all_frames, ignore_index=True)
            preview_final = apply_coverage_logic(preview_raw)

            st.markdown("### Preview")
            st.dataframe(preview_final.head(20), use_container_width=True, hide_index=True)

            if upload_mode == "Replace Selected Month(s)":
                preview_months = (
                    pd.to_datetime(preview_final["week_ending"], errors="coerce")
                    .dt.to_period("M")
                    .astype(str)
                    .dropna()
                    .unique()
                    .tolist()
                )
                st.info(f"Months to replace: {', '.join(sorted(preview_months))}")

            if upload_mode == "Replace Selected Week(s)":
                preview_weeks = sorted(preview_final["week_ending"].dropna().astype(str).unique().tolist())
                st.info(f"Weeks to replace: {', '.join(preview_weeks)}")

            if st.button("Run Upload"):
                if upload_mode == "Full Replace":
                    replace_all_task_data(preview_final)
                    st.success(f"Full replace complete. Database now contains {len(preview_final):,} task rows.")
                elif upload_mode == "Replace Selected Month(s)":
                    replace_selected_months(preview_final)
                    st.success("Selected month data replaced successfully.")
                elif upload_mode == "Replace Selected Week(s)":
                    replace_selected_weeks(preview_final)
                    st.success("Selected week data replaced successfully.")
        else:
            st.warning("No usable task data was found in the uploaded files.")
