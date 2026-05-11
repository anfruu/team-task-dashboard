# dashboard.py
# MAO Workflow Tracker Dashboard - Redesigned
# Executive-ready | LPL Financial Operations

import os
from io import BytesIO

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(
   page_title="MAO Workflow Tracker",
   layout="wide",
   initial_sidebar_state="collapsed"
)

# ---------------------------------
# Design Tokens
# ---------------------------------
TEXT_COLOR = "#0F1F2E"
SUBTEXT_COLOR = "#4A5C6A"
BORDER = "#DDE6EE"
CARD_BG = "#F7FAFC"
PAGE_BG = "#F2F6FA"
SECTION_BG = "#FFFFFF"

PRIMARY = "#2A5F8F"
SECONDARY = "#3A8A8B"
ACCENT = "#6B5FA6"
PRODUCTION_COLOR = "#2A5F8F"
COVERAGE_COLOR = "#3A8A8B"
WARM = "#C07A3A"

# ---------------------------------
# Global Styles
# ---------------------------------
st.markdown(f"""
<style>
   .stApp {{
       background: {PAGE_BG};
   }}
   .block-container {{
       padding-top: 1.5rem;
       padding-bottom: 2rem;
       max-width: 1480px;
   }}
   html, body, [class*="css"] {{
       color: {TEXT_COLOR} !important;
       font-family: "Segoe UI", "Inter", sans-serif;
   }}
   h1 {{
       color: {TEXT_COLOR} !important;
       font-weight: 800 !important;
       font-size: 1.8rem !important;
       letter-spacing: -0.03em;
       margin-bottom: 0.1rem !important;
   }}
   h2, h3, h4 {{
       color: {TEXT_COLOR} !important;
       font-weight: 700 !important;
   }}
   p, label, .stCaption {{
       color: {SUBTEXT_COLOR} !important;
   }}
   .stTabs [data-baseweb="tab-list"] {{
       gap: 8px;
       border-bottom: 2px solid {BORDER};
       padding-bottom: 0;
   }}
   .stTabs [data-baseweb="tab"] {{
       height: 42px;
       background: transparent;
       border: none;
       border-radius: 0;
       padding: 0 20px;
       color: {SUBTEXT_COLOR} !important;
       font-weight: 600;
       font-size: 0.95rem;
       border-bottom: 3px solid transparent;
   }}
   .stTabs [aria-selected="true"] {{
       background: transparent !important;
       border-bottom: 3px solid {PRIMARY} !important;
       color: {PRIMARY} !important;
   }}
   div[data-testid="stMetric"] {{
       background: {SECTION_BG};
       border: 1px solid {BORDER};
       border-radius: 14px;
       padding: 18px 20px;
       box-shadow: 0 2px 8px rgba(15,31,46,0.05);
   }}
   div[data-testid="stMetricLabel"] {{
       color: {SUBTEXT_COLOR} !important;
       font-weight: 600 !important;
       font-size: 0.85rem !important;
       text-transform: uppercase;
       letter-spacing: 0.04em;
   }}
   div[data-testid="stMetricValue"] {{
       color: {TEXT_COLOR} !important;
       font-weight: 800 !important;
       font-size: 1.8rem !important;
   }}
   div[data-testid="stDataFrame"] {{
       border: 1px solid {BORDER};
       border-radius: 12px;
       overflow: hidden;
       box-shadow: 0 1px 4px rgba(15,31,46,0.04);
   }}
   .card {{
       background: {SECTION_BG};
       border: 1px solid {BORDER};
       border-radius: 16px;
       padding: 20px 22px;
       margin-bottom: 1rem;
       box-shadow: 0 2px 10px rgba(15,31,46,0.04);
   }}
   .card-title {{
       color: {TEXT_COLOR};
       font-weight: 700;
       font-size: 1rem;
       margin-bottom: 2px;
   }}
   .card-sub {{
       color: {SUBTEXT_COLOR};
       font-size: 0.85rem;
       margin-bottom: 12px;
   }}
   .pill {{
       display: inline-block;
       padding: 3px 12px;
       border-radius: 20px;
       font-size: 0.78rem;
       font-weight: 700;
       margin-right: 6px;
   }}
   .pill-prod {{
       background: #E8F0F8;
       color: {PRIMARY};
   }}
   .pill-cov {{
       background: #E6F4F4;
       color: {SECONDARY};
   }}
   .divider {{
       border: none;
       border-top: 1px solid {BORDER};
       margin: 1rem 0;
   }}
   .stSelectbox label {{
       color: {TEXT_COLOR} !important;
       font-weight: 600 !important;
       font-size: 0.9rem !important;
   }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Header
# ---------------------------------
st.markdown(f"""
<div style="margin-bottom: 1.2rem;">
   <h1>MAO Workflow Tracker Dashboard</h1>
   <p style="font-size:0.95rem; margin:0;">LPL Financial – Operations &nbsp;|&nbsp;
   <span class="pill pill-prod">Production</span>
   <span class="pill pill-cov">Coverage</span>
   </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Constants
# ---------------------------------
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_TO_OFFSET = {"Monday": 4, "Tuesday": 3, "Wednesday": 2,
                "Thursday": 1, "Friday": 0, "Saturday": -1, "Sunday": -2}
WORK_SECONDS = 28800

# ---------------------------------
# Helpers
# ---------------------------------
def seconds_to_hhmm(s):
   try:
       s = max(0, int(round(float(s))))
       return f"{s//3600:02d}:{(s%3600)//60:02d}"
   except:
       return "00:00"

def seconds_to_hours(s):
   try:
       return round(float(s) / 3600, 2)
   except:
       return 0.0

def fmt_hours(x):
   try:
       return f"{float(x):,.2f}"
   except:
       return "0.00"

def fmt_int(x):
   try:
       return f"{int(x):,}"
   except:
       return "0"

def clean_task_name(name):
   if pd.isna(name):
       return ""
   return str(name).strip().replace("*", "").replace("  ", " ").strip()

def apply_layout(fig, height=340, show_legend=True):
   fig.update_layout(
       height=height,
       margin=dict(l=16, r=16, t=48, b=16),
       plot_bgcolor="white",
       paper_bgcolor="white",
       font=dict(color=TEXT_COLOR, size=12),
       legend=dict(
           orientation="h", yanchor="bottom", y=1.02,
           xanchor="right", x=1,
           font=dict(color=TEXT_COLOR, size=11), title=None,
       ),
       showlegend=show_legend,
   )
   fig.update_xaxes(gridcolor="#EDF2F7", zeroline=False,
                    tickfont=dict(color=TEXT_COLOR))
   fig.update_yaxes(gridcolor="#EDF2F7", zeroline=False,
                    tickfont=dict(color=TEXT_COLOR))
   return fig

def card(title, subtitle=""):
   sub_html = f'<div class="card-sub">{subtitle}</div>' if subtitle else ""
   st.markdown(f"""
   <div class="card">
       <div class="card-title">{title}</div>
       {sub_html}
   </div>
   """, unsafe_allow_html=True)

def delta_str(curr, prev, is_int=False):
   if prev is None:
       return None
   try:
       d = (int(curr) - int(prev)) if is_int else round(float(curr) - float(prev), 2)
       return f"{d:+,}" if is_int else f"{d:+.2f}"
   except:
       return None

def color_delta(val):
   try:
       v = float(val)
       return "color: #2E7D32;" if v > 0 else ("color: #C62828;" if v < 0 else "")
   except:
       return ""

# ---------------------------------
# DB Setup
# ---------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
   st.error("DATABASE_URL is not set. Add it in Render Environment Variables.")
   st.stop()

ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
IS_SQLITE = ENGINE.dialect.name == "sqlite"

def init_db():
   with ENGINE.begin() as conn:
       conn.execute(text("""
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
               month TEXT,
               quarter TEXT,
               uploaded_at TEXT
           )
       """))
       if not IS_SQLITE:
           for col, typedef in [
               ("task_type", "TEXT"), ("role_type", "TEXT"),
               ("raw_duration_seconds", "INTEGER"),
               ("effective_duration_seconds", "INTEGER"),
               ("volume", "INTEGER"), ("day", "TEXT"),
               ("week_ending", "TEXT"), ("work_date", "TEXT"),
               ("month", "TEXT"), ("quarter", "TEXT"),
           ]:
               try:
                   conn.execute(text(f"ALTER TABLE tasks ADD COLUMN IF NOT EXISTS {col} {typedef}"))
               except:
                   pass
           for idx, cols in [
               ("idx_tasks_member_week", "team_member, week_ending"),
               ("idx_tasks_month", "month"),
               ("idx_tasks_quarter", "quarter"),
               ("idx_tasks_taskid", "task_id"),
           ]:
               try:
                   conn.execute(text(f"CREATE INDEX IF NOT EXISTS {idx} ON tasks({cols})"))
               except:
                   pass

init_db()

# ---------------------------------
# Normalization
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

def normalize_df(df):
   df = df.copy()
   df.columns = [str(c).strip() for c in df.columns]
   missing = [s for s in EXPECTED_MAP if s not in df.columns]
   if missing:
       return pd.DataFrame()

   out = pd.DataFrame()
   for src, dst in EXPECTED_MAP.items():
       out[dst] = df[src]

   # Clean task names — strip asterisks and extra spaces
   out["task_id"] = out["task_id"].apply(clean_task_name)
   out["team_member"] = out["team_member"].apply(lambda x: str(x).strip() if pd.notna(x) else "")
   out["task_type"] = out["task_type"].apply(lambda x: str(x).strip().title() if pd.notna(x) else "")
   out["role_type"] = out["role_type"].apply(lambda x: str(x).strip().title() if pd.notna(x) else "")
   out["day"] = out["day"].apply(lambda x: str(x).strip().title() if pd.notna(x) else "")

   out["raw_duration_seconds"] = pd.to_numeric(out["raw_duration_seconds"], errors="coerce").fillna(0).astype(int)
   out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype(int)
   out["week_ending"] = pd.to_datetime(out["week_ending"], errors="coerce")

   out = out.dropna(subset=["week_ending"])
   out = out[out["task_id"] != ""]
   out = out[out["team_member"] != ""]

   out["week_ending"] = out["week_ending"].dt.strftime("%Y-%m-%d")
   return out

def apply_coverage_logic(df):
   if df is None or df.empty:
       return df

   df = df.copy()
   df["week_dt"] = pd.to_datetime(df["week_ending"], errors="coerce")
   day_clean = df["day"].astype(str).str.strip()
   offsets = day_clean.map(DAY_TO_OFFSET)
   df["work_date"] = (df["week_dt"] - pd.to_timedelta(offsets, unit="D")).dt.strftime("%Y-%m-%d")
   df["month"] = pd.to_datetime(df["week_ending"]).dt.strftime("%B %Y")
   df["quarter"] = pd.to_datetime(df["week_ending"]).dt.to_period("Q").astype(str)

   prod_mask = df["task_type"].str.lower() == "production"
   cov_mask = df["task_type"].str.lower() == "coverage"

   prod_day = df[prod_mask].groupby(["team_member", "work_date"], as_index=False)["raw_duration_seconds"].sum()
   prod_day.columns = ["team_member", "work_date", "prod_seconds"]

   cov_day = df[cov_mask].groupby(["team_member", "work_date"], as_index=False).agg(
       cov_task_count=("task_id", "count")
   )

   daily = prod_day.merge(cov_day, on=["team_member", "work_date"], how="outer").fillna(0)
   daily["leftover"] = (WORK_SECONDS - daily["prod_seconds"]).clip(lower=0)
   daily["secs_per_cov_task"] = daily.apply(
       lambda r: r["leftover"] / r["cov_task_count"] if r["cov_task_count"] > 0 else 0, axis=1)

   df = df.merge(daily[["team_member", "work_date", "secs_per_cov_task"]],
                 on=["team_member", "work_date"], how="left")
   df["secs_per_cov_task"] = df["secs_per_cov_task"].fillna(0)

   df["effective_duration_seconds"] = df["raw_duration_seconds"]
   df.loc[cov_mask, "effective_duration_seconds"] = df.loc[cov_mask, "secs_per_cov_task"].round().astype(int)

   return df.drop(columns=["week_dt", "secs_per_cov_task"], errors="ignore")

# ---------------------------------
# DB Operations
# ---------------------------------
def db_count():
   with ENGINE.begin() as conn:
       return int(conn.execute(text("SELECT COUNT(*) FROM tasks")).scalar() or 0)

def fetch_all():
   with ENGINE.begin() as conn:
       rows = conn.execute(text("""
           SELECT task_id, team_member, task_type, role_type,
                  raw_duration_seconds, effective_duration_seconds,
                  volume, day, week_ending, work_date, month, quarter
           FROM tasks
       """)).fetchall()
   return pd.DataFrame(rows, columns=[
       "task_id", "team_member", "task_type", "role_type",
       "raw_duration_seconds", "effective_duration_seconds",
       "volume", "day", "week_ending", "work_date", "month", "quarter"
   ])

def upsert_data(df, mode):
   months = pd.to_datetime(df["week_ending"], errors="coerce").dt.to_period("M").astype(str).dropna().unique().tolist()
   weeks = df["week_ending"].dropna().astype(str).unique().tolist()

   with ENGINE.begin() as conn:
       if mode == "Full Replace":
           conn.execute(text("DELETE FROM tasks"))
       elif mode == "Replace Month(s)":
           for m in months:
               if IS_SQLITE:
                   conn.execute(text("DELETE FROM tasks WHERE substr(week_ending,1,7) = :m"), {"m": m})
               else:
                   conn.execute(text("DELETE FROM tasks WHERE TO_CHAR(TO_DATE(week_ending,'YYYY-MM-DD'),'YYYY-MM') = :m"), {"m": m})
       elif mode == "Replace Week(s)":
           for w in weeks:
               conn.execute(text("DELETE FROM tasks WHERE week_ending = :w"), {"w": w})

       for _, r in df.iterrows():
           conn.execute(text("""
               INSERT INTO tasks (
                   task_id, team_member, task_type, role_type,
                   raw_duration_seconds, effective_duration_seconds,
                   volume, day, week_ending, work_date, month, quarter, uploaded_at
               ) VALUES (
                   :task_id, :team_member, :task_type, :role_type,
                   :raw_duration_seconds, :effective_duration_seconds,
                   :volume, :day, :week_ending, :work_date, :month, :quarter, :uploaded_at
               )
           """), {
               "task_id": r["task_id"],
               "team_member": r["team_member"],
               "task_type": r["task_type"],
               "role_type": r["role_type"],
               "raw_duration_seconds": int(r["raw_duration_seconds"]),
               "effective_duration_seconds": int(r["effective_duration_seconds"]),
               "volume": int(r["volume"]),
               "day": r["day"],
               "week_ending": r["week_ending"],
               "work_date": r.get("work_date", ""),
               "month": r.get("month", ""),
               "quarter": r.get("quarter", ""),
               "uploaded_at": pd.Timestamp.now().isoformat(timespec="seconds"),
           })

def load_upload(file):
   data = BytesIO(file.read())
   xls = pd.ExcelFile(data)
   frames = []
   for sheet in xls.sheet_names:
       if sheet.strip().lower() in ["unmatched task ids"]:
           continue
       try:
           raw = pd.read_excel(xls, sheet_name=sheet)
           norm = normalize_df(raw)
           if not norm.empty:
               frames.append(norm)
       except:
           continue
   if not frames:
       return pd.DataFrame()
   return pd.concat(frames, ignore_index=True)

# ---------------------------------
# Data Prep
# ---------------------------------
def prep(df):
   if df is None or df.empty:
       return pd.DataFrame()
   out = df.copy()
   out["task_type_n"] = out["task_type"].str.strip().str.lower()
   out["role_type_n"] = out["role_type"].str.strip().str.lower()
   out["week_dt"] = pd.to_datetime(out["week_ending"], errors="coerce")
   out["work_dt"] = pd.to_datetime(out["work_date"], errors="coerce")
   out["month"] = out["month"].fillna(out["week_dt"].dt.strftime("%B %Y"))
   out["quarter"] = out["quarter"].fillna(out["week_dt"].dt.to_period("Q").astype(str))
   out["month_sort"] = out["week_dt"].dt.to_period("M").astype(str)
   return out

def top_metrics(df):
   empty = {"prod_primary": 0, "prod_backup": 0, "coverage": 0,
            "prod_volume": 0, "cov_volume": 0, "total_hours": 0}
   if df is None or df.empty:
       return empty
   pp = df[(df["task_type_n"] == "production") & (df["role_type_n"] == "primary")]
   pb = df[(df["task_type_n"] == "production") & (df["role_type_n"] == "backup")]
   cv = df[df["task_type_n"] == "coverage"]
   pa = df[df["task_type_n"] == "production"]
   total = int(pp["effective_duration_seconds"].sum()) + int(pb["effective_duration_seconds"].sum()) + int(cv["effective_duration_seconds"].sum())
   return {
       "prod_primary": int(pp["effective_duration_seconds"].sum()),
       "prod_backup": int(pb["effective_duration_seconds"].sum()),
       "coverage": int(cv["effective_duration_seconds"].sum()),
       "prod_volume": int(pa["volume"].sum()),
       "cov_volume": int(cv["volume"].sum()),
       "total_hours": seconds_to_hours(total),
   }

# ---------------------------------
# Chart Builders
# ---------------------------------
def bar_chart(df, x, y, title, color=PRIMARY, height=300):
   if df is None or df.empty:
       return None
   fig = px.bar(df, x=x, y=y, text_auto=True, title=title,
                color_discrete_sequence=[color])
   fig = apply_layout(fig, height=height, show_legend=False)
   fig.update_xaxes(title="")
   fig.update_yaxes(title="")
   fig.update_traces(marker_line_width=0, textfont_size=11)
   return fig

def line_chart(df, x, y, color_col, title, height=320):
   if df is None or df.empty:
       return None
   fig = px.line(df, x=x, y=y, color=color_col, title=title,
                 markers=True,
                 color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT, WARM,
                                          "#B86A6A", "#60758A", "#5A9E6F"])
   fig = apply_layout(fig, height=height, show_legend=True)
   fig.update_xaxes(title="")
   fig.update_yaxes(title="")
   return fig

# ---------------------------------
# Table Builders
# ---------------------------------
def production_table(df):
   if df is None or df.empty:
       return pd.DataFrame()
   g = df[df["task_type_n"] == "production"].groupby("task_id", as_index=False).agg(
       Times_Performed=("effective_duration_seconds", "count"),
       Avg_Seconds=("effective_duration_seconds", "mean"),
       Total_Seconds=("effective_duration_seconds", "sum"),
       Volume=("volume", "sum"),
   )
   g["Avg Time"] = g["Avg_Seconds"].apply(seconds_to_hhmm)
   g["Total Time"] = g["Total_Seconds"].apply(seconds_to_hhmm)
   g["Hours"] = g["Total_Seconds"].apply(seconds_to_hours)
   return g.rename(columns={"task_id": "Task", "Times_Performed": "Times Performed",
                             "Volume": "Volume"})[
       ["Task", "Times Performed", "Avg Time", "Total Time", "Hours", "Volume"]
   ].sort_values("Hours", ascending=False)

def coverage_table(df):
   if df is None or df.empty:
       return pd.DataFrame()
   g = df[df["task_type_n"] == "coverage"].groupby("task_id", as_index=False).agg(
       Est_Seconds=("effective_duration_seconds", "sum"),
       Volume=("volume", "sum"),
   )
   g["Est Time"] = g["Est_Seconds"].apply(seconds_to_hhmm)
   g["Est Hours"] = g["Est_Seconds"].apply(seconds_to_hours)
   return g.rename(columns={"task_id": "Task", "Volume": "Volume"})[
       ["Task", "Est Time", "Est Hours", "Volume"]
   ].sort_values("Est Hours", ascending=False)

def person_prod_table(df):
   if df is None or df.empty:
       return pd.DataFrame()
   g = df[df["task_type_n"] == "production"].groupby(["team_member", "task_id"], as_index=False).agg(
       Times_Performed=("effective_duration_seconds", "count"),
       Avg_Seconds=("effective_duration_seconds", "mean"),
       Total_Seconds=("effective_duration_seconds", "sum"),
   )
   g["Avg Time"] = g["Avg_Seconds"].apply(seconds_to_hhmm)
   g["Total Time"] = g["Total_Seconds"].apply(seconds_to_hhmm)
   g["Hours"] = g["Total_Seconds"].apply(seconds_to_hours)
   return g.rename(columns={"team_member": "Person", "task_id": "Task",
                             "Times_Performed": "Times Performed"})[
       ["Person", "Task", "Times Performed", "Avg Time", "Total Time", "Hours"]
   ].sort_values(["Person", "Hours"], ascending=[True, False])

def person_cov_table(df):
   if df is None or df.empty:
       return pd.DataFrame()
   g = df[df["task_type_n"] == "coverage"].groupby(["team_member", "task_id"], as_index=False).agg(
       Est_Seconds=("effective_duration_seconds", "sum"),
       Volume=("volume", "sum"),
   )
   g["Est Time"] = g["Est_Seconds"].apply(seconds_to_hhmm)
   g["Est Hours"] = g["Est_Seconds"].apply(seconds_to_hours)
   return g.rename(columns={"team_member": "Person", "task_id": "Task", "Volume": "Volume"})[
       ["Person", "Task", "Est Time", "Est Hours", "Volume"]
   ].sort_values(["Person", "Est Hours"], ascending=[True, False])

# ---------------------------------
# Tabs
# ---------------------------------
tabs = st.tabs(["Overview", "By Person", "By Task", "Weekly", "Quarterly", "Admin Upload"])

# =================================
# TAB 1 — OVERVIEW
# =================================
with tabs[0]:
   all_df = prep(fetch_all())

   if all_df.empty:
       st.info("No data loaded yet. Go to Admin Upload to get started.")
   else:
       months = sorted(all_df["month_sort"].dropna().unique().tolist())
       month_labels = sorted(all_df["month"].dropna().unique().tolist(),
                             key=lambda m: pd.to_datetime(m, format="%B %Y", errors="coerce"))
       sel_idx = len(month_labels) - 1
       sel_month_label = st.selectbox("Month", month_labels, index=sel_idx, key="ov_month")

       cur = all_df[all_df["month"] == sel_month_label].copy()
       prev_label = None
       try:
           cur_period = pd.to_datetime(sel_month_label, format="%B %Y").to_period("M")
           prev_label = (cur_period - 1).strftime("%B %Y")
       except:
           pass
       prev = all_df[all_df["month"] == prev_label].copy() if prev_label in month_labels else pd.DataFrame()

       ct = top_metrics(cur)
       pt = top_metrics(prev) if not prev.empty else None

       # Metrics Row
       m1, m2, m3, m4, m5 = st.columns(5)
       m1.metric("Total Hours", fmt_hours(ct["total_hours"]),
                 delta_str(ct["total_hours"], pt["total_hours"] if pt else None))
       m2.metric("Production Hours",
                 fmt_hours(seconds_to_hours(ct["prod_primary"] + ct["prod_backup"])),
                 delta_str(seconds_to_hours(ct["prod_primary"] + ct["prod_backup"]),
                           seconds_to_hours(pt["prod_primary"] + pt["prod_backup"]) if pt else None))
       m3.metric("Coverage Hours", fmt_hours(seconds_to_hours(ct["coverage"])),
                 delta_str(seconds_to_hours(ct["coverage"]),
                           seconds_to_hours(pt["coverage"]) if pt else None))
       m4.metric("Production Volume", fmt_int(ct["prod_volume"]),
                 delta_str(ct["prod_volume"], pt["prod_volume"] if pt else None, is_int=True))
       m5.metric("Coverage Volume", fmt_int(ct["cov_volume"]),
                 delta_str(ct["cov_volume"], pt["cov_volume"] if pt else None, is_int=True))

       st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

       # Charts Row
       c1, c2 = st.columns(2)
       with c1:
           monthly_trend = all_df.groupby(["month_sort", "month"], as_index=False).agg(
               prod_hours=("effective_duration_seconds",
                           lambda s: seconds_to_hours(all_df.loc[s.index][all_df.loc[s.index, "task_type_n"] == "production"]["effective_duration_seconds"].sum())),
               cov_hours=("effective_duration_seconds",
                          lambda s: seconds_to_hours(all_df.loc[s.index][all_df.loc[s.index, "task_type_n"] == "coverage"]["effective_duration_seconds"].sum())),
           ).sort_values("month_sort")

           fig = go.Figure()
           fig.add_bar(x=monthly_trend["month"], y=monthly_trend["prod_hours"],
                       name="Production", marker_color=PRODUCTION_COLOR)
           fig.add_bar(x=monthly_trend["month"], y=monthly_trend["cov_hours"],
                       name="Coverage", marker_color=COVERAGE_COLOR)
           fig.update_layout(barmode="group", title="Hours by Month")
           fig = apply_layout(fig, height=300)
           st.plotly_chart(fig, use_container_width=True)

       with c2:
           member_summary = cur.groupby("team_member", as_index=False).agg(
               hours=("effective_duration_seconds",
                      lambda s: seconds_to_hours(s.sum()))
           ).sort_values("hours", ascending=True)
           fig2 = px.bar(member_summary, x="hours", y="team_member",
                         orientation="h", title="Hours by Team Member",
                         color_discrete_sequence=[PRIMARY])
           fig2 = apply_layout(fig2, height=300, show_legend=False)
           st.plotly_chart(fig2, use_container_width=True)

       st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

       # Production Table
       st.markdown('<div class="card-title">Production Tasks</div>', unsafe_allow_html=True)
       st.markdown('<div class="card-sub">Team-wide totals for the selected month</div>', unsafe_allow_html=True)
       pt_table = production_table(cur)
       if pt_table.empty:
           st.info("No production data.")
       else:
           st.dataframe(pt_table, use_container_width=True, hide_index=True)

       st.markdown("<br>", unsafe_allow_html=True)

       # Coverage Table
       st.markdown('<div class="card-title">Coverage Tasks</div>', unsafe_allow_html=True)
       st.markdown('<div class="card-sub">Estimated time based on remaining capacity after production</div>', unsafe_allow_html=True)
       cv_table = coverage_table(cur)
       if cv_table.empty:
           st.info("No coverage data.")
       else:
           st.dataframe(cv_table, use_container_width=True, hide_index=True)

# =================================
# TAB 2 — BY PERSON
# =================================
with tabs[1]:
   all_df = prep(fetch_all())

   if all_df.empty:
       st.info("No data loaded yet.")
   else:
       members = sorted(all_df["team_member"].dropna().unique().tolist())
       col1, col2, col3 = st.columns([2, 2, 1])
       with col1:
           sel_member = st.selectbox("Team Member", members, key="bp_member")
       with col2:
           month_labels = sorted(all_df["month"].dropna().unique().tolist(),
                                 key=lambda m: pd.to_datetime(m, format="%B %Y", errors="coerce"))
           sel_month = st.selectbox("Month", month_labels,
                                    index=len(month_labels) - 1, key="bp_month")
       with col3:
           view_type = st.selectbox("View", ["Monthly", "Quarterly"], key="bp_view")

       member_df = all_df[all_df["team_member"] == sel_member].copy()

       if view_type == "Quarterly":
           quarters = sorted(member_df["quarter"].dropna().unique().tolist())
           sel_q = st.selectbox("Quarter", quarters, index=len(quarters) - 1, key="bp_q")
           cur = member_df[member_df["quarter"] == sel_q].copy()
           prev_q = None
           try:
               prev_q = str(pd.Period(sel_q, freq="Q") - 1)
           except:
               pass
           prev = member_df[member_df["quarter"] == prev_q].copy() if prev_q in quarters else pd.DataFrame()
       else:
           cur = member_df[member_df["month"] == sel_month].copy()
           prev_label = None
           try:
               cur_period = pd.to_datetime(sel_month, format="%B %Y").to_period("M")
               prev_label = (cur_period - 1).strftime("%B %Y")
           except:
               pass
           prev = member_df[member_df["month"] == prev_label].copy() \
               if prev_label in month_labels else pd.DataFrame()

       ct = top_metrics(cur)
       pt = top_metrics(prev) if not prev.empty else None

       m1, m2, m3, m4 = st.columns(4)
       m1.metric("Production Hours",
                 fmt_hours(seconds_to_hours(ct["prod_primary"] + ct["prod_backup"])),
                 delta_str(seconds_to_hours(ct["prod_primary"] + ct["prod_backup"]),
                           seconds_to_hours(pt["prod_primary"] + pt["prod_backup"]) if pt else None))
       m2.metric("Coverage Hours", fmt_hours(seconds_to_hours(ct["coverage"])),
                 delta_str(seconds_to_hours(ct["coverage"]),
                           seconds_to_hours(pt["coverage"]) if pt else None))
       m3.metric("Production Volume", fmt_int(ct["prod_volume"]),
                 delta_str(ct["prod_volume"], pt["prod_volume"] if pt else None, is_int=True))
       m4.metric("Coverage Volume", fmt_int(ct["cov_volume"]),
                 delta_str(ct["cov_volume"], pt["cov_volume"] if pt else None, is_int=True))

       st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

       # Trend chart for person
       person_trend = member_df.groupby(["month_sort", "month"], as_index=False).agg(
           prod_hours=("effective_duration_seconds",
                       lambda s: seconds_to_hours(
                           member_df.loc[s.index][member_df.loc[s.index, "task_type_n"] == "production"]["effective_duration_seconds"].sum())),
           cov_hours=("effective_duration_seconds",
                      lambda s: seconds_to_hours(
                          member_df.loc[s.index][member_df.loc[s.index, "task_type_n"] == "coverage"]["effective_duration_seconds"].sum())),
       ).sort_values("month_sort")

       fig = go.Figure()
       fig.add_scatter(x=person_trend["month"], y=person_trend["prod_hours"],
                       name="Production", mode="lines+markers",
                       line=dict(color=PRODUCTION_COLOR, width=2))
       fig.add_scatter(x=person_trend["month"], y=person_trend["cov_hours"],
                       name="Coverage", mode="lines+markers",
                       line=dict(color=COVERAGE_COLOR, width=2))
       fig.update_layout(title=f"{sel_member} — Hours Trend")
       fig = apply_layout(fig, height=280)
       st.plotly_chart(fig, use_container_width=True)

       st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

       col_a, col_b = st.columns(2)
       with col_a:
           st.markdown('<div class="card-title">Production Tasks</div>', unsafe_allow_html=True)
           st.markdown(f'<div class="card-sub"><span class="pill pill-prod">Production</span></div>',
                       unsafe_allow_html=True)
           pt_tbl = production_table(cur)
           if pt_tbl.empty:
               st.info("No production tasks.")
           else:
               st.dataframe(pt_tbl, use_container_width=True, hide_index=True)

       with col_b:
           st.markdown('<div class="card-title">Coverage Tasks</div>', unsafe_allow_html=True)
           st.markdown(f'<div class="card-sub"><span class="pill pill-cov">Coverage</span></div>',
                       unsafe_allow_html=True)
           cv_tbl = coverage_table(cur)
           if cv_tbl.empty:
               st.info("No coverage tasks.")
           else:
               st.dataframe(cv_tbl, use_container_width=True, hide_index=True)

# =================================
# TAB 3 — BY TASK
# =================================
with tabs[2]:
   all_df = prep(fetch_all())

   if all_df.empty:
       st.info("No data loaded yet.")
   else:
       col1, col2 = st.columns([3, 1])
       with col1:
           all_tasks = sorted(all_df["task_id"].dropna().unique().tolist())
           sel_task = st.selectbox("Task", all_tasks, key="bt_task")
       with col2:
           task_type_filter = st.selectbox("Type", ["All", "Production", "Coverage"], key="bt_type")

       task_df = all_df[all_df["task_id"] == sel_task].copy()
       if task_type_filter != "All":
           task_df = task_df[task_df["task_type_n"] == task_type_filter.lower()]

       if task_df.empty:
           st.info("No data for this task.")
       else:
           ct = top_metrics(task_df)
           m1, m2, m3 = st.columns(3)
           m1.metric("Total Hours", fmt_hours(ct["total_hours"]))
           m2.metric("Total Volume", fmt_int(ct["prod_volume"] + ct["cov_volume"]))
           m3.metric("Times Performed", fmt_int(len(task_df)))

           st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

           # Trend by month
           task_trend = task_df.groupby(["month_sort", "month"], as_index=False).agg(
               avg_seconds=("effective_duration_seconds", "mean"),
               total_seconds=("effective_duration_seconds", "sum"),
               volume=("volume", "sum"),
               count=("task_id", "count"),
           ).sort_values("month_sort")
           task_trend["Avg Time (hrs)"] = task_trend["avg_seconds"].apply(seconds_to_hours)
           task_trend["Total Hours"] = task_trend["total_seconds"].apply(seconds_to_hours)

           c1, c2 = st.columns(2)
           with c1:
               fig = px.line(task_trend, x="month", y="Avg Time (hrs)",
                             title="Avg Time per Instance — Month over Month",
                             markers=True, color_discrete_sequence=[PRIMARY])
               fig = apply_layout(fig, height=280, show_legend=False)
               st.plotly_chart(fig, use_container_width=True)
           with c2:
               fig2 = px.bar(task_trend, x="month", y="volume",
                             title="Volume by Month",
                             color_discrete_sequence=[COVERAGE_COLOR])
               fig2 = apply_layout(fig2, height=280, show_legend=False)
               st.plotly_chart(fig2, use_container_width=True)

           st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

           # By person for this task
           st.markdown(f'<div class="card-title">{sel_task} — By Person</div>', unsafe_allow_html=True)
           st.markdown('<div class="card-sub">How each team member performed on this task</div>',
                       unsafe_allow_html=True)
           person_task = task_df.groupby("team_member", as_index=False).agg(
               Times_Performed=("effective_duration_seconds", "count"),
               Avg_Seconds=("effective_duration_seconds", "mean"),
               Total_Seconds=("effective_duration_seconds", "sum"),
               Volume=("volume", "sum"),
           )
           person_task["Avg Time"] = person_task["Avg_Seconds"].apply(seconds_to_hhmm)
           person_task["Total Time"] = person_task["Total_Seconds"].apply(seconds_to_hhmm)
           person_task["Hours"] = person_task["Total_Seconds"].apply(seconds_to_hours)
           person_task = person_task.rename(columns={
               "team_member": "Person", "Times_Performed": "Times Performed"
           })[["Person", "Times Performed", "Avg Time", "Total Time", "Hours", "Volume"]]
           st.dataframe(person_task.sort_values("Hours", ascending=False),
                        use_container_width=True, hide_index=True)

# =================================
# TAB 4 — WEEKLY
# =================================
with tabs[3]:
   all_df = prep(fetch_all())

   if all_df.empty:
       st.info("No data loaded yet.")
   else:
       members = sorted(all_df["team_member"].dropna().unique().tolist())
       col1, col2 = st.columns(2)
       with col1:
           sel_member = st.selectbox("Team Member", members, key="wk_member")
       with col2:
           member_weeks = sorted(
               all_df[all_df["team_member"] == sel_member]["week_ending"].dropna().unique().tolist(),
               reverse=True)
           sel_week = st.selectbox("Week Ending", member_weeks, key="wk_week")

       wk_df = all_df[(all_df["team_member"] == sel_member) &
                      (all_df["week_ending"] == sel_week)].copy()

       if wk_df.empty:
           st.info("No data for this selection.")
       else:
           ct = top_metrics(wk_df)
           m1, m2, m3, m4 = st.columns(4)
           m1.metric("Production Hours",
                     fmt_hours(seconds_to_hours(ct["prod_primary"] + ct["prod_backup"])))
           m2.metric("Coverage Hours", fmt_hours(seconds_to_hours(ct["coverage"])))
           m3.metric("Production Volume", fmt_int(ct["prod_volume"]))
           m4.metric("Coverage Volume", fmt_int(ct["cov_volume"]))

           st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

           col_a, col_b = st.columns(2)
           with col_a:
               st.markdown('<div class="card-title">Production</div>', unsafe_allow_html=True)
               prod = wk_df[wk_df["task_type_n"] == "production"].copy()
               if prod.empty:
                   st.info("No production entries.")
               else:
                   view = prod[["task_id", "day", "role_type", "volume",
                                "effective_duration_seconds"]].copy()
                   view["Duration"] = view["effective_duration_seconds"].apply(seconds_to_hhmm)
                   view["day_sort"] = view["day"].apply(
                       lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                   view = view.sort_values(["day_sort", "task_id"]).drop(
                       columns=["day_sort", "effective_duration_seconds"])
                   view = view.rename(columns={"task_id": "Task", "day": "Day",
                                               "role_type": "Role", "volume": "Volume"})
                   st.dataframe(view, use_container_width=True, hide_index=True)

           with col_b:
               st.markdown('<div class="card-title">Coverage</div>', unsafe_allow_html=True)
               cov = wk_df[wk_df["task_type_n"] == "coverage"].copy()
               if cov.empty:
                   st.info("No coverage entries.")
               else:
                   view = cov[["task_id", "day", "volume",
                               "effective_duration_seconds"]].copy()
                   view["Est Duration"] = view["effective_duration_seconds"].apply(seconds_to_hhmm)
                   view["day_sort"] = view["day"].apply(
                       lambda d: DAY_ORDER.index(d) if d in DAY_ORDER else 99)
                   view = view.sort_values(["day_sort", "task_id"]).drop(
                       columns=["day_sort", "effective_duration_seconds"])
                   view = view.rename(columns={"task_id": "Task", "day": "Day",
                                               "volume": "Volume"})
                   st.dataframe(view, use_container_width=True, hide_index=True)

# =================================
# TAB 5 — QUARTERLY
# =================================
with tabs[4]:
   all_df = prep(fetch_all())

   if all_df.empty:
       st.info("No data loaded yet.")
   else:
       quarters = sorted(all_df["quarter"].dropna().unique().tolist())
       col1, col2 = st.columns([2, 3])
       with col1:
           sel_q = st.selectbox("Quarter", quarters, index=len(quarters) - 1, key="q_sel")
       with col2:
           view_scope = st.selectbox("View", ["Team", "By Person"], key="q_scope")

       prev_q = None
       try:
           prev_q = str(pd.Period(sel_q, freq="Q") - 1)
       except:
           pass

       cur = all_df[all_df["quarter"] == sel_q].copy()
       prev = all_df[all_df["quarter"] == prev_q].copy() if prev_q in quarters else pd.DataFrame()

       ct = top_metrics(cur)
       pt = top_metrics(prev) if not prev.empty else None

       m1, m2, m3, m4 = st.columns(4)
       m1.metric("Production Hours",
                 fmt_hours(seconds_to_hours(ct["prod_primary"] + ct["prod_backup"])),
                 delta_str(seconds_to_hours(ct["prod_primary"] + ct["prod_backup"]),
                           seconds_to_hours(pt["prod_primary"] + pt["prod_backup"]) if pt else None))
       m2.metric("Coverage Hours", fmt_hours(seconds_to_hours(ct["coverage"])),
                 delta_str(seconds_to_hours(ct["coverage"]),
                           seconds_to_hours(pt["coverage"]) if pt else None))
       m3.metric("Production Volume", fmt_int(ct["prod_volume"]),
                 delta_str(ct["prod_volume"], pt["prod_volume"] if pt else None, is_int=True))
       m4.metric("Coverage Volume", fmt_int(ct["cov_volume"]),
                 delta_str(ct["cov_volume"], pt["cov_volume"] if pt else None, is_int=True))

       st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

       # Quarterly volume chart
       q_trend = all_df.groupby("quarter", as_index=False).agg(
           prod_hours=("effective_duration_seconds",
                       lambda s: seconds_to_hours(
                           all_df.loc[s.index][all_df.loc[s.index, "task_type_n"] == "production"]["effective_duration_seconds"].sum())),
           cov_hours=("effective_duration_seconds",
                      lambda s: seconds_to_hours(
                          all_df.loc[s.index][all_df.loc[s.index, "task_type_n"] == "coverage"]["effective_duration_seconds"].sum())),
       ).sort_values("quarter")

       fig = go.Figure()
       fig.add_bar(x=q_trend["quarter"], y=q_trend["prod_hours"],
                   name="Production", marker_color=PRODUCTION_COLOR)
       fig.add_bar(x=q_trend["quarter"], y=q_trend["cov_hours"],
                   name="Coverage", marker_color=COVERAGE_COLOR)
       fig.update_layout(barmode="group", title="Hours by Quarter")
       fig = apply_layout(fig, height=280)
       st.plotly_chart(fig, use_container_width=True)

       st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

       if view_scope == "Team":
           col_a, col_b = st.columns(2)
           with col_a:
               st.markdown('<div class="card-title">Production Tasks</div>', unsafe_allow_html=True)
               st.dataframe(production_table(cur), use_container_width=True, hide_index=True)
           with col_b:
               st.markdown('<div class="card-title">Coverage Tasks</div>', unsafe_allow_html=True)
               st.dataframe(coverage_table(cur), use_container_width=True, hide_index=True)
       else:
           col_a, col_b = st.columns(2)
           with col_a:
               st.markdown('<div class="card-title">Production by Person</div>', unsafe_allow_html=True)
               st.dataframe(person_prod_table(cur), use_container_width=True, hide_index=True)
           with col_b:
               st.markdown('<div class="card-title">Coverage by Person</div>', unsafe_allow_html=True)
               st.dataframe(person_cov_table(cur), use_container_width=True, hide_index=True)

# =================================
# TAB 6 — ADMIN UPLOAD
# =================================
with tabs[5]:
   st.markdown('<div class="card-title">Admin Upload</div>', unsafe_allow_html=True)
   st.markdown('<div class="card-sub">Upload the Workflow Q1 Summary Excel file to update the dashboard</div>',
               unsafe_allow_html=True)

   st.markdown("<br>", unsafe_allow_html=True)

   col1, col2 = st.columns([2, 1])
   with col1:
       uploaded = st.file_uploader("Upload Workflow Summary File", type=["xlsx"])
   with col2:
       mode = st.selectbox("Upload Mode", ["Full Replace", "Replace Month(s)", "Replace Week(s)"])

   st.metric("Current Rows in Database", fmt_int(db_count()))

   if uploaded:
       raw = load_upload(uploaded)

       if raw.empty:
           st.warning("No usable data found in the uploaded file.")
       else:
           final = apply_coverage_logic(raw)

           st.markdown("### Preview")
           st.dataframe(final.head(15), use_container_width=True, hide_index=True)

           months_found = sorted(pd.to_datetime(final["week_ending"], errors="coerce")
                                 .dt.strftime("%B %Y").dropna().unique().tolist())
           st.info(f"Data found for: {', '.join(months_found)} — {len(final):,} rows")

           if st.button("Upload to Dashboard", type="primary"):
               with st.spinner("Uploading..."):
                   upsert_data(final, mode)
               st.success(f"Done! {len(final):,} rows uploaded successfully.")
               st.rerun()
