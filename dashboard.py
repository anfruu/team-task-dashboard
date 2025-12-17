
import streamlit as st
import pandas as pd
import re
from pathlib import Path
from PIL import Image

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
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Header: logo + title using Streamlit columns (robust, no cropping)
# ------------------------------------------------------------
HERE = Path(__file__).parent
LOGO = HERE / "static" / "lpl-logo-blue.png"

col_logo, col_title = st.columns([1, 6], gap="small")
with col_logo:
    if LOGO.exists():
        try:
            st.image(str(LOGO), width=140)  # fixed width keeps full logo visible
        except Exception:
            st.empty()
    else:
        st.empty()

with col_title:
    st.title("MAO Task Tracker Dashboard")
    st.caption("LPL Financial — Operations")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
REQUIRED_COLS = [
    "Task ID", "Task Description", "Task Type",
    "Team Member", "Day", "Duration", "Volume",
]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

def escape_md(text: object) -> str:
    """Escape Markdown special characters for safe rendering in task list."""
    return re.sub(r'([*_`])', r'\\\1', str(text)) if pd.notna(text) else ""

def has_required_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Return (ok, missing_columns)."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return (len(missing) == 0, missing)

# ------------------------------------------------------------
# Upload & validation
# ------------------------------------------------------------
uploaded = st.file_uploader("Upload Combined Excel File", type=["xlsx"])

if uploaded is not None:
    try:
        # Read Excel
        df = pd.read_excel(uploaded, engine="openpyxl")

        ok, missing = has_required_columns(df)
        if not ok:
            st.error("Missing required columns: " + ", ".join(missing))
            st.stop()

        # Normalize fields
        df["Task Type"] = df["Task Type"].astype(str).str.strip().str.lower()
        df["Day"] = df["Day"].astype(str).str.strip().str.title()

        # Team Member selector
        members = sorted([m for m in df["Team Member"].dropna().unique()])
        if len(members) == 0:
            st.warning("No team members found.")
            st.stop()

        selected = st.selectbox("Select Team Member", members)
        mdf = df[df["Team Member"] == selected]

        # ----------------------------------------------------
        # Task Summary (unique tasks)
        # ----------------------------------------------------
        st.markdown("### ✅ Task Summary for the Week")
        unique_tasks = mdf.drop_duplicates(subset=["Task ID"])
        if unique_tasks.empty:
            st.info("No tasks for this team member.")
        else:
            for _, row in unique_tasks[["Task ID", "Task Description"]].iterrows():
                tid = escape_md(row["Task ID"])
                desc = (
                    escape_md(row["Task Description"])
                    if pd.notna(row["Task Description"]) else "(No Description)"
                )
                st.markdown(f"- **{tid}**: {desc}")

        # ----------------------------------------------------
        # Raw Data (Production only)
        # ----------------------------------------------------
        st.markdown("### 📋 Raw Data (Production Tasks Only)")
        prod = mdf[mdf["Task Type"] == "production"]
        if not prod.empty:
            display = prod[["Task ID", "Task Type", "Day", "Duration", "Volume"]].copy()
            display["Task Type"] = display["Task Type"].str.capitalize()
            st.dataframe(display, use_container_width=True)

            st.download_button(
                label="Download Filtered Data (Production)",
                data=display.to_csv(index=False),
                file_name=f"{selected}_ProductionTasks.csv",
                mime="text/csv"
            )
        else:
            st.write("No production tasks.")

        # ----------------------------------------------------
        # Coverage Summary (unique IDs)
        # ----------------------------------------------------
        st.markdown("### 🛡 Coverage Tasks")
        coverage = (
            mdf[mdf["Task Type"] == "coverage"]
            .drop_duplicates(subset=["Task ID"])[["Task ID"]]
        )
        if not coverage.empty:
            st.dataframe(coverage, use_container_width=True)
            st.download_button(
                label="Download Coverage Task IDs",
                data=coverage.to_csv(index=False),
                file_name=f"{selected}_CoverageTasks.csv",
                mime="text/csv"
            )
        else:
            st.write("No coverage tasks.")

        # ----------------------------------------------------
        # Production Summary (by day)
        # ----------------------------------------------------
        st.markdown("### ⚙ Production Summary")
        if not prod.empty:
            summary = (
                prod.groupby("Day")
                .agg(
                    Production_Tasks=("Task ID", "count"),
                    Total_Duration=("Duration", "sum")
                )
                .reset_index()
            )
            summary["Day"] = pd.Categorical(
                summary["Day"], categories=DAY_ORDER, ordered=True
            )
            summary = summary.sort_values("Day")
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No production summary available for this member.")

    except Exception as e:
        st.error(f"Could not read the Excel file: {e}")
else:
    st.info("Please upload a combined Excel file to see the dashboard.")
