
import streamlit as st
import pandas as pd
import re

# Escape markdown special characters
def escape_markdown(text):
    return re.sub(r'([*_`])', r'\\\1', str(text))

# Wide layout
st.set_page_config(layout="wide")

st.title("📊 Team Task Tracker Dashboard")

uploaded_file = st.file_uploader("Upload Combined Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Normalize Task Type for filtering
    df['Task Type'] = df['Task Type'].str.strip().str.lower()

    # Dropdown for Team Member
    team_members = df['Team Member'].dropna().unique()
    selected_member = st.selectbox("Select Team Member", sorted(team_members))

    member_data = df[df['Team Member'] == selected_member]

    # --- Task Summary (unique tasks only, escape markdown) ---
    st.markdown("### ✅ Task Summary for the Week")
    unique_tasks = member_data.drop_duplicates(subset=['Task ID'])
    for _, row in unique_tasks[['Task ID', 'Task Description']].iterrows():
        task_id = escape_markdown(row['Task ID'])
        task_desc = escape_markdown(row['Task Description']) if pd.notna(row['Task Description']) else "(No Description)"
        st.markdown(f"- **{task_id}**: {task_desc}")

    # --- Raw Data Table (Production only, capitalize Task Type) ---
    st.markdown("### 📋 Raw Data (Production Tasks Only)")
    production_raw = member_data[member_data['Task Type'] == 'production']
    display_data = production_raw[['Task ID', 'Task Type', 'Day', 'Duration', 'Volume']].copy()
    display_data['Task Type'] = display_data['Task Type'].str.capitalize()
    st.dataframe(display_data, use_container_width=True)

    # Download button for filtered data
    st.download_button(
        label="Download Filtered Data",
        data=display_data.to_csv(index=False),
        file_name=f"{selected_member}_ProductionTasks.csv",
        mime="text/csv"
    )

    # --- Coverage Summary (unique tasks only) ---
    st.markdown("### 🛡 Coverage Tasks")
    coverage_data = member_data[member_data['Task Type'] == 'coverage'].drop_duplicates(subset=['Task ID'])[['Task ID']]
    if not coverage_data.empty:
        st.dataframe(coverage_data, use_container_width=True)
    else:
        st.write("No coverage tasks.")

    # --- Production Summary (sorted by day order) ---
    st.markdown("### ⚙ Production Summary")
    if not production_raw.empty:
        summary = production_raw.groupby('Day').agg({'Task ID': 'count', 'Duration': 'sum'}).reset_index()
        summary.columns = ['Day', 'Production Tasks', 'Total Duration']

        # Sort days in correct order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        summary['Day'] = pd.Categorical(summary['Day'], categories=day_order, ordered=True)
        summary = summary.sort_values('Day')

        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Please upload a file to see the dashboard.")
