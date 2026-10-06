import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="mlplat efficiency", page_icon=":material/monitoring:", layout="wide")
st.navigation([
    st.Page("pages/runs.py", title="Runs", icon=":material/list:", default=True),
    st.Page("pages/run_detail.py", title="Run detail", icon=":material/monitoring:", url_path="run"),
    st.Page("pages/compare.py", title="Compare", icon=":material/compare_arrows:", url_path="compare"),
]).run()
