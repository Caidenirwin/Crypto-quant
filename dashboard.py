"""
Streamlit entry point. Run with: streamlit run dashboard.py
"""

import runpy
import traceback
from pathlib import Path

import streamlit as st

try:
    # Execute the app fresh on every rerun (exactly once). The old import + reload
    # pattern executed it twice on first load, which duplicated every element.
    runpy.run_path(str(Path(__file__).with_name("dashboard_app.py")), run_name="__main__")
except Exception as exc:  # pragma: no cover
    st.markdown("## Dashboard Error")
    st.error("The dashboard failed to load. Traceback below.")
    st.code("".join(traceback.format_exception(exc)), language="python")
