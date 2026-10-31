"""
Streamlit entry point. Run with: streamlit run dashboard.py
"""

import importlib
import traceback

import streamlit as st

try:
    # Reload on every rerun so Streamlit doesn't serve a stale cached module.
    import dashboard_app as _dashboard_app
    importlib.reload(_dashboard_app)
except Exception as exc:  # pragma: no cover
    st.set_page_config(page_title="Dashboard Error", page_icon="!", layout="wide")
    st.markdown("## Dashboard Error")
    st.error("The dashboard failed to load. Traceback below.")
    st.code("".join(traceback.format_exception(exc)), language="python")
