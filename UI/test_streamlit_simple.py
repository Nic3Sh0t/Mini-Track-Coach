#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Streamlit Test - Verify if Streamlit can start normally
"""

import streamlit as st

st.set_page_config(
    page_title="Streamlit Test",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Streamlit Connection Test")
st.success("If you can see this page, the Streamlit server is working correctly!")

st.info("""
This simple test page is used to verify:
- Whether Streamlit server starts normally
- Whether browser can successfully connect to the server
- Whether port 8501 is working normally
""")

st.markdown("---")
st.markdown("**If this page displays normally, the problem may be in the app.py code.**")
