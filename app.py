"""
Root launcher for CardioSense Streamlit App.
"""

import os
import sys
import runpy

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "streamlit_app.py")
runpy.run_path(target, run_name="__main__")
