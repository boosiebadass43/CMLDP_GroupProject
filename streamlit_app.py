# Simple Streamlit app entry point

import streamlit as st
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

if __name__ == "__main__":
    # Import main function from embedded_app
    try:
        from embedded_app import main
        main()
    except Exception as e:
        st.error(f"Error starting the application: {str(e)}")
        st.code(f"Error details:\n{type(e).__name__}: {str(e)}")