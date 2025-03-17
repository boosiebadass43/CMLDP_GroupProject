# This is the main entry point for Streamlit Cloud
# Import and run the embedded_app version which has all dependencies packaged

import sys
import os
import streamlit as st

# If running in Streamlit Cloud, we need to ensure the current directory is in the path
# so Python can find our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Check if data directory exists and has survey data
data_path = os.path.join(current_dir, 'data', 'survey_data.csv')
if not os.path.exists(data_path):
    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(current_dir, 'data'), exist_ok=True)
    
    # Show error and instructions for uploading data
    st.error("Data file not found: data/survey_data.csv")
    st.info("Please check that your repository includes the data/survey_data.csv file.")
    
    # Add option to upload data
    uploaded_file = st.file_uploader("Upload survey_data.csv", type="csv")
    if uploaded_file is not None:
        # Save uploaded file
        with open(data_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Data file uploaded successfully. Restarting the app...")
        st.experimental_rerun()
    
    # Stop further execution
    st.stop()

# Import the main function from embedded_app
from embedded_app import main

# Execute the main function to run the dashboard
if __name__ == "__main__":
    main()