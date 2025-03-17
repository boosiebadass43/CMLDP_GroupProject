#!/usr/bin/env python
"""
Run script to start the Small Business Federal Contracting Dashboard.
This script will check if NLTK data is available and download it if needed,
then launch the Streamlit app.
"""

import os
import sys
import subprocess
import nltk

def setup_nltk():
    """Check and download necessary NLTK data if needed"""
    print("Checking NLTK resources...")
    
    # Create a directory for NLTK data
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add to NLTK path
    nltk.data.path.append(nltk_data_dir)
    
    # Check and download required resources
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
            print(f"✅ Resource '{resource}' is already available")
        except LookupError:
            print(f"📥 Downloading missing resource: '{resource}'...")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)

def run_app():
    """Start the Streamlit app"""
    print("\n🚀 Starting the Small Business Federal Contracting Dashboard...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    # Check for and install NLTK data if needed
    setup_nltk()
    
    # Run the Streamlit app
    run_app()