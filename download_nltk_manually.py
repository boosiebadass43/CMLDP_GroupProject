#!/usr/bin/env python3
"""
Script to manually download NLTK data without using the NLTK downloader
(useful when SSL certificate issues prevent automatic downloads)
"""
import os
import sys
import urllib.request
import zipfile
import shutil
import tempfile

def download_nltk_manually():
    """Download NLTK data packages manually from GitHub"""
    # Create NLTK data directory in user's home directory
    nltk_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    print(f"NLTK data directory: {nltk_dir}")
    
    # Dictionary of package names and their GitHub URLs
    packages = {
        "punkt": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip",
        "stopwords": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip",
        "wordnet": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"
    }
    
    # Download and extract each package
    for package_name, url in packages.items():
        print(f"\nDownloading {package_name}...")
        try:
            # Create a temporary file for the download
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                temp_file_path = temp_file.name
            
            # Download the package
            urllib.request.urlretrieve(url, temp_file_path)
            print(f"Downloaded {package_name} to {temp_file_path}")
            
            # Determine the correct directory to extract to
            if package_name == "punkt":
                extract_dir = os.path.join(nltk_dir, "tokenizers")
            else:
                extract_dir = os.path.join(nltk_dir, "corpora")
            
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract the package
            with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted {package_name} to {extract_dir}")
            
            # Clean up
            os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"Error downloading {package_name}: {e}")
    
    # Test if the packages are correctly installed
    try:
        import nltk
        nltk.data.path.append(nltk_dir)
        
        # Try to access each resource
        print("\nTesting installed packages:")
        nltk.data.find('tokenizers/punkt')
        print("✅ punkt is available")
        
        nltk.data.find('corpora/stopwords')
        print("✅ stopwords is available")
        
        nltk.data.find('corpora/wordnet')
        print("✅ wordnet is available")
        
        print("\nAll NLTK packages installed successfully!")
        print(f"NLTK data directory: {nltk_dir}")
        print("\nYou can now run the dashboard with:")
        print("streamlit run embedded_app.py")
        
    except Exception as e:
        print(f"Error testing packages: {e}")
        print("\nSome packages may not be correctly installed.")
        print("Consider using the embedded_app.py which doesn't require NLTK resources.")

if __name__ == "__main__":
    download_nltk_manually()