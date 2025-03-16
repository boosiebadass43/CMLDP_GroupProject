"""
Setup script to download NLTK data before running the Streamlit app.
Run this once before starting the app to ensure all NLTK resources are available.
"""

import os
import nltk

def setup_nltk():
    """Download necessary NLTK data packages"""
    print("Setting up NLTK resources...")
    
    # Create a directory for NLTK data
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add to NLTK path
    nltk.data.path.append(nltk_data_dir)
    
    # Download required resources
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")
    
    print("\nNLTK setup complete. You can now run the Streamlit app with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    setup_nltk()