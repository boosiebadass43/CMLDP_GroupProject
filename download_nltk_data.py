#!/usr/bin/env python
"""
Script to download NLTK data to the user's home directory.
This is useful when there are permission issues with the normal NLTK download locations.
"""

import os
import sys
import nltk

def download_to_home():
    """Download NLTK data to user's home directory"""
    print("Downloading NLTK resources to your home directory...")
    
    # Create NLTK data directory in home directory
    home_dir = os.path.expanduser("~")
    nltk_data_dir = os.path.join(home_dir, "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Ensure the path is in NLTK's search path
    nltk.data.path.insert(0, nltk_data_dir)
    
    # Resources to download
    resources = [
        'punkt',
        'stopwords',
        'wordnet'
    ]
    
    # Download each resource
    for resource in resources:
        print(f"Downloading {resource}...")
        try:
            nltk.download(resource, download_dir=nltk_data_dir)
            print(f"✅ Successfully downloaded {resource} to {nltk_data_dir}")
        except Exception as e:
            print(f"❌ Error downloading {resource}: {e}")
    
    # Verify downloads
    print("\nVerifying downloads...")
    success = True
    for resource in resources:
        try:
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet')
            print(f"✅ {resource} is available")
        except LookupError:
            print(f"❌ {resource} could not be found")
            success = False
    
    if success:
        print("\n🎉 All NLTK resources were successfully downloaded!")
        print(f"Resources are located in: {nltk_data_dir}")
        print("\nYou can now run the dashboard with: streamlit run app.py")
    else:
        print("\n⚠️ Some resources could not be verified.")
        print("Please check the errors above and try again.")

if __name__ == "__main__":
    download_to_home()