#!/usr/bin/env python3
"""
Script to manually install NLTK resources
"""
import os
import nltk
import sys

def main():
    """Install NLTK resources and verify installation"""
    # Print current Python and NLTK versions
    print(f"Python version: {sys.version}")
    print(f"NLTK version: {nltk.__version__}")
    
    # Get the home directory path
    home_dir = os.path.expanduser("~")
    nltk_dir = os.path.join(home_dir, "nltk_data")
    
    # Create directory if it doesn't exist
    if not os.path.exists(nltk_dir):
        print(f"Creating directory: {nltk_dir}")
        os.makedirs(nltk_dir)
    else:
        print(f"Directory exists: {nltk_dir}")
    
    # Show all search paths
    print("\nNLTK search paths:")
    for path in nltk.data.path:
        print(f"  - {path}")
    
    # Add home directory to path
    if nltk_dir not in nltk.data.path:
        nltk.data.path.append(nltk_dir)
        print(f"\nAdded {nltk_dir} to NLTK search path")
    
    # Download resources with explicit path
    resources = ['punkt', 'stopwords', 'wordnet']
    
    print("\nDownloading NLTK resources:")
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource, download_dir=nltk_dir)
    
    # Verify downloads
    print("\nVerifying downloads:")
    success = True
    
    for resource in resources:
        try:
            # Try to find the resource
            if resource == 'punkt':
                path = nltk.data.find('tokenizers/punkt/english.pickle')
            elif resource == 'stopwords':
                path = nltk.data.find('corpora/stopwords/english')
                # Actually try to use it
                from nltk.corpus import stopwords
                words = stopwords.words('english')
                print(f"  - Found {len(words)} stopwords")
            elif resource == 'wordnet':
                path = nltk.data.find('corpora/wordnet')
            
            print(f"  ✓ {resource}: {path}")
        except LookupError as e:
            print(f"  ✗ {resource} not found: {e}")
            success = False
    
    if success:
        print("\n✓ All NLTK resources successfully installed!")
        print("\nYou can now run the dashboard with:")
        print("  streamlit run app.py")
    else:
        print("\n✗ Some resources could not be found. Please see the errors above.")

if __name__ == "__main__":
    main()