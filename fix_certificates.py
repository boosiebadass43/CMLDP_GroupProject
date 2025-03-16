#!/usr/bin/env python3
"""
This script fixes SSL certificate issues for Python on macOS 
by installing certificates from the system keychain.
"""
import os
import sys
import subprocess
import ssl
import certifi

def fix_certificates():
    """Check SSL certificates and try to fix them if needed"""
    print("Current certificate file being used:")
    print(f"certifi: {certifi.where()}")
    print(f"SSL default: {ssl.get_default_verify_paths().cafile}")
    
    # Test SSL connection to verify it works
    try:
        import urllib.request
        print("\nTesting connection to python.org...")
        response = urllib.request.urlopen("https://www.python.org")
        print(f"Connection successful! Status code: {response.status}")
        
        print("\nTesting connection to pypi.org...")
        response = urllib.request.urlopen("https://pypi.org")
        print(f"Connection successful! Status code: {response.status}")
    except Exception as e:
        print(f"Connection test failed: {e}")
        
        # Try to fix certificates on macOS
        if sys.platform == 'darwin':  # macOS
            print("\nAttempting to fix certificates on macOS...")
            
            # Option 1: Run Install Certificates.command if available
            python_base = os.path.dirname(os.path.dirname(sys.executable))
            cert_script = os.path.join(python_base, 'Install Certificates.command')
            
            if os.path.exists(cert_script):
                print(f"Running: {cert_script}")
                try:
                    subprocess.run(['bash', cert_script], check=True)
                    print("Certificate installation completed.")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to run certificate script: {e}")
            else:
                print(f"Certificate script not found at: {cert_script}")
                
            # Option 2: Copy certifi certificates to SSL directory
            print("\nCopying certifi certificates to SSL directory...")
            try:
                ssl_dir = os.path.dirname(ssl.get_default_verify_paths().cafile)
                if not os.path.exists(ssl_dir):
                    os.makedirs(ssl_dir, exist_ok=True)
                certifi_file = certifi.where()
                ssl_file = ssl.get_default_verify_paths().cafile
                
                print(f"Copying {certifi_file} to {ssl_file}")
                subprocess.run(['sudo', 'cp', certifi_file, ssl_file], check=True)
                print("Certificate copy completed.")
            except Exception as e:
                print(f"Certificate copy failed: {e}")
    
    # Check if we can download from NLTK now
    try:
        import nltk
        print("\nTesting NLTK download...")
        nltk.download('punkt', quiet=True)
        print("NLTK download successful!")
    except Exception as e:
        print(f"NLTK download still failing: {e}")
        print("\nAlternative solution for NLTK: Download data manually")
        print("1. Visit https://www.nltk.org/nltk_data/")
        print("2. Download these packages: punkt, stopwords, wordnet")
        print("3. Extract them to ~/nltk_data/")
    
    print("\nCertificate check and fix completed.")

if __name__ == "__main__":
    fix_certificates()