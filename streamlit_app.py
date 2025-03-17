# This is the main entry point for Streamlit Cloud
# Import and run the embedded_app version which has all dependencies packaged

import sys
import os

# If running in Streamlit Cloud, we need to ensure the current directory is in the path
# so Python can find our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the main function from embedded_app
from embedded_app import main

# Execute the main function to run the dashboard
if __name__ == "__main__":
    main()