# Small Business Federal Contracting Dashboard

An interactive dashboard for analyzing small business challenges with federal contract onboarding.

## Overview

This dashboard provides actionable insights into the barriers small businesses face when pursuing federal contracts. It includes comprehensive data cleaning, visualization, and analysis features designed for executive-level presentations.

## Features

- **Interactive Filtering**: Filter data by respondent type, complexity rating, and timeline
- **Executive Summary**: At-a-glance key insights for busy executives
- **Multiple Analysis Views**: Tabbed interface with key challenges, detailed analysis, text analysis, and recommendations
- **Professional Visualizations**: High-quality charts and graphs with consistent professional styling
- **Text Analysis**: Analysis of open-ended responses including word frequency and common phrases
- **Actionable Recommendations**: Data-driven recommendations based on survey findings

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Fix any potential issues with SSL certificates (macOS users):
   ```
   python fix_certificates.py
   ```
   This script will test and attempt to fix SSL certificate issues that might prevent downloading NLTK data.

4. Download NLTK data using one of these approaches:

   **Option A: Manual download (Most reliable):**
   ```
   python download_nltk_manually.py
   ```
   This script downloads NLTK data packages directly from GitHub and extracts them to the proper location, bypassing SSL issues.

   **Option B: Download to your home directory:**
   ```
   python download_nltk_data.py
   ```
   This script will download NLTK resources to your home directory (`~/nltk_data`), which is checked by default.

   **Option C: Download to the project directory:**
   ```
   python setup_nltk.py
   ```
   This script will download NLTK resources to a local directory in the project.

5. Fix any syntax errors in the embedded dashboard:
   ```
   bash fix_embedded_app.sh
   ```
   This will automatically detect and fix any syntax errors in the embedded_app.py file.

## Usage

You can run the application in one of these ways:

### Option 1: Embedded Dashboard (Most Reliable)
This version includes all necessary data processing code with no external dependencies:
```
streamlit run embedded_app.py
```
This dashboard has built-in stopwords, pure Python text processing, and robust CSV parsing with fallbacks.

### Option 2: Simplified Dashboard
A simpler version with minimal dependencies:
```
streamlit run simple_app.py
```
This version uses pure Python for text processing with a more streamlined interface.

### Option 3: Full Dashboard (Original)
The original dashboard implementation:
```
streamlit run app.py
```
Note: This version may require additional NLTK resources.

## Data Structure

The dashboard uses the `survey_data.csv` file located in the `data/` directory. This dataset contains survey responses from various stakeholders in the federal contracting ecosystem.

## Dashboard Sections

1. **Key Challenges**: Overview of the most significant hurdles and barriers
2. **Detailed Analysis**: In-depth analysis of challenges by respondent type and other factors
3. **Open-Ended Responses**: Analysis of text responses and suggested changes
4. **Recommendations**: Data-driven recommendations and implementation timeline

## Data Cleaning Process

The dashboard implements a comprehensive data cleaning pipeline that includes:
- Standardization of column names
- Conversion of appropriate data types
- Handling of missing values
- Splitting of multi-entry columns
- Standardization of text entries
- Removal of duplicates
- Creation of derived features for analysis