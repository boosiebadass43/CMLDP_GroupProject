#!/bin/bash
# Fix the embedded_app.py file if there's a syntax error on line 235

# Define file path
EMBEDDED_APP="embedded_app.py"

# Check if the file exists
if [ ! -f "$EMBEDDED_APP" ]; then
    echo "Error: $EMBEDDED_APP not found in the current directory."
    exit 1
fi

# Backup the original file
cp "$EMBEDDED_APP" "${EMBEDDED_APP}.bak"
echo "Created backup: ${EMBEDDED_APP}.bak"

# Look for the problematic line
if grep -q "col = re.sub(r'_," "$EMBEDDED_APP"; then
    echo "Found syntax error. Fixing..."
    
    # Replace the problematic line with the correct one
    # This uses sed to find and replace the line with the syntax error
    sed -i '' 's/col = re.sub(r'\''_,/col = re.sub(r'\''_\$'\'',/g' "$EMBEDDED_APP"
    echo "Fixed the syntax error."
else
    echo "No syntax error found on line 235. The file appears to be correct."
fi

# Check if the file can be parsed by Python
python3 -c "import ast; ast.parse(open('$EMBEDDED_APP').read())" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ File syntax looks good!"
else
    echo "❌ There are still syntax errors in the file."
    echo "Please use the embedded_app.py.bak file and fix the errors manually:"
    
    # Attempt to identify any remaining syntax errors
    python3 -c "import ast; ast.parse(open('$EMBEDDED_APP').read())" 2>&1
fi

echo -e "\nNext steps:"
echo "1. Run the dashboard with: streamlit run embedded_app.py"
echo "2. If you still have issues, try: streamlit run simple_app.py"