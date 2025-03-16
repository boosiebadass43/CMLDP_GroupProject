#!/usr/bin/env python3

import sys
import re

def check_syntax(filename):
    """Check file for common syntax errors line by line."""
    errors_found = False
    
    with open(filename, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                # Check for mismatched quotes
                single_quotes = line.count("'")
                double_quotes = line.count('"')
                if single_quotes % 2 != 0:
                    print(f"Line {i}: Odd number of single quotes: {line.strip()}")
                    errors_found = True
                if double_quotes % 2 != 0:
                    print(f"Line {i}: Odd number of double quotes: {line.strip()}")
                    errors_found = True
                
                # Check for unbalanced parentheses
                if line.count('(') != line.count(')'):
                    print(f"Line {i}: Unbalanced parentheses: {line.strip()}")
                    errors_found = True
                
                # Check for mismatched brackets
                if line.count('[') != line.count(']'):
                    print(f"Line {i}: Unbalanced square brackets: {line.strip()}")
                    errors_found = True
                
                # Check for mismatched braces
                if line.count('{') != line.count('}'):
                    print(f"Line {i}: Unbalanced curly braces: {line.strip()}")
                    errors_found = True
                
                # Check for trailing commas or semicolons
                if re.search(r',\s*$', line) and not re.search(r'\[\s*,\s*\]', line):
                    print(f"Line {i}: Trailing comma: {line.strip()}")
                
                # Look for unterminated strings
                if re.search(r'(".*?[^\\]"|\'.*?[^\\]\')$', line):
                    print(f"Line {i}: Possible unterminated string: {line.strip()}")
                
            except Exception as e:
                print(f"Error checking line {i}: {e}")
                
    if not errors_found:
        print(f"No obvious syntax errors found in {filename}")
    
    return not errors_found

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_syntax.py <filename>")
        sys.exit(1)
        
    filename = sys.argv[1]
    success = check_syntax(filename)
    sys.exit(0 if success else 1)