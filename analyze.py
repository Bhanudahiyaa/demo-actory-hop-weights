#!/usr/bin/env python3
"""
Simple command-line interface for analyzing web page neighborhoods.
Usage: python analyze.py <url>
"""

import sys
from analyze_neighbors import analyze_neighborhood

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <url>")
        print("Example: python analyze.py https://zineps.com")
        sys.exit(1)
    
    url = sys.argv[1]
    analyze_neighborhood(url)

if __name__ == "__main__":
    main()
