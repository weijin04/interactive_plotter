#!/bin/bash

# Interactive Scientific Plotter - One-Click Launcher

echo "🚀 Starting Interactive Scientific Plotter..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check and install required packages
echo "📦 Checking dependencies..."
pip3 install -q streamlit pandas numpy matplotlib seaborn scipy scikit-learn openpyxl xlrd 2>/dev/null

# Clear terminal for clean interface
clear

echo "="*50
echo "   📊 Interactive Scientific Plotter"
echo "="*50
echo ""
echo "✅ All dependencies installed"
echo "🌐 Opening browser interface..."
echo ""
echo "📌 Tips:"
echo "   • Drag & drop your CSV/Excel file"
echo "   • Or enter file path directly"
echo "   • Select plot type and columns"
echo "   • Adjust settings in real-time"
echo ""
echo "Press Ctrl+C to stop the application"
echo "-"*50

# Run Streamlit app
streamlit run interactive_plotter.py --server.headless true --browser.gatherUsageStats false