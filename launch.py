#!/usr/bin/env python3
"""
One-click launcher for Interactive Scientific Plotter
Just run: python launch.py
"""

import subprocess
import sys
import os
import webbrowser
import time

def install_dependencies():
    """Auto-install required packages"""
    packages = [
        'streamlit',
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'openpyxl',
        'xlrd'
    ]
    
    print("📦 Installing dependencies...")
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
    print("✅ All dependencies ready!\n")

def main():
    """Launch the interactive plotter"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("="*50)
    print("   📊 Interactive Scientific Plotter")
    print("="*50)
    print()
    
    # Install dependencies
    try:
        import streamlit
    except ImportError:
        install_dependencies()
    
    print("🚀 Launching interactive interface...")
    print()
    print("The application will open in your browser automatically.")
    print("If it doesn't, visit: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop")
    print("-"*50)
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run',
        'interactive_plotter.py',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped")
        sys.exit(0)