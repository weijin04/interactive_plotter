# Interactive Scientific Plotter

A powerful Streamlit-based tool for creating publication-quality scientific plots with interactive controls.

## Features

### Supported Plot Types
- **Scatter Plot**: With regression analysis, perfect prediction line, and statistical metrics
- **Bar Plot**: With value labels and customizable colors
- **Box Plot**: With baseline comparison and group analysis
- **Histogram**: With KDE and normal distribution overlay
- **Pie Chart**: With percentage labels and exploded slices

### Key Capabilities
- Load data from CSV or Excel files (via file path or upload)
- Interactive column selection for axes
- Real-time plot preview
- Customizable plot elements:
  - Perfect prediction line (y=x)
  - Linear regression with equation
  - Statistical metrics (RMSE, MAE, R², MAPE)
  - Customizable colors, transparency, and sizes
- Export options: PNG, SVG, PDF
- Data preview and statistics

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run interactive_plotter.py
```

The application will open in your default web browser.

### Basic Workflow
1. **Load Data**: Enter file path or upload CSV/Excel file
2. **Select Plot Type**: Choose from available plot types
3. **Configure Axes**: Select columns for X and Y axes
4. **Customize**: Adjust optional elements and styling
5. **Export**: Download plot in desired format

### Plot-Specific Features

#### Scatter Plot
- Perfect prediction line (red dashed)
- Linear regression line (green solid) with equation
- Statistical metrics box (RMSE, MAE, R², MAPE)
- Adjustable point size and transparency

#### Bar Plot
- Optional value labels on bars
- Automatic sorting by value
- Customizable bar colors

#### Box Plot
- Group comparison
- Baseline reference line
- Outlier detection

#### Histogram
- Adjustable bin count
- KDE (Kernel Density Estimation) overlay
- Normal distribution fitting

## Style Guide

The tool follows scientific publication standards:
- Clean, minimalist design
- High contrast colors
- Clear axis labels
- Grid lines for reference
- Statistical information in legend boxes

## Tips

1. **For best results**: Ensure your data is clean and properly formatted
2. **Column names**: Use descriptive column names as they become axis labels
3. **Large datasets**: The tool handles large datasets efficiently but preview is limited to 100 rows
4. **Export quality**: Use SVG or PDF for publication-quality vector graphics

## Example Data Format

### CSV
```csv
Predicted,Actual
1.2,1.3
2.1,2.0
3.5,3.7
```

### Excel
Standard Excel files with headers in the first row are supported.

## Troubleshooting

- **File not found**: Ensure the file path is absolute and correct
- **Columns not showing**: Check that numeric columns are properly formatted
- **Plot errors**: Verify data types match the selected plot type


## Screenshot
<img width="1354" height="897" alt="截屏2025-08-31 05 11 55" src="https://github.com/user-attachments/assets/e298d883-74ba-4b9b-90f7-5d2af1a376b2" />


## License

MIT License - Free for academic and commercial use
