#!/usr/bin/env python3
"""
Interactive Scientific Plotter
A comprehensive tool for creating publication-quality scientific plots
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import io
import base64
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Interactive Scientific Plotter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class DataLoader:
    """Handle data loading from CSV and Excel files"""
    
    @staticmethod
    def load_file(file_path: str = None, uploaded_file = None) -> pd.DataFrame:
        """Load data from file path or uploaded file"""
        try:
            if file_path:
                path = Path(file_path)
                if not path.exists():
                    st.error(f"File not found: {file_path}")
                    return None
                    
                if path.suffix.lower() == '.csv':
                    return pd.read_csv(path)
                elif path.suffix.lower() in ['.xlsx', '.xls']:
                    return pd.read_excel(path)
                else:
                    st.error("Unsupported file format. Please use CSV or Excel files.")
                    return None
                    
            elif uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    return pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    return pd.read_excel(uploaded_file)
                    
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None


class PlotGenerator:
    """Generate various types of scientific plots"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.fig = None
        self.ax = None
        
    def create_figure(self, figsize=(10, 6)):
        """Create a new figure with specified size"""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        return self.fig, self.ax
        
    def scatter_plot(self, x_col, y_col, **kwargs):
        """Create scatter plot with optional regression lines"""
        if not self.fig:
            self.create_figure()
            
        # Extract options
        show_perfect = kwargs.get('show_perfect', False)
        show_regression = kwargs.get('show_regression', False)
        show_stats = kwargs.get('show_stats', False)
        alpha = kwargs.get('alpha', 0.6)
        size = kwargs.get('size', 50)
        color = kwargs.get('color', '#4287f5')
        
        # Plot scatter points
        x_data = self.data[x_col].dropna()
        y_data = self.data[y_col].dropna()
        
        # Align data
        valid_idx = x_data.index.intersection(y_data.index)
        x_data = x_data[valid_idx]
        y_data = y_data[valid_idx]
        
        self.ax.scatter(x_data, y_data, alpha=alpha, s=size, color=color, edgecolors='white', linewidth=0.5)
        
        # Add perfect prediction line
        if show_perfect:
            lims = [
                np.min([self.ax.get_xlim(), self.ax.get_ylim()]),
                np.max([self.ax.get_xlim(), self.ax.get_ylim()]),
            ]
            self.ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
            
        # Add regression line
        if show_regression and len(x_data) > 1:
            X = x_data.values.reshape(-1, 1)
            y = y_data.values
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Plot regression line
            self.ax.plot(x_data, y_pred, 'g-', linewidth=2, 
                        label=f'Linear Fit: y={model.coef_[0]:.2f}x{model.intercept_:+.2f}')
            
            # Calculate statistics
            if show_stats:
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
                mape = np.mean(np.abs((y - y_pred) / y)) * 100 if not np.any(y == 0) else np.nan
                
                # Add statistics to legend
                stats_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
                if not np.isnan(mape):
                    stats_text += f'\nMAPE: {mape:.2f}%'
                    
                # Create text box for statistics
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                self.ax.text(0.05, 0.95, stats_text, transform=self.ax.transAxes, 
                           fontsize=9, verticalalignment='top', bbox=props)
        
        # Labels and grid
        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.grid(True, alpha=0.3)
        
        if show_perfect or show_regression:
            self.ax.legend()
            
    def bar_plot(self, x_col, y_col, **kwargs):
        """Create bar plot"""
        if not self.fig:
            self.create_figure()
            
        # Extract options
        color = kwargs.get('color', '#4287f5')
        show_values = kwargs.get('show_values', False)
        
        # Prepare data
        if y_col:
            plot_data = self.data.groupby(x_col)[y_col].mean().sort_values(ascending=False)
        else:
            plot_data = self.data[x_col].value_counts()
            
        # Create bar plot
        bars = self.ax.bar(range(len(plot_data)), plot_data.values, color=color, alpha=0.8)
        self.ax.set_xticks(range(len(plot_data)))
        self.ax.set_xticklabels(plot_data.index, rotation=45, ha='right')
        
        # Add value labels on bars
        if show_values:
            for bar in bars:
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Labels and grid
        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col if y_col else 'Count')
        self.ax.grid(True, alpha=0.3, axis='y')
        
    def box_plot(self, y_col, group_col=None, **kwargs):
        """Create box plot"""
        if not self.fig:
            self.create_figure()
            
        # Extract options
        show_baseline = kwargs.get('show_baseline', False)
        baseline_value = kwargs.get('baseline_value', 0)
        colors = kwargs.get('colors', None)
        
        # Prepare data
        if group_col:
            plot_data = [self.data[self.data[group_col] == g][y_col].dropna() 
                        for g in self.data[group_col].unique()]
            labels = self.data[group_col].unique()
        else:
            plot_data = [self.data[y_col].dropna()]
            labels = [y_col]
            
        # Create box plot
        bp = self.ax.boxplot(plot_data, labels=labels, patch_artist=True,
                             boxprops=dict(alpha=0.7),
                             medianprops=dict(color='black', linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor='gray', 
                                           markersize=4, alpha=0.5))
        
        # Color boxes
        if colors:
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        else:
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(f'C{i}')
                
        # Add baseline
        if show_baseline:
            self.ax.axhline(y=baseline_value, color='r', linestyle='--', 
                          alpha=0.7, label=f'Baseline ({baseline_value:.3f})')
            self.ax.legend()
            
        # Labels and grid
        self.ax.set_ylabel(y_col)
        if group_col:
            self.ax.set_xlabel(group_col)
        self.ax.grid(True, alpha=0.3, axis='y')
        
    def histogram(self, col, **kwargs):
        """Create histogram with optional KDE"""
        if not self.fig:
            self.create_figure()
            
        # Extract options
        bins = kwargs.get('bins', 30)
        show_kde = kwargs.get('show_kde', False)
        show_normal = kwargs.get('show_normal', False)
        color = kwargs.get('color', '#4287f5')
        alpha = kwargs.get('alpha', 0.7)
        
        # Plot histogram
        data = self.data[col].dropna()
        n, bins_edges, patches = self.ax.hist(data, bins=bins, alpha=alpha, 
                                              color=color, edgecolor='black', linewidth=0.5)
        
        # Add KDE
        if show_kde:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            kde_values = kde(x_range)
            
            # Scale KDE to match histogram
            kde_values = kde_values * len(data) * (bins_edges[1] - bins_edges[0])
            self.ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
            
        # Add normal distribution
        if show_normal:
            mu, std = data.mean(), data.std()
            x_range = np.linspace(data.min(), data.max(), 200)
            normal_values = stats.norm.pdf(x_range, mu, std)
            
            # Scale to match histogram
            normal_values = normal_values * len(data) * (bins_edges[1] - bins_edges[0])
            self.ax.plot(x_range, normal_values, 'g--', linewidth=2, 
                        label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
            
        # Labels and grid
        self.ax.set_xlabel(col)
        self.ax.set_ylabel('Frequency')
        self.ax.grid(True, alpha=0.3, axis='y')
        
        if show_kde or show_normal:
            self.ax.legend()
            
    def pie_chart(self, col, **kwargs):
        """Create pie chart"""
        if not self.fig:
            self.create_figure()
            
        # Extract options
        show_percent = kwargs.get('show_percent', True)
        explode_first = kwargs.get('explode_first', False)
        
        # Prepare data
        value_counts = self.data[col].value_counts()
        
        # Create explode array
        explode = None
        if explode_first:
            explode = [0.1] + [0] * (len(value_counts) - 1)
            
        # Create pie chart
        wedges, texts, autotexts = self.ax.pie(
            value_counts.values, 
            labels=value_counts.index,
            autopct='%1.1f%%' if show_percent else None,
            explode=explode,
            startangle=90,
            colors=[f'C{i}' for i in range(len(value_counts))]
        )
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
            
        self.ax.set_title(f'Distribution of {col}')


def main():
    """Main application"""
    st.title("📊 Interactive Scientific Plotter")
    st.markdown("Create publication-quality scientific plots with ease")
    
    # Sidebar for data input and settings
    with st.sidebar:
        st.header("Data Input")
        
        # File input method
        input_method = st.radio("Input Method", ["File Path", "Upload File"])
        
        df = None
        if input_method == "File Path":
            file_path = st.text_input("Enter file path (CSV or Excel)", 
                                     placeholder="/path/to/your/data.csv")
            if file_path:
                df = DataLoader.load_file(file_path=file_path)
        else:
            uploaded_file = st.file_uploader("Choose a file", 
                                            type=['csv', 'xlsx', 'xls'])
            if uploaded_file:
                df = DataLoader.load_file(uploaded_file=uploaded_file)
        
        if df is not None:
            st.success(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Column selection
            st.header("Plot Configuration")
            plot_type = st.selectbox("Plot Type", 
                                    ["Scatter Plot", "Bar Plot", "Box Plot", 
                                     "Histogram", "Pie Chart"])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = df.columns.tolist()
            
    # Main area for plot
    if df is not None:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Plot Settings")
            
            if plot_type == "Scatter Plot":
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", numeric_cols)
                
                st.subheader("Optional Elements")
                show_perfect = st.checkbox("Show Perfect Prediction Line (y=x)", value=False)
                show_regression = st.checkbox("Show Linear Regression", value=False)
                show_stats = st.checkbox("Show Statistics in Legend", value=False)
                
                st.subheader("Style Settings")
                alpha = st.slider("Point Transparency", 0.1, 1.0, 0.6)
                size = st.slider("Point Size", 10, 200, 50)
                color = st.color_picker("Point Color", "#4287f5")
                
            elif plot_type == "Bar Plot":
                x_col = st.selectbox("X-axis (Category)", all_cols)
                y_col = st.selectbox("Y-axis (Value)", [None] + numeric_cols)
                
                st.subheader("Style Settings")
                show_values = st.checkbox("Show Values on Bars", value=False)
                color = st.color_picker("Bar Color", "#4287f5")
                
            elif plot_type == "Box Plot":
                y_col = st.selectbox("Value Column", numeric_cols)
                group_col = st.selectbox("Group By (Optional)", [None] + all_cols)
                
                st.subheader("Optional Elements")
                show_baseline = st.checkbox("Show Baseline", value=False)
                if show_baseline:
                    baseline_value = st.number_input("Baseline Value", value=0.0)
                    
            elif plot_type == "Histogram":
                col = st.selectbox("Column", numeric_cols)
                
                st.subheader("Optional Elements")
                bins = st.slider("Number of Bins", 10, 100, 30)
                show_kde = st.checkbox("Show KDE", value=False)
                show_normal = st.checkbox("Show Normal Distribution", value=False)
                
                st.subheader("Style Settings")
                alpha = st.slider("Bar Transparency", 0.1, 1.0, 0.7)
                color = st.color_picker("Bar Color", "#4287f5")
                
            elif plot_type == "Pie Chart":
                col = st.selectbox("Column", all_cols)
                
                st.subheader("Style Settings")
                show_percent = st.checkbox("Show Percentages", value=True)
                explode_first = st.checkbox("Explode First Slice", value=False)
            
            # Common settings
            st.subheader("Figure Settings")
            fig_width = st.slider("Figure Width", 6, 15, 10)
            fig_height = st.slider("Figure Height", 4, 10, 6)
            title = st.text_input("Plot Title", "")
            
        with col1:
            st.subheader("Plot Preview")
            
            # Generate plot
            plotter = PlotGenerator(df)
            plotter.create_figure(figsize=(fig_width, fig_height))
            
            try:
                if plot_type == "Scatter Plot":
                    plotter.scatter_plot(x_col, y_col, 
                                       show_perfect=show_perfect,
                                       show_regression=show_regression,
                                       show_stats=show_stats,
                                       alpha=alpha, size=size, color=color)
                                       
                elif plot_type == "Bar Plot":
                    plotter.bar_plot(x_col, y_col, 
                                   show_values=show_values, color=color)
                                   
                elif plot_type == "Box Plot":
                    kwargs = {}
                    if show_baseline:
                        kwargs['show_baseline'] = True
                        kwargs['baseline_value'] = baseline_value
                    plotter.box_plot(y_col, group_col, **kwargs)
                    
                elif plot_type == "Histogram":
                    plotter.histogram(col, bins=bins, show_kde=show_kde,
                                    show_normal=show_normal, 
                                    alpha=alpha, color=color)
                                    
                elif plot_type == "Pie Chart":
                    plotter.pie_chart(col, show_percent=show_percent,
                                    explode_first=explode_first)
                
                if title:
                    plotter.ax.set_title(title, fontsize=14, fontweight='bold')
                    
                plt.tight_layout()
                st.pyplot(plotter.fig)
                
                # Export options
                st.subheader("Export Options")
                col_exp1, col_exp2, col_exp3 = st.columns(3)
                
                with col_exp1:
                    # Save as PNG
                    buf = io.BytesIO()
                    plotter.fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="Download PNG",
                        data=buf,
                        file_name="plot.png",
                        mime="image/png"
                    )
                    
                with col_exp2:
                    # Save as SVG
                    buf = io.BytesIO()
                    plotter.fig.savefig(buf, format='svg', bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="Download SVG",
                        data=buf,
                        file_name="plot.svg",
                        mime="image/svg+xml"
                    )
                    
                with col_exp3:
                    # Save as PDF
                    buf = io.BytesIO()
                    plotter.fig.savefig(buf, format='pdf', bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="Download PDF",
                        data=buf,
                        file_name="plot.pdf",
                        mime="application/pdf"
                    )
                    
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
                
        # Data preview
        with st.expander("Data Preview"):
            st.dataframe(df.head(100))
            
        # Statistics
        with st.expander("Data Statistics"):
            st.write(df.describe())
            
    else:
        st.info("Please load a data file to begin plotting")


if __name__ == "__main__":
    main()