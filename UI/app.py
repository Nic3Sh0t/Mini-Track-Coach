#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrackCoach Web UI - Streamlit Application
Provides a graphical interface for track driving data analysis
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import json
import subprocess
import io
from datetime import datetime
import time

# Fix Windows console encoding (only in non-Streamlit environments)
# Streamlit already handles encoding, no need to replace sys.stdout
# In Streamlit environment, sys.stdout is already taken over by Streamlit, cannot be directly replaced
if sys.platform == 'win32':
    try:
        # Check if stdout is available and not closed
        if (hasattr(sys.stdout, 'buffer') and 
            not sys.stdout.buffer.closed and 
            not isinstance(sys.stdout, io.TextIOWrapper)):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if (hasattr(sys.stderr, 'buffer') and 
            not sys.stderr.buffer.closed and 
            not isinstance(sys.stderr, io.TextIOWrapper)):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, ValueError, OSError):
        # If replacement fails (e.g., in Streamlit environment), ignore error
        # Streamlit already handles encoding, no need for manual handling
        pass

# Set page configuration
st.set_page_config(
    page_title="TrackCoach - Track Data Analysis System",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project path settings
# Get the directory where this file is located (UI directory), use resolve() to ensure absolute path
UI_DIR = Path(__file__).resolve().parent
# Parent directory (contains trackcoach module)
NEW_TEST_DIR = UI_DIR.parent.resolve()
# Project root directory (parent of NEW_TEST_DIR)
PROJECT_ROOT = NEW_TEST_DIR.parent.resolve()

# Add parent directory to Python path (for importing trackcoach)
if NEW_TEST_DIR.exists():
    if str(NEW_TEST_DIR) not in sys.path:
        sys.path.insert(0, str(NEW_TEST_DIR))

# Note: Do not change working directory here
# Streamlit needs to run in the directory where app.py is located
# Pipeline will temporarily change directory in the run_pipeline function

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    if 'pipeline_output' not in st.session_state:
        st.session_state.pipeline_output = None
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None


def run_pipeline(csv_path, output_dir, params):
    """Run pipeline"""
    try:
        # Save current working directory and Python path
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        # Set up environment similar to run_pipeline.py
        # 1. Add parent directory to Python path (so Python can find trackcoach module)
        if str(NEW_TEST_DIR) not in sys.path:
            sys.path.insert(0, str(NEW_TEST_DIR))
        
        # 2. Change to project root directory (consistent with run_pipeline.py)
        os.chdir(PROJECT_ROOT)
        
        # Use the same approach as run_pipeline.py: run via module import
        # Instead of directly running pipeline.py file (because correct module path is needed)
        # Build command
        cmd = [
            sys.executable, '-m', 'trackcoach.pipeline',
            '--csv', str(csv_path),
            '--outdir', str(output_dir)
        ]
        
        # Add column name override parameters
        if params.get('lat_col'):
            cmd.extend(['--lat-col', params['lat_col']])
        if params.get('lon_col'):
            cmd.extend(['--lon-col', params['lon_col']])
        if params.get('time_col'):
            cmd.extend(['--time-col', params['time_col']])
        if params.get('speed_col'):
            cmd.extend(['--speed-col', params['speed_col']])
        if params.get('lap_col'):
            cmd.extend(['--lap-col', params['lap_col']])
        if params.get('along_col'):
            cmd.extend(['--along-col', params['along_col']])
        
        # Add other parameters
        if params.get('analyze_issues'):
            cmd.append('--analyze-issues')
        if params.get('generate_advice'):
            cmd.append('--generate-advice')
            if params.get('advice_config'):
                cmd.extend(['--advice-config', params['advice_config']])
        if params.get('plot', True):
            cmd.extend(['--plot', 'true'])
        if params.get('ref_lap'):
            cmd.extend(['--ref-lap', params['ref_lap']])
        if params.get('smooth_win_sec'):
            cmd.extend(['--smooth-win-sec', str(params['smooth_win_sec'])])
        if params.get('apex_height_quantile'):
            cmd.extend(['--apex-height-quantile', str(params['apex_height_quantile'])])
        if params.get('min_turn_gap_m'):
            cmd.extend(['--min-turn-gap-m', str(params['min_turn_gap_m'])])
        
        # Run command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            cwd=str(PROJECT_ROOT)
        )
        
        # Real-time output
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            yield line
        
        process.wait()
        
        if process.returncode == 0:
            yield "\n✅ Pipeline executed successfully!"
        else:
            yield f"\n❌ Pipeline execution failed, return code: {process.returncode}"
        
        # Restore original working directory and Python path
        os.chdir(original_cwd)
        sys.path[:] = original_path
            
    except Exception as e:
        # Ensure working directory and Python path are restored even on error
        try:
            os.chdir(original_cwd)
            sys.path[:] = original_path
        except:
            pass
        yield f"❌ Error: {str(e)}"


def load_results(output_dir=None, output_path=None):
    """Load analysis results"""
    results = {}
    if output_path is None:
        output_path = PROJECT_ROOT / output_dir if output_dir else NEW_TEST_DIR
    else:
        output_path = Path(output_path)
    
    # Load CSV files
    csv_files = {
        'corner_cards': 'corner_cards.csv',
        'corner_cards_delta': 'corner_cards_with_delta.csv',
        'events': 'events.csv',
        'top_issues': 'top_issues.csv',
        'stability': 'stability_report.csv'
    }
    
    for key, filename in csv_files.items():
        file_path = output_path / filename
        if file_path.exists():
            try:
                results[key] = pd.read_csv(file_path)
            except Exception as e:
                st.warning(f"Failed to load {filename}: {e}")
    
    # Load JSON files
    json_files = {
        'benchmark_lap': 'benchmark_lap.json',
        'pb_memory': 'pb_memory.json'
    }
    
    for key, filename in json_files.items():
        file_path = output_path / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results[key] = json.load(f)
            except Exception as e:
                st.warning(f"Failed to load {filename}: {e}")
    
    # Find image files
    image_files = {
        'track_layout': 'track_layout_apexes.png',
        'curvature_apex': 'curvature_apex.png',
        'top_issues_bar': 'top_issues_bar.png',
        'consistency_heatmap': 'consistency_heatmap.png',
        'consistency_heatmap_norm': 'consistency_heatmap_norm.png',
        'xy_speed': 'xy_speed.png',
        'xy_brakes': 'xy_brakes.png'
    }
    
    results['images'] = {}
    for key, filename in image_files.items():
        file_path = output_path / filename
        if file_path.exists():
            results['images'][key] = str(file_path)
    
    # Find report files
    report_files = {
        'coaching_advice_md': 'coaching_advice_report.md',
        'coaching_advice_pdf': 'coaching_advice_report.pdf',
        'events_summary': 'events_summary.md'
    }
    
    results['reports'] = {}
    for key, filename in report_files.items():
        file_path = output_path / filename
        if file_path.exists():
            results['reports'][key] = str(file_path)
    
    return results


def main():
    """Main function"""
    init_session_state()
    
    # Title
    st.markdown('<h1 class="main-header">🏎️ TrackCoach - Track Data Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar - Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Data Analysis", "📈 Results Viewer", "🔍 Model Evaluation", "ℹ️ About"]
    )
    
    if page == "📊 Data Analysis":
        show_analysis_page()
    elif page == "📈 Results Viewer":
        show_results_page()
    elif page == "🔍 Model Evaluation":
        show_evaluation_page()
    else:
        show_about_page()


def show_analysis_page():
    """Data analysis page"""
    st.header("📊 Data Analysis")
    st.markdown("Upload CSV data file and configure parameters to run analysis")
    
    # File upload
    st.subheader("1. Upload Data File")
    uploaded_file = st.file_uploader(
        "Select CSV File",
        type=['csv'],
        help="Upload CSV file containing track data (time, GPS coordinates, speed, lap number, etc.)"
    )
    
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        
        # Display file information
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📄 File Name: {uploaded_file.name}")
        with col2:
            st.info(f"📦 File Size: {uploaded_file.size / 1024:.2f} KB")
        
        # Preview data
        if st.checkbox("Preview Data (First 5 Rows)"):
            try:
                df_preview = pd.read_csv(uploaded_file, nrows=5)
                st.dataframe(df_preview)
                uploaded_file.seek(0)  # Reset file pointer
            except Exception as e:
                st.error(f"Failed to read file: {e}")
    
    # Parameter configuration
    st.subheader("2. Configure Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Settings")
        output_dir = st.text_input(
            "Output Directory",
            value=f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Analysis results will be saved to this directory (relative to project root)"
        )
        
        ref_lap = st.selectbox(
            "Reference Lap Mode",
            ['fastest', 'median'],
            help="Select reference lap for comparison"
        )
        
        enable_plot = st.checkbox("Generate Plots", value=True)
        
        analyze_issues = st.checkbox("Analyze Issues", value=True)
        generate_advice = st.checkbox("Generate AI Advice", value=False)
    
    with col2:
        st.markdown("#### Advanced Parameters")
        smooth_win_sec = st.number_input(
            "Velocity Smoothing Window (seconds)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        
        apex_height_quantile = st.slider(
            "Curvature Peak Threshold Quantile",
            min_value=0.5,
            max_value=0.95,
            value=0.78,
            step=0.01
        )
        
        min_turn_gap_m = st.number_input(
            "Minimum Turn Gap (meters)",
            min_value=20.0,
            max_value=100.0,
            value=65.0,
            step=5.0
        )
        
        if generate_advice:
            advice_config = st.text_input(
                "Advice Config File",
                value="llamacloud_config.json",
                help="LLM configuration file path. Create from llamacloud_config.json.example and fill in your API keys"
            )
        else:
            advice_config = None
    
    # Column name override options (if auto-detection fails)
    st.subheader("2.5. Column Name Override (Optional)")
    st.markdown("If auto-detection of column names fails, you can manually specify column names")
    
    col3, col4 = st.columns(2)
    with col3:
        lat_col = st.text_input("Latitude Column Name", value="", help="e.g., latitude_deg")
        lon_col = st.text_input("Longitude Column Name", value="", help="e.g., longitude_deg")
        time_col = st.text_input("Time Column Name", value="", help="e.g., timestamp")
    
    with col4:
        speed_col = st.text_input("Speed Column Name", value="", help="e.g., speed_m/s")
        lap_col = st.text_input("Lap Column Name", value="", help="e.g., lap_number")
        along_col = st.text_input("Longitudinal Acceleration Column Name", value="", help="e.g., longitudinal_acc_G")
    
    # Run analysis
    st.subheader("3. Run Analysis")
    
    if st.button("🚀 Start Analysis", type="primary", disabled=not uploaded_file):
        if not uploaded_file:
            st.error("Please upload a CSV file first")
            return
        
        # Save uploaded file
        temp_dir = PROJECT_ROOT / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)
        csv_path = temp_dir / uploaded_file.name
        
        with open(csv_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.output_dir = output_dir
        st.session_state.pipeline_running = True
        
        # Prepare parameters
        params = {
            'analyze_issues': analyze_issues,
            'generate_advice': generate_advice,
            'advice_config': advice_config,
            'plot': enable_plot,
            'ref_lap': ref_lap,
            'smooth_win_sec': smooth_win_sec,
            'apex_height_quantile': apex_height_quantile,
            'min_turn_gap_m': min_turn_gap_m,
            'lat_col': lat_col if lat_col else None,
            'lon_col': lon_col if lon_col else None,
            'time_col': time_col if time_col else None,
            'speed_col': speed_col if speed_col else None,
            'lap_col': lap_col if lap_col else None,
            'along_col': along_col if along_col else None
        }
        
        # Display progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_container = st.container()
        
        with output_container:
            st.markdown("#### Execution Log")
            log_output = st.empty()
            log_lines = []
            
            status_text.text("Running Pipeline...")
            
            try:
                for line in run_pipeline(csv_path, output_dir, params):
                    log_lines.append(line)
                    log_output.text_area("Execution Log", "\n".join(log_lines[-50:]), height=300, label_visibility="collapsed")
                    progress_bar.progress(min(100, len(log_lines) * 2))
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                st.session_state.pipeline_running = False
                st.session_state.pipeline_output = output_dir
                
                st.success(f"Analysis results saved to: {output_dir}")
                st.balloons()
                
            except Exception as e:
                st.error(f"Execution failed: {e}")
                st.session_state.pipeline_running = False


def show_results_page():
    """Results viewer page"""
    st.header("📈 Results Viewer")
    
    # Select output directory
    output_dirs = []
    base_dir = NEW_TEST_DIR  # Parent directory
    if base_dir.exists():
        for item in base_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != 'trackcoach' and item.name != 'Eval' and item.name != 'UI':
                # Check if it contains result files
                if (item / 'corner_cards.csv').exists():
                    output_dirs.append(item.name)
    
    if not output_dirs:
        st.warning("No analysis results found. Please run analysis on the 'Data Analysis' page first.")
        return
    
    selected_dir = st.selectbox("Select Output Directory", output_dirs)
    
    if selected_dir:
        output_path = base_dir / selected_dir
        results = load_results(selected_dir, base_dir / selected_dir)
        
        if not results:
            st.warning("No result files found in this directory.")
            return
        
        # Display overview
        st.subheader("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if 'corner_cards' in results:
            with col1:
                st.metric("Number of Corners", len(results['corner_cards']) if isinstance(results['corner_cards'], pd.DataFrame) else 0)
        
        if 'events' in results:
            with col2:
                events_df = results['events']
                if isinstance(events_df, pd.DataFrame):
                    st.metric("Number of Events", len(events_df))
        
        if 'top_issues' in results:
            with col3:
                issues_df = results['top_issues']
                if isinstance(issues_df, pd.DataFrame):
                    st.metric("Top Issues Count", len(issues_df))
        
        if 'benchmark_lap' in results:
            with col4:
                benchmark = results['benchmark_lap']
                if isinstance(benchmark, dict):
                    st.metric("Benchmark Lap", benchmark.get('lap', 'N/A'))
        
        # Display charts
        if 'images' in results and results['images']:
            st.subheader("📈 Visualization Charts")
            
            # Track layout
            if 'track_layout' in results['images']:
                st.markdown("#### Track Layout with Apexes")
                st.image(results['images']['track_layout'], use_container_width=True)
            
            # Consistency heatmap
            cols = st.columns(2)
            if 'consistency_heatmap' in results['images']:
                with cols[0]:
                    st.markdown("#### Consistency Heatmap (Raw)")
                    st.image(results['images']['consistency_heatmap'], use_container_width=True)
            
            if 'consistency_heatmap_norm' in results['images']:
                with cols[1]:
                    st.markdown("#### Consistency Heatmap (Normalized)")
                    st.image(results['images']['consistency_heatmap_norm'], use_container_width=True)
            
            # Top Issues
            if 'top_issues_bar' in results['images']:
                st.markdown("#### Top Issues")
                st.image(results['images']['top_issues_bar'], use_container_width=True)
            
            # Speed and braking charts
            cols2 = st.columns(2)
            if 'xy_speed' in results['images']:
                with cols2[0]:
                    st.markdown("#### Speed Trajectory")
                    st.image(results['images']['xy_speed'], use_container_width=True)
            
            if 'xy_brakes' in results['images']:
                with cols2[1]:
                    st.markdown("#### Braking Points")
                    st.image(results['images']['xy_brakes'], use_container_width=True)
        
        # Display data tables
        st.subheader("📋 Data Tables")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Corner Data", "Events", "Top Issues", "Stability"])
        
        with tab1:
            if 'corner_cards_delta' in results:
                st.dataframe(results['corner_cards_delta'], use_container_width=True)
            elif 'corner_cards' in results:
                st.dataframe(results['corner_cards'], use_container_width=True)
        
        with tab2:
            if 'events' in results:
                st.dataframe(results['events'], use_container_width=True)
        
        with tab3:
            if 'top_issues' in results:
                st.dataframe(results['top_issues'], use_container_width=True)
        
        with tab4:
            if 'stability' in results:
                st.dataframe(results['stability'], use_container_width=True)
        
        # Display reports
        if 'reports' in results and results['reports']:
            st.subheader("📄 Reports")
            
            if 'coaching_advice_md' in results['reports']:
                st.markdown("#### AI Coaching Advice Report")
                with open(results['reports']['coaching_advice_md'], 'r', encoding='utf-8') as f:
                    report_content = f.read()
                st.markdown(report_content)
                
                # Download button
                st.download_button(
                    "Download Markdown Report",
                    report_content,
                    file_name="coaching_advice_report.md",
                    mime="text/markdown"
                )


def show_evaluation_page():
    """Model evaluation page"""
    st.header("🔍 Model Evaluation")
    st.markdown("Compare the quality of coaching advice generated by different LLM models")
    
    eval_dir = NEW_TEST_DIR / "Eval"
    
    if not eval_dir.exists():
        st.warning("Evaluation directory does not exist.")
        return
    
    # Find evaluation results
    eval_files = list(eval_dir.glob("eval_*.csv"))
    eval_summary = eval_dir / "eval_summary.csv"
    
    if eval_summary.exists():
        st.subheader("📊 Evaluation Summary")
        try:
            summary_df = pd.read_csv(eval_summary)
            st.dataframe(summary_df, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load evaluation summary: {e}")
    
    if eval_files:
        st.subheader("📈 Detailed Evaluation Results")
        
        selected_file = st.selectbox(
            "Select Model Evaluation Result",
            [f.name for f in eval_files]
        )
        
        if selected_file:
            try:
                eval_df = pd.read_csv(eval_dir / selected_file)
                st.dataframe(eval_df, use_container_width=True)
                
                # Display evaluation metrics
                if 'bert_f1' in eval_df.columns:
                    st.markdown("#### BERTScore F1")
                    st.bar_chart(eval_df.set_index('turn')['bert_f1'])
                
            except Exception as e:
                st.error(f"Failed to load evaluation result: {e}")
    
    # Run evaluation button
    st.subheader("🔄 Run New Evaluation")
    
    if st.button("Run All Model Evaluations"):
        with st.spinner("Running evaluation..."):
            try:
                eval_script = NEW_TEST_DIR / "Eval" / "evaluate_all_models.py"
                result = subprocess.run(
                    [sys.executable, str(eval_script)],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    cwd=str(PROJECT_ROOT)
                )
                
                if result.returncode == 0:
                    st.success("Evaluation complete!")
                    st.text(result.stdout)
                else:
                    st.error(f"Evaluation failed: {result.stderr}")
            except Exception as e:
                st.error(f"Execution error: {e}")


def show_about_page():
    """About page"""
    st.header("ℹ️ About TrackCoach")
    
    st.markdown("""
    ### Project Introduction
    
    TrackCoach is a professional track driving data analysis tool for:
    
    - 📊 **Data Analysis**: Process track driving data (time, position, speed, etc.)
    - 🔍 **Issue Detection**: Automatically identify driving issues (late braking, early braking, overslow, etc.)
    - 🤖 **AI Advice**: Generate personalized coaching recommendations using large language models
    - 📈 **Visualization**: Generate various charts to display analysis results
    - 📊 **Model Evaluation**: Compare the quality of advice generated by different LLM models
    
    ### Main Features
    
    1. **Data Preprocessing**
       - Automatic CSV column name recognition
       - Speed unit conversion
       - Coordinate projection
    
    2. **Apex Detection**
       - Automatic apex detection based on curvature
       - Generate corner-by-corner data cards
    
    3. **Event Detection**
       - Heavy braking detection
       - Conservative entry detection
       - Relative event detection
    
    4. **Stability Analysis**
       - Consistency heatmap
       - Stability scoring
    
    5. **AI Coaching Advice**
       - Support for multiple LLM models
       - Personalized advice generation
    
    ### Technology Stack
    
    - Python 3.x
    - Pandas, NumPy (data processing)
    - Matplotlib, Seaborn (visualization)
    - Streamlit (Web UI)
    - Transformers (NLI evaluation)
    
    ### Usage Instructions
    
    1. Upload CSV file on the "Data Analysis" page
    2. Configure analysis parameters
    3. Run analysis and view results
    4. Browse detailed results on the "Results Viewer" page
    5. Compare different models on the "Model Evaluation" page
    
    ### Contact & Support
    
    For questions or suggestions, please contact the project maintainer.
    """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # If error occurs during startup, display error message
        st.error(f"Error starting application: {str(e)}")
        st.exception(e)
        import traceback
        st.code(traceback.format_exc(), language='python')
