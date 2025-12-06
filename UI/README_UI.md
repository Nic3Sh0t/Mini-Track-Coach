# TrackCoach Web UI User Guide

## Introduction

TrackCoach Web UI is a Streamlit-based graphical interface that allows you to easily use all TrackCoach features through a web browser without command-line operations.

## Installation

### 1. Install Dependencies

```bash
# Navigate to UI directory
cd UI

# Install Streamlit (if not already installed)
pip install streamlit>=1.28.0

# Or install all UI-related dependencies
pip install -r requirements_ui.txt
```

### 2. Ensure Project Dependencies are Installed

```bash
# In project root directory (parent of UI directory)
cd ..
pip install -r requirements.txt
```

## Launch Methods

### Windows

Double-click `run_ui.bat`, or run in command line:

```bash
cd UI
run_ui.bat
```

### Linux/Mac

```bash
cd UI
chmod +x run_ui.sh
./run_ui.sh
```

### Manual Launch

```bash
cd UI
streamlit run app.py
```

After launching, the browser will automatically open at `http://localhost:8501`. If it doesn't open automatically, please manually visit this address.

## Feature Description

### 📊 Data Analysis Page

1. **Upload Data File**
   - Click "Select CSV File" to upload your track data CSV file
   - Supports preview of first 5 rows of data

2. **Configure Parameters**
   - **Basic Settings**:
     - Output Directory: Specify where results will be saved
     - Reference Lap Mode: Choose fastest or median
     - Generate Plots: Whether to generate visualization charts
     - Analyze Issues: Whether to perform issue analysis
     - Generate AI Advice: Whether to generate AI coaching advice
   
   - **Advanced Parameters**:
     - Velocity smoothing window
     - Curvature peak threshold
     - Minimum turn gap

3. **Run Analysis**
   - Click "Start Analysis" button
   - View execution logs in real-time
   - Success message will be displayed after analysis completes

### 📈 Results Viewer Page

1. **Select Output Directory**
   - Select previously analyzed result directory from dropdown list

2. **View Content**
   - **Data Overview**: Key metric statistics
   - **Visualization Charts**:
     - Track layout with apexes
     - Consistency heatmap
     - Top Issues bar chart
     - Speed trajectory chart
     - Braking points chart
   - **Data Tables**:
     - Corner data
     - Event list
     - Top Issues
     - Stability report
   - **Reports**:
     - AI coaching advice report (Markdown format)
     - Support for downloading reports

### 🔍 Model Evaluation Page

1. **View Evaluation Results**
   - Automatically loads evaluation summary
   - Select specific model to view detailed evaluation results
   - View evaluation metrics such as BERTScore, NLI, etc.

2. **Run New Evaluation**
   - Click "Run All Model Evaluations" button
   - System will evaluate all configured models

### ℹ️ About Page

View project introduction, feature descriptions, and technology stack information.

## Notes

1. **File Paths**: Ensure CSV file paths are correct, recommend using relative paths
2. **Output Directory**: Output directory will be created automatically, existing directories may be overwritten
3. **Model Configuration**: Generating AI advice requires LLM model configuration. Copy `llamacloud_config.json.example` to `llamacloud_config.json` and fill in your API keys
4. **Memory Usage**: Processing large data files may require significant memory
5. **Execution Time**: Complete analysis may take several minutes to over ten minutes, please be patient

## Troubleshooting

### Issue: Cannot Start Streamlit

**Solution**:
```bash
pip install --upgrade streamlit
```

### Issue: Module Not Found

**Solution**:
- Ensure running from project root directory
- Check Python path configuration
- Reinstall dependencies: `pip install -r requirements.txt`

### Issue: Pipeline Execution Failed

**Solution**:
- Check if CSV file format is correct
- View error messages in execution logs
- Ensure all required columns exist

### Issue: Charts Cannot Be Displayed

**Solution**:
- Ensure "Generate Plots" is selected when running analysis
- Check if PNG files exist in output directory
- Confirm matplotlib is correctly installed

## Technical Support

If you encounter issues, please:
1. Check error messages in execution logs
2. Verify if command-line version of pipeline runs normally
3. View project README.md for more information

## Changelog

- v1.0.0: Initial version, supports basic data analysis and results viewing functionality
