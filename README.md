# TrackCoach: Track Driving Data Analysis Tool

TrackCoach is a comprehensive Python toolkit for analyzing track driving data. It reads time, GPS coordinates (latitude/longitude), speed, lap numbers, and other data from cleaned CSV files, computes features such as smoothed velocity, acceleration, and curvature, detects apex positions, generates corner-by-corner data cards and event lists, and provides optional visualization capabilities.

## Features

- **Intelligent Column Recognition**: Automatically identifies time, GPS coordinates, speed, lap number, and other columns with support for multiple aliases and case/whitespace/parentheses-insensitive matching
- **Automatic Unit Conversion**: Automatically detects speed units (m/s, km/h, mph) and converts them to m/s
- **Coordinate Projection**: Projects WGS84 latitude/longitude to local Transverse Mercator coordinate system (metric)
- **Velocity Smoothing**: Uses Savitzky-Golay filter to smooth velocity data
- **Acceleration Calculation**: Prioritizes sensor data, otherwise uses numerical derivatives
- **Curvature Calculation**: Curvature computation based on arc-length parameterization for apex detection
- **Apex Detection**: Automatically detects apex positions based on curvature peaks
- **Corner Data Cards**: Generates detailed speed and acceleration metrics for each corner
- **Event Detection**: Detects heavy braking, conservative entry, and other driving events
- **Stability Analysis**: Analyzes lap-to-lap consistency and generates stability reports
- **Issue Analysis**: Identifies top driving issues based on performance and consistency metrics
- **AI Coaching Advice**: Generates personalized coaching recommendations using LLM
- **Visualization**: Generates trajectory plots, curvature plots, braking point annotations, and consistency heatmaps
- **Web UI**: Streamlit-based graphical interface for easy data analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

For Web UI usage, also install Streamlit dependencies:

```bash
cd UI
pip install -r requirements_ui.txt
```

### Configuration (Optional)

If you want to use AI coaching advice features, you need to configure API keys:

1. Copy the example configuration file:
   ```bash
   cp llamacloud_config.json.example llamacloud_config.json
   ```

2. Edit `llamacloud_config.json` and fill in your API keys:
   - `api_key`: Your LlamaCloud API key
   - `llm.api_key`: Your OpenRouter API key (or other LLM provider API key)
   - `name`, `project_name`, `organization_id`: Your LlamaCloud project details

**Note**: The `llamacloud_config.json` file is already in `.gitignore` and will not be committed to the repository. Never commit API keys to version control.

## Usage

### Command Line Interface

#### Basic Usage

```bash
python -m trackcoach.pipeline --csv path/to/data.csv --outdir output_directory
```

#### Complete Parameter Example

```bash
python -m trackcoach.pipeline \
  --csv Data/ss_v3_processed.csv \
  --outdir out \
  --smooth-win-sec 0.5 \
  --brake-thr-g -0.28 \
  --brake-onset-thr-g -0.15 \
  --min-turn-gap-m 35 \
  --apex-height-quantile 0.70 \
  --segment-window-m 60 \
  --plot true \
  --analyze-issues \
  --generate-advice
```

#### Using run_pipeline.py

You can also use the provided script:

```bash
python run_pipeline.py
```

Note: Edit `run_pipeline.py` to customize input file paths and parameters.

### Web UI

#### Start the Web Interface

**Windows:**
```bash
cd UI
run_ui.bat
```

**Linux/Mac:**
```bash
cd UI
chmod +x run_ui.sh
./run_ui.sh
```

**Manual Start:**
```bash
cd UI
streamlit run app.py
```

The web interface will automatically open at `http://localhost:8501`.

#### Web UI Features

1. **Data Analysis Page**
   - Upload CSV files
   - Configure analysis parameters
   - Run analysis with real-time logs
   - View results immediately

2. **Results Viewer Page**
   - Browse analysis outputs
   - View visualizations (track layout, consistency heatmaps, speed/braking plots)
   - Explore data tables (corner cards, events, top issues, stability reports)
   - Download coaching advice reports

3. **Model Evaluation Page**
   - View evaluation results
   - Compare different model outputs

## Command Line Arguments

### Required Arguments

- `--csv`: Input CSV file path

### Output Arguments

- `--outdir`: Output directory (default: `out/`)
- `--ref-lap`: Reference lap mode: `fastest`, `median`, or a positive integer lap number (default: `fastest`)

### Column Name Overrides (Optional)

- `--lat-col`: Latitude column name (overrides auto-detection)
- `--lon-col`: Longitude column name (overrides auto-detection)
- `--time-col`: Time column name (overrides auto-detection)
- `--speed-col`: Speed column name (overrides auto-detection)
- `--lap-col`: Lap column name (overrides auto-detection)
- `--along-col`: Longitudinal acceleration column name (overrides auto-detection)

### Speed Unit

- `--speed-unit`: Speed unit (`auto`, `ms`, `kmh`, `mph`, default: `auto`)

### Processing Parameters

- `--smooth-win-sec`: Velocity smoothing window length in seconds (default: 0.5)
- `--apex-height-quantile`: Curvature peak threshold quantile (default: 0.78)
- `--min-turn-gap-m`: Minimum gap between turns in meters (default: 65)
- `--segment-window-m`: Corner segmentation window width in meters (default: 60)
- `--apex-prom-mult`: Prominence threshold multiplier (default: 0.60)
- `--post-merge-gap-m`: Post-merge minimum gap in meters (default: 45)

### Apex Detection Parameters

- `--apex-auto-tune`: Enable auto-tuning to hit target turn count
- `--apex-scan`: Grid scan and output suggested parameter combinations
- `--apex-target-min`: Target minimum number of turns (default: 12)
- `--apex-target-max`: Target maximum number of turns (default: 16)
- `--debug-curvature`: Output curvature debug data to CSV
- `--centerline-csv`: External centerline CSV path (optional, contains x,y columns)

### Braking Parameters

- `--brake-thr-g`: Heavy brake threshold in G (default: -0.28)
- `--brake-onset-thr-g`: Brake onset threshold in G (default: -0.15)

### Relative Event Detection Parameters

- `--late-brake-thr-m`: Late brake detection threshold in meters (default: 7.0)
- `--early-brake-thr-m`: Early brake detection threshold in meters (default: 12.0)
- `--overslow-thr-ms`: Overslow detection threshold in m/s (default: 1.2)
- `--weak-exit-thr-ms`: Weak exit speed threshold in m/s (default: 1.5)
- `--weak-exit-adecel-g`: Weak exit acceleration threshold in G (default: 0.25)

### Stability Analysis Parameters

- `--stability-heatmap-agg`: Consistency heatmap aggregation method (`median`, `iqr`, `std`, default: `iqr`)
- `--no-stability`: Skip stability analysis
- `--keep-warmup-cooldown`: Keep warmup lap (lap 0) and cooldown lap (last lap)

### Issue Analysis Parameters

- `--analyze-issues`: Execute issue analysis and generate Top Issues report
- `--topk`: Number of Top Issues to report (default: 8)
- `--stability-weight`: Stability score weight (0-1, default: 0.3)

### Advice Generation Parameters

- `--generate-advice`: Generate coaching advice report
- `--advice-config`: LLM config file path (default: `llamacloud_config.json`)

**Note**: To use AI coaching advice, you must first create and configure `llamacloud_config.json` (see Configuration section above).

### Visualization

- `--plot`: Whether to generate plots (`true`, `false`, default: `false`)

### Configuration Presets

- `--preset`: Use parameter preset (`default`, `sows_track`, `high_speed_track`, `technical_track`)

### Debug

- `--debug`: Enable DEBUG logging

## Output Files

### corner_cards.csv

Corner-by-corner data cards containing:

- `lap`: Lap number
- `turn`: Turn number
- `s_apex`: Apex arc-length position (meters)
- `s_brake_onset`: Brake onset arc-length position (meters)
- `Ventry`: Entry speed (m/s)
- `Vmin`: Minimum speed in corner (m/s)
- `Vexit`: Exit speed (m/s)
- `a_long_peak`: Peak braking acceleration (m/s²)

### events.csv

Event list containing:

- `lap`: Lap number
- `turn`: Turn number
- `type`: Event type (`heavy_brake`, `conservative_entry`, `late_brake`, `early_brake`, `overslow`, `weak_exit`)
- Event-specific fields

### stability_report.csv

Lap-to-lap consistency metrics:

- `metric`: Metric name (e.g., `braking_distance`, `apex_speed`)
- `lap`: Lap number
- `value`: Metric value
- `median`: Median across all laps
- `iqr`: Interquartile range
- `std`: Standard deviation

### top_issues.csv

Top driving issues ranked by composite score:

- `turn`: Turn number
- `issue_type`: Issue category
- `severity`: Severity score
- `stability_score`: Consistency score
- `composite_score`: Overall score

### coaching_advice_report.md

AI-generated coaching advice report with personalized recommendations.

### Visualization Files (when `--plot true`)

- `xy_speed.png`: XY trajectory colored by speed
- `xy_speed_lap_*.png`: Per-lap speed trajectories
- `curvature_apex.png`: Curvature curve with apex annotations
- `track_layout_apexes.png`: Track layout with apex markers
- `xy_brakes.png`: Trajectory with heavy braking positions marked
- `xy_brakes_lap_*.png`: Per-lap braking point plots
- `consistency_heatmap_norm.png`: Normalized consistency heatmap
- `top_issues_bar.png`: Top issues bar chart

## Apex Detection and Auto-Tuning

### Overview

TrackCoach uses curvature analysis to detect apex positions. Through multi-parameter optimization, the apex count can be stably converged to a target range (default 12-16 turns).

### Core Algorithm

#### 1. Curvature Calculation

Based on arc-length parameterization:

```
κ(s) = |x'(s)y''(s) - y'(s)x''(s)| / (x'(s)² + y'(s)²)^(3/2)
```

#### 2. Peak Detection Parameters

- **Height Threshold (q)**: Curvature threshold quantile, default 0.78
  - Increase q (e.g., 0.80-0.85) → fewer apexes
  - Decrease q (e.g., 0.70-0.75) → more apexes

- **Minimum Gap (min_gap_m)**: Minimum arc-length between peaks in meters, default 65m
  - Increase gap → fewer adjacent apexes
  - Decrease gap → allow denser apexes

- **Prominence Threshold (prom_mult)**: prominence = IQR(κ) × prom_mult, default 0.60
  - Used to filter noise peaks
  - Increase prom_mult → stricter significance → fewer apexes

- **Post-Merge Gap (post_merge_gap_m)**: Minimum gap for neighbor merging in meters, default 45m
  - For peaks that are too close, keep the one with higher curvature
  - Increase merge gap → merge more peaks → fewer apexes

### Auto-Tuning Mode

Use `--apex-auto-tune` to enable auto-tuning. The system will try parameter combinations in the following order:

1. **Stage 1**: Fix `min_gap_m`, adjust `q = {0.75, 0.78, 0.80}`
2. **Stage 2**: Fix `q`, adjust `min_gap_m = {55, 65, 75}`
3. **Stage 3**: Adjust `prom_mult = {0.5, 0.6, 0.7}` and `post_merge_gap_m = {40, 45, 55}`

Stops at the first combination that satisfies the target range `[--apex-target-min, --apex-target-max]` and saves parameters to `out/apex_params.json`.

**Example:**
```bash
python -m trackcoach.pipeline \
  --csv Data/ss_v3_processed.csv \
  --outdir out \
  --apex-auto-tune \
  --apex-target-min 12 \
  --apex-target-max 16
```

### Grid Scan Mode

Use `--apex-scan` to enable grid scanning. The system will try all parameter combinations (3×3×3×3 = 81 combinations), generate `out/apex_scan.csv`, and highlight candidate combinations that hit the target range in the console.

**Example:**
```bash
python -m trackcoach.pipeline \
  --csv Data/ss_v3_processed.csv \
  --outdir out \
  --apex-scan \
  --apex-target-min 12 \
  --apex-target-max 16
```

### Debug Output

Use `--debug-curvature` to output curvature debug data to `out/curvature_debug.csv`, containing:

- `s`: Arc-length
- `kappa_s`: Curvature value
- `is_peak`: Whether it's a peak (0/1)
- `peak_s_before_merge`: Peak arc-length before merging (optional)
- `peak_s_after_merge`: Peak arc-length after merging (optional)

### Recommended Parameters

- **SOWS Track (12-16 turns)**:
  - `q=0.75-0.78`, `min_gap_m=60-70m`, `prom_mult=0.6`, `merge_gap=45m`
  
- **More Turns (>16)**:
  - Decrease `q` (0.70-0.75)
  - Decrease `min_gap_m` (50-60m)
  
- **Fewer Turns (<12)**:
  - Increase `q` (0.80-0.85)
  - Increase `min_gap_m` (75-85m)
  - Increase `prom_mult` (0.7-0.8)

### Visualization

The generated `curvature_apex.png` displays the apex count and key parameters in the title, making it easy to compare the effects of different parameter combinations.

## Common Issues

### Speed Unit Problems

If speed unit detection is incorrect:

1. Use `--speed-unit` parameter to manually specify
2. Use `--speed-col` to specify the correct speed column

### Too Many or Too Few Apexes

Adjust the following parameters:

- `--apex-height-quantile`: Curvature threshold quantile (default: 0.78)
- `--min-turn-gap-m`: Minimum arc-length between peaks (default: 65m)
- `--apex-prom-mult`: Prominence threshold multiplier (default: 0.60)
- `--post-merge-gap-m`: Post-merge minimum gap (default: 45m)

Or use `--apex-auto-tune` for auto-tuning, or `--apex-scan` for grid scanning.

### Direction Problems

If the trajectory direction is reversed:

1. Check the time order of the data
2. Adjust the data order in the CSV file

### Missing Lap Column

If the CSV doesn't have a lap number column:

- The tool will automatically create a default lap=0
- All data will be treated as a single lap
- You can use `--lap-col` to specify the correct column name

### Column Name Recognition Failure

If column names cannot be automatically recognized:

1. Check the available column names list in the logs
2. Use the corresponding `--xxx-col` parameters to manually specify
3. Check CSV file encoding (the tool automatically tries utf-8, utf-8-sig, latin-1)

## Project Structure

```
trackcoach/
  __init__.py          # Package initialization
  io_utils.py          # CSV reading, column name normalization, unit unification
  geom.py              # Coordinate projection, arc-length calculation, resampling
  features.py          # Velocity smoothing, acceleration, curvature calculation
  detect.py            # Apex detection, corner data cards, event detection
  viz.py               # Visualization
  stability.py         # Stability analysis
  analysis.py          # Issue analysis
  advice.py            # AI coaching advice generation
  export.py            # Report export (Excel, PDF)
  pipeline.py          # CLI main entry
  config.py            # Configuration management
  pb_memory.py         # Personal best memory
  comparison.py        # Lap comparison

UI/
  app.py               # Streamlit web application
  requirements_ui.txt  # UI-specific dependencies
  run_ui.bat           # Windows launcher
  run_ui.sh            # Linux/Mac launcher
  README_UI.md         # UI documentation

run_pipeline.py        # Example pipeline script
requirements.txt       # Main dependencies
README.md             # This file
```

## Algorithm Details

### Curvature Calculation

Curvature formula (based on arc-length parameterization):

```
κ(s) = |x'(s)y''(s) - y'(s)x''(s)| / (x'(s)² + y'(s)²)^(3/2)
```

### Velocity Smoothing

Uses Savitzky-Golay filter. Window length is automatically calculated based on sampling interval (approximately 0.5 seconds), polynomial order is 2.

### Longitudinal Acceleration

Priority:

1. Sensor longitudinal acceleration (G) → convert to m/s²
2. Numerical derivative: a_long = d(v_smooth)/dt

### Apex Detection

Uses `scipy.signal.find_peaks` to detect curvature peaks. Threshold is based on curvature quantile, minimum spacing is based on arc-length.

### Stability Analysis

Computes lap-to-lap consistency metrics using:
- Median values
- Interquartile Range (IQR)
- Standard deviation

Generates normalized heatmaps showing consistency across turns and laps.

### Issue Analysis

Ranks driving issues by composite score:
- Performance deviation from reference lap
- Consistency (stability) across laps
- Weighted combination: `composite = (1 - stability_weight) × performance + stability_weight × stability`

## Notes

- This project is for technical analysis and educational demonstration of track driving data
- Does not output specific speed/braking target values
- If data is insufficient (missing key sensors, etc.), the system should refuse to answer or degrade gracefully
- Ensure CSV files contain essential columns: time, GPS coordinates, speed, etc.

## License

This project is for learning and research purposes only.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.
