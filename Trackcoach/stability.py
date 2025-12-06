"""Stability analysis module: generates stability reports and consistency heatmaps focused on stabilizing PB lap performance."""

import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def build_delta_view(
    corner_cards_df: pd.DataFrame,
    pb_lap: Optional[int],
    pb_metrics: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Build delta view in memory: compute deltas of key metrics for non-PB laps relative to PB.
    
    Args:
        corner_cards_df: Corner cards (containing lap, turn, and various metrics)
        pb_lap: PB lap number
        pb_metrics: Optional PB lap per-turn data (if None, extracted from corner_cards_df)
    
    Returns:
        DataFrame view containing delta columns (does not modify original data)
    """
    df = corner_cards_df.copy()
    
    if pb_lap is None:
        logger.warning("PB lap number is None, unable to build delta view")
        return df
    
    # Extract PB lap data as benchmark
    if pb_metrics is None:
        pb_mask = df['lap'] == pb_lap
        if not pb_mask.any():
            logger.warning(f"PB lap #{pb_lap} does not exist in corner_cards")
            return df
        pb_metrics = df[pb_mask].copy()
    
    # Index PB lap data by turn
    pb_by_turn = pb_metrics.set_index('turn')
    
    # Define delta metric columns to compute
    metric_cols = {
        'delta_s_to_apex_m': 'd_delta_s',
        'Ventry': 'd_Ventry',
        'Vmin': 'd_Vmin',
        'Vexit': 'd_Vexit',
        'a_long_peak': 'd_a_long_peak',
    }
    
    # If delta_s_to_apex_m does not exist, try to compute from s_apex and s_brake_onset
    if 'delta_s_to_apex_m' not in df.columns and 's_apex' in df.columns and 's_brake_onset' in df.columns:
        df['delta_s_to_apex_m'] = np.where(
            df['s_brake_onset'].notna(),
            df['s_apex'] - df['s_brake_onset'],
            np.nan
        )
        # AlsoUpdate PB lapdata
        if 'delta_s_to_apex_m' not in pb_by_turn.columns and 's_apex' in pb_by_turn.columns and 's_brake_onset' in pb_by_turn.columns:
            pb_metrics['delta_s_to_apex_m'] = np.where(
                pb_metrics['s_brake_onset'].notna(),
                pb_metrics['s_apex'] - pb_metrics['s_brake_onset'],
                np.nan
            )
            pb_by_turn = pb_metrics.set_index('turn')
    
    # Compute delta for non-PB laps (only when delta column does not exist or needs recomputation)
    for metric_col, delta_col in metric_cols.items():
        if metric_col not in df.columns:
            logger.debug(f"Metric column {metric_col} does not exist, skipping delta computation")
            continue
        
        # Check if delta column already exists and is valid (has non-null values)
        if delta_col in df.columns:
            non_pb_df = df[df['lap'] != pb_lap] if pb_lap is not None else df
            if non_pb_df[delta_col].notna().any():
                logger.debug(f"Delta column {delta_col} already exists and is valid, skipping recomputation")
                continue
        
        # Initialize delta column to NaN
        if delta_col not in df.columns:
            df[delta_col] = np.nan
        
        # Compute delta for each row (non-PB lap)
        for idx, row in df.iterrows():
            if row['lap'] == pb_lap:
                continue  # PB lap itself does not compute delta
            
            turn = row['turn']
            if turn not in pb_by_turn.index:
                continue
            
            try:
                pb_val = pb_by_turn.loc[turn, metric_col]
                # If pb_val is Series (multiple rows), take first row
                if isinstance(pb_val, pd.Series):
                    pb_val = pb_val.iloc[0]
                
                curr_val = row[metric_col]
                
                # Only compute delta when both values are non-NaN
                if pd.notna(curr_val) and pd.notna(pb_val):
                    df.at[idx, delta_col] = curr_val - pb_val
            except (KeyError, IndexError):
                logger.debug(f"Unable to get PB value for turn {turn} metric {metric_col}, skipping")
                continue
    
    logger.debug(f"Delta view construction completed, contains {len([c for c in df.columns if c.startswith('d_')])} delta columns")
    return df


def compute_stability_report(
    corner_cards_df: pd.DataFrame,
    pb_lap: Optional[int],
    delta_metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute stability report: aggregate cross-lap deltas by turn, providing robust statistics.
    
    Args:
        corner_cards_df: Corner cards containing delta columns
        pb_lap: PB lap number (not included in aggregation)
        delta_metrics: List of delta metrics to aggregate (default: ['d_delta_s', 'd_Ventry', 'd_Vmin', 'd_Vexit', 'd_a_long_peak'])
    
    Returns:
        Stability report DataFrame, one row per turn, columns contain statistics for each delta metric
    """
    if delta_metrics is None:
        delta_metrics = ['d_delta_s', 'd_Ventry', 'd_Vmin', 'd_Vexit', 'd_a_long_peak']
    
    # Build delta view (ensure delta columns exist)
    df_delta = build_delta_view(corner_cards_df, pb_lap)
    
    # Exclude PB lap
    if pb_lap is not None:
        df_non_pb = df_delta[df_delta['lap'] != pb_lap].copy()
    else:
        df_non_pb = df_delta.copy()
    
    if len(df_non_pb) == 0:
        logger.warning("No non-PB lap data available for stability analysis")
        return pd.DataFrame()
    
    # Get all turn positions
    turns = sorted(df_non_pb['turn'].unique())
    
    # Compute stability metrics for each turn
    stability_rows = []
    
    for turn in turns:
        turn_data = df_non_pb[df_non_pb['turn'] == turn]
        
        if len(turn_data) == 0:
            continue
        
        # Compute metadata
        lap_count = turn_data['lap'].nunique()
        total_laps = df_non_pb['lap'].nunique()
        coverage = lap_count / total_laps if total_laps > 0 else 0.0
        
        row = {
            'turn': turn,
            'lap_count': lap_count,
            'total_laps': total_laps,
            'coverage': coverage,
        }
        
        # Compute robust statistics for each delta metric
        for metric in delta_metrics:
            if metric not in turn_data.columns:
                continue
            
            values = turn_data[metric].dropna()
            
            if len(values) == 0:
                # All NaN
                row[f'{metric}_median'] = np.nan
                row[f'{metric}_q25'] = np.nan
                row[f'{metric}_q75'] = np.nan
                row[f'{metric}_iqr'] = np.nan
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_count'] = 0
                # Absolute value statistics also NaN
                row[f'med_|{metric}|'] = np.nan
                row[f'iqr_|{metric}|'] = np.nan
            else:
                # Compute robust statistics (with sign)
                row[f'{metric}_median'] = float(np.median(values))
                row[f'{metric}_q25'] = float(np.percentile(values, 25))
                row[f'{metric}_q75'] = float(np.percentile(values, 75))
                row[f'{metric}_iqr'] = float(row[f'{metric}_q75'] - row[f'{metric}_q25'])
                row[f'{metric}_mean'] = float(np.mean(values))
                row[f'{metric}_std'] = float(np.std(values))
                row[f'{metric}_count'] = len(values)
                
                # Compute absolute value statistics (consistent with Top Issues)
                # median(|x|) for measuring deviation magnitude, corresponds to Top Issues med_|x|
                abs_values = np.abs(values)
                row[f'med_|{metric}|'] = float(np.median(abs_values))
                row[f'iqr_|{metric}|'] = float(np.percentile(abs_values, 75) - np.percentile(abs_values, 25))
        
        stability_rows.append(row)
    
    if not stability_rows:
        logger.warning("Stability report is empty")
        return pd.DataFrame()
    
    stability_df = pd.DataFrame(stability_rows)
    return stability_df


def compute_composite_stability_score(
    stability_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Compute composite stability score.
    
    Args:
        stability_df: Stability report DataFrame
        weights: Weight dictionary, default: {'d_delta_s_iqr': 0.5, 'd_Vmin_iqr': 0.3, 'd_a_long_peak_iqr': 0.2}
    
    Returns:
        Composite stability score Series (higher value = less stable)
    """
    if weights is None:
        weights = {'d_delta_s_iqr': 0.5, 'd_Vmin_iqr': 0.3, 'd_a_long_peak_iqr': 0.2}
    
    scores = pd.Series(0.0, index=stability_df.index)
    
    for metric, weight in weights.items():
        if metric in stability_df.columns:
            values = stability_df[metric].fillna(0)
            # Normalize to [0, 1] (by 95th percentile)
            valid_values = values[values > 0]
            if len(valid_values) > 0:
                q95 = np.percentile(valid_values, 95)
                if q95 > 0:
                    normalized = np.clip(values / q95, 0, 1)
                    scores += weight * normalized
    
    return scores


def plot_consistency_heatmap(
    stability_df: pd.DataFrame,
    output_path: Path,
    pb_lap: Optional[int],
    delta_metrics: Optional[List[str]] = None,
    agg_method: str = 'iqr',
    figsize: Tuple[float, float] = (14, 10),
    total_laps: Optional[int] = None,
    sort_by: Optional[str] = None,
    top_n: int = 3,
    min_coverage: float = 0.6,
    normalize: bool = False,
    sort_version: str = 'v1',
    export_csv: bool = True
) -> Tuple[Optional[Path], Optional[Dict]]:
    """
    Plot consistency heatmap: rows for turns, columns for key delta metrics, showing cross-lap fluctuation strength.
    
    Args:
        stability_df: Stability report DataFrame
        output_path: Output image path
        pb_lap: PB lap number (for title)
        delta_metrics: List of delta metrics to display (default: ['d_delta_s', 'd_Ventry', 'd_Vmin', 'd_a_long_peak'])
        agg_method: Aggregation method ('median' for trend offset, 'iqr' for dispersion)
        figsize: Figure size (default (14, 10), ensure 1600px+)
        total_laps: Total number of laps in statistics (for title)
        sort_by: Sort criterion (e.g., 'd_delta_s_iqr' or None for no sorting)
        top_n: Annotate top N least stable turns
        min_coverage: Minimum coverage threshold (below this shows N/A)
        normalize: Whether to normalize within columns (each column normalized by 95th percentile)
        sort_version: Sort version ('v1'=natural order, 'v2'=sorted by composite stability score)
        export_csv: Whether to export support table CSV
    
    Returns:
        (csv_path, norm_meta) tuple, norm_meta contains normalization metadata
    """
    from datetime import datetime
    from matplotlib.patches import Rectangle, FancyBboxPatch
    
    # Unit mapping
    unit_map = {
        'd_delta_s': '(m)',
        'd_Ventry': '(m/s)',
        'd_Vmin': '(m/s)',
        'd_Vexit': '(m/s)',
        'd_a_long_peak': '(g)'
    }
    
    # Metric name mapping (for labels)
    metric_name_map = {
        'd_delta_s': 's',
        'd_Ventry': 'Ventry',
        'd_Vmin': 'Vmin',
        'd_Vexit': 'Vexit',
        'd_a_long_peak': 'a_long_peak'
    }
    
    # Decimal places mapping (Δs 1 decimal, others 2 decimals)
    decimal_map = {
        'd_delta_s': 1,
        'd_Ventry': 2,
        'd_Vmin': 2,
        'd_Vexit': 2,
        'd_a_long_peak': 2
    }
    
    if delta_metrics is None:
        delta_metrics = ['d_delta_s', 'd_Ventry', 'd_Vmin', 'd_a_long_peak']
    
    if stability_df.empty:
        logger.warning("Stability report is empty, unable to generate heatmap")
        return None, None
    
    # Copy DataFrame to avoid modifying original data
    df = stability_df.copy()
    
    # Sorting: choose sorting method according to version
    if sort_version == 'v2':
        # v2: Sort by composite stability score
        composite_score = compute_composite_stability_score(df)
        df['_composite_score'] = composite_score
        df = df.sort_values(by='_composite_score', ascending=False, na_position='last').reset_index(drop=True)
        sort_desc = "Sorted by composite stability score"
    elif sort_by and sort_by in df.columns:
        # v1 or specified sort criterion: sort by specified column
        df = df.sort_values(by=sort_by, ascending=False, na_position='last').reset_index(drop=True)
        sort_desc = f"Sorted by {sort_by}"
    else:
        # v1: Natural order
        sort_desc = "Natural order (v1)"
    
    # Prepare heatmap data
    heatmap_data = []
    raw_data = []  # Save original values for export
    metric_labels = []
    norm_meta = {}  # Normalization metadata
    coverage_col = df['coverage'].values if 'coverage' in df.columns else None
    
    for metric in delta_metrics:
        # Select column according to aggregation method
        if agg_method == 'median':
            col_name = f'{metric}_median'
            label_suffix = '(median)'
        elif agg_method == 'iqr':
            col_name = f'{metric}_iqr'
            label_suffix = '(IQR)'
        elif agg_method == 'std':
            col_name = f'{metric}_std'
            label_suffix = '(std)'
        else:
            logger.warning(f"Unknown aggregation method {agg_method}, using IQR")
            col_name = f'{metric}_iqr'
            label_suffix = '(IQR)'
        
        if col_name not in df.columns:
            logger.debug(f"Column {col_name} does not exist, skipping metric {metric}")
            continue
        
        # Extract data
        values = df[col_name].values.copy()
        raw_values = values.copy()
        
        # Apply minimum coverage threshold: set to NaN if coverage insufficient
        low_coverage_mask = None
        if coverage_col is not None:
            low_coverage_mask = coverage_col < min_coverage
            values[low_coverage_mask] = np.nan
        
        # Save original values
        raw_data.append(raw_values)
        
        # Normalize within column (each column normalized by 95th percentile)
        if normalize:
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                q95 = np.percentile(valid_values, 95)
                vmin = np.min(valid_values) if len(valid_values) > 0 else 0
                
                # Normalize: [0, q95] -> [0, 1], clipped to 1
                if q95 > 0:
                    values_norm = np.clip(values / q95, 0, 1)
                else:
                    values_norm = np.zeros_like(values)
                
                # Save normalization metadata
                norm_meta[metric] = {
                    'min': float(vmin),
                    'q95': float(q95),
                    'method': 'per-column min-max on [0, q95], clipped to 1'
                }
                
                values = values_norm
            else:
                norm_meta[metric] = {
                    'min': None,
                    'q95': None,
                    'method': 'per-column min-max on [0, q95], clipped to 1 (no valid data)'
                }
        else:
            # Original values: record statistics
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                norm_meta[metric] = {
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'mean': float(np.mean(valid_values)),
                    'q95': float(np.percentile(valid_values, 95))
                }
        
        # Build label (including unit)
        unit = unit_map.get(metric, '')
        metric_name = metric_name_map.get(metric, metric[2:] if metric.startswith('d_') else metric)
        if normalize:
            label = f'Δ{metric_name}\n{label_suffix}'
        else:
            label = f'Δ{metric_name}{unit}\n{label_suffix}'
        
        heatmap_data.append(values)
        metric_labels.append(label)
    
    if not heatmap_data:
        logger.warning("No columns available for heatmap")
        return None, None
    
    heatmap_array = np.array(heatmap_data).T  # Transpose: rows for turns, columns for metrics
    raw_array = np.array(raw_data).T  # Original value array
    turns = df['turn'].values
    
    # Identify top N least stable turns (based on first metric, usually d_delta_s_iqr)
    top_turns = []
    top_indices = []
    if len(heatmap_data) > 0:
        first_col_values = heatmap_array[:, 0]
        valid_indices = ~np.isnan(first_col_values)
        if valid_indices.sum() > 0:
            valid_turns_with_values = [
                (i, first_col_values[i]) 
                for i in range(len(turns)) 
                if valid_indices[i]
            ]
            valid_turns_with_values.sort(key=lambda x: x[1], reverse=True)
            top_turns = [turns[idx] for idx, _ in valid_turns_with_values[:top_n]]
            top_indices = [idx for idx, _ in valid_turns_with_values[:top_n]]
    
    # Create figure (ensure 1600px+ resolution)
    # figsize=(14, 10) at 300dpi = 4200x3000px, sufficient for printing
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Prepare annotation text (set decimal places according to metric)
    annot_array = []
    for i in range(len(turns)):
        row_annot = []
        for j, metric in enumerate(delta_metrics):
            if j >= len(heatmap_data):
                continue
            
            val = heatmap_array[i, j]
            raw_val = raw_array[i, j] if i < len(raw_array) and j < len(raw_data) else np.nan
            
            if np.isnan(val):
                # Check coverage
                if coverage_col is not None and coverage_col[i] < min_coverage:
                    row_annot.append("N/A")
                else:
                    row_annot.append("N/A")
            else:
                # Select decimal places according to metric
                decimals = decimal_map.get(metric, 2)
                if normalize:
                    # Display normalized value
                    row_annot.append(f"{val:.2f}")
                else:
                    # Display original value (Δs 1 decimal, others 2 decimals)
                    row_annot.append(f"{raw_val:.{decimals}f}")
        annot_array.append(row_annot)
    annot_array = np.array(annot_array, dtype=object)
    
    # Create light gray mask for insufficient coverage cells
    coverage_mask = np.zeros_like(heatmap_array, dtype=bool)
    if coverage_col is not None:
        for i in range(len(turns)):
            if coverage_col[i] < min_coverage:
                coverage_mask[i, :] = True
    
    # Combine mask (NaN + insufficient coverage)
    combined_mask = np.isnan(heatmap_array) | coverage_mask
    
    # Use seaborn to plot heatmap
    heatmap = sns.heatmap(
        heatmap_array,
        xticklabels=metric_labels,
        yticklabels=[f'Turn {int(t)}' for t in turns],
        annot=annot_array,
        fmt='',
        cmap='YlOrRd',
        cbar_kws={
            'label': f'{agg_method.upper()} (normalized)' if normalize else f'{agg_method.upper()} (raw values)',
            'ticks': [0, 0.25, 0.5, 0.75, 1.0] if normalize else None,
            'format': '%.2f' if normalize else None
        },
        ax=ax,
        linewidths=0.5,
        linecolor='gray',
        mask=combined_mask,
        cbar=True,
        vmin=0 if normalize else None,
        vmax=1 if normalize else None
    )
    
    # Overlay light gray for insufficient coverage cells
    if coverage_mask.any():
        for i in range(len(turns)):
            for j in range(len(metric_labels)):
                if coverage_mask[i, j]:
                    rect = plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        facecolor='lightgray',
                        alpha=0.5,
                        zorder=1
                    )
                    ax.add_patch(rect)
    
    # Highlight top N turns (add border and light shadow)
    for i, turn in enumerate(turns):
        if turn in top_turns:
            rank = top_turns.index(turn) + 1
            # Add border and light shadow to entire row
            for j in range(len(metric_labels)):
                # Light shadow
                shadow = FancyBboxPatch(
                    (j - 0.02, i - 0.02),
                    1.04,
                    1.04,
                    boxstyle="round,pad=0.02",
                    facecolor='blue',
                    alpha=0.15,
                    edgecolor='none',
                    zorder=2
                )
                ax.add_patch(shadow)
                
                # Border
                rect = Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=False,
                    edgecolor='blue',
                    linewidth=2,
                    linestyle='--',
                    zorder=3
                )
                ax.add_patch(rect)
            
            # Add asterisk annotation
            ax.text(
                -0.5,
                i + 0.5,
                f'*{rank}',
                ha='right',
                va='center',
                fontsize=14,
                fontweight='bold',
                color='blue',
                zorder=4
            )
    
    # Setup title (include more information)
    pb_label = f"PB Lap #{pb_lap}" if pb_lap is not None else "PB Lap (Unknown)"
    date_str = datetime.now().strftime("%Y-%m-%d")
    total_laps_str = f", {total_laps} laps" if total_laps is not None else ""
    
    # Build title
    title_parts = [
        f'Consistency Heatmap - {pb_label}',
        f'Aggregation: {agg_method.upper()} | Date: {date_str}{total_laps_str}',
        f'Sort: {sort_desc} | Display threshold: coverage ≥ {min_coverage:.0%}'
    ]
    
    if normalize:
        title_parts.append('Normalization: per-column min-max on [0, q95], clipped to 1')
    
    title = '\n'.join(title_parts)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    
    # Adjust layout
    ax.set_xlabel('Delta Metrics', fontsize=12)
    ax.set_ylabel('Turn', fontsize=12)
    
    # Add coverage information to Y axis labels (if available)
    if coverage_col is not None:
        ytick_labels = []
        for i, turn in enumerate(turns):
            coverage_pct = coverage_col[i] * 100 if coverage_col[i] is not None else 0
            ytick_labels.append(f'T{int(turn)}\n({coverage_pct:.0f}%)')
        ax.set_yticklabels(ytick_labels, fontsize=9)
    
    plt.tight_layout()
    
    # Save image (ensure 1600px+ resolution)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Export support table CSV
    csv_path = None
    if export_csv:
        csv_filename = 'consistency_values_norm.csv' if normalize else 'consistency_values_raw.csv'
        csv_path = output_path.parent / csv_filename
        
        # Build export DataFrame (remove newlines from column names for CSV)
        export_data = {'turn': turns}
        
        # Create CSV-friendly column names (no newlines)
        csv_col_names = []
        for j, metric in enumerate(delta_metrics):
            if j >= len(metric_labels):
                continue
            
            # Get original label
            label = metric_labels[j]
            # CSV column name: remove newlines, replace with space
            csv_col_name = label.replace('\n', ' ').strip()
            csv_col_names.append(csv_col_name)
            
            if normalize:
                # Normalized values
                export_data[csv_col_name] = heatmap_array[:, j]
            else:
                # Original values
                export_data[csv_col_name] = raw_array[:, j] if j < len(raw_data) else heatmap_array[:, j]
        
        # Add coverage
        if coverage_col is not None:
            export_data['coverage'] = coverage_col
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(csv_path, index=False)
        logger.info(f"Support table saved: {csv_path}")
    
    # Save normalization metadata (if normalized version)
    if normalize and norm_meta:
        meta_path = output_path.parent / 'consistency_norm_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(norm_meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Normalization metadata saved: {meta_path}")
    
    logger.info(f"Consistency heatmap saved: {output_path}")
    
    return csv_path, norm_meta if normalize else None


def generate_stability_artifacts(
    corner_cards_df: pd.DataFrame,
    pb_lap: Optional[int],
    output_dir: Path,
    delta_metrics: Optional[List[str]] = None,
    heatmap_metrics: Optional[List[str]] = None,
    heatmap_agg: str = 'iqr'
) -> Tuple[Path, Path]:
    """
    Generate stability analysis artifacts: report and heatmap.
    
    Args:
        corner_cards_df: Corner cards
        pb_lap: PB lap number
        output_dir: Output directory
        delta_metrics: Delta metrics list (for report)
        heatmap_metrics: Delta metrics list to display in heatmap (default: first 4)
        heatmap_agg: Heatmap aggregation method ('median' or 'iqr')
    
    Returns:
        (stability_report_path, heatmap_path) tuple
    """
    logger.info("=" * 60)
    logger.info("Generating stability analysis artifacts")
    logger.info("=" * 60)
    
    # 1. Generate stability report
    logger.info("Computing stability report...")
    stability_df = compute_stability_report(corner_cards_df, pb_lap, delta_metrics)
    
    if stability_df.empty:
        logger.error("Stability report is empty, please check input data")
        raise ValueError("Stability report generation failed: data is empty")
    
    # Check if there are non-zero statistics
    numeric_cols = stability_df.select_dtypes(include=[np.number]).columns
    has_nonzero = False
    for col in numeric_cols:
        if col != 'turn' and col != 'lap_count' and col != 'total_laps':
            if stability_df[col].notna().any() and (stability_df[col].abs() > 1e-10).any():
                has_nonzero = True
                break
    
    if not has_nonzero:
        logger.warning("All delta statistics are 0 or NaN, please check if PB lap is included in aggregation")
    
    # Save report
    report_path = output_dir / 'stability_report.csv'
    stability_df.to_csv(report_path, index=False)
    logger.info(f"Stability report saved: {report_path} ({len(stability_df)} turns)")
    
    # 2. Generate consistency heatmap (raw and normalized versions, each with two sort versions)
    logger.info("Generating consistency heatmap...")
    
    total_turns = len(stability_df)
    total_laps = stability_df['total_laps'].max() if 'total_laps' in stability_df.columns else 0
    avg_coverage = stability_df['coverage'].mean() if 'coverage' in stability_df.columns else 0.0
    
    # Determine sort criterion (prioritize d_delta_s_iqr, if not available use first available metric)
    sort_by_col = None
    if 'd_delta_s_iqr' in stability_df.columns:
        sort_by_col = 'd_delta_s_iqr'
    elif heatmap_metrics and len(heatmap_metrics) > 0:
        first_metric = heatmap_metrics[0]
        if heatmap_agg == 'iqr':
            sort_by_col = f'{first_metric}_iqr'
        elif heatmap_agg == 'median':
            sort_by_col = f'{first_metric}_median'
    
    # v1: Natural order (raw values)
    heatmap_path_v1 = output_dir / 'consistency_heatmap_v1.png'
    csv_path_v1, _ = plot_consistency_heatmap(
        stability_df,
        heatmap_path_v1,
        pb_lap,
        delta_metrics=heatmap_metrics,
        agg_method=heatmap_agg,
        total_laps=total_laps,
        sort_by=None,
        top_n=3,
        min_coverage=0.6,
        normalize=False,
        sort_version='v1',
        export_csv=True
    )
    
    # v2: Sorted by composite stability score (raw values)
    heatmap_path_v2 = output_dir / 'consistency_heatmap_v2.png'
    csv_path_v2, _ = plot_consistency_heatmap(
        stability_df,
        heatmap_path_v2,
        pb_lap,
        delta_metrics=heatmap_metrics,
        agg_method=heatmap_agg,
        total_laps=total_laps,
        sort_by=None,
        top_n=3,
        min_coverage=0.6,
        normalize=False,
        sort_version='v2',
        export_csv=True
    )
    
    # v1: Natural order (normalized)
    heatmap_norm_path_v1 = output_dir / 'consistency_heatmap_norm_v1.png'
    csv_norm_path_v1, norm_meta_v1 = plot_consistency_heatmap(
        stability_df,
        heatmap_norm_path_v1,
        pb_lap,
        delta_metrics=heatmap_metrics,
        agg_method=heatmap_agg,
        total_laps=total_laps,
        sort_by=None,
        top_n=3,
        min_coverage=0.6,
        normalize=True,
        sort_version='v1',
        export_csv=True
    )
    
    # v2: Sorted by composite stability score (normalized)
    heatmap_norm_path_v2 = output_dir / 'consistency_heatmap_norm_v2.png'
    csv_norm_path_v2, norm_meta_v2 = plot_consistency_heatmap(
        stability_df,
        heatmap_norm_path_v2,
        pb_lap,
        delta_metrics=heatmap_metrics,
        agg_method=heatmap_agg,
        total_laps=total_laps,
        sort_by=None,
        top_n=3,
        min_coverage=0.6,
        normalize=True,
        sort_version='v2',
        export_csv=True
    )
    
    # For backward compatibility, also generate files without version number (use v2 sorting)
    heatmap_path = output_dir / 'consistency_heatmap.png'
    heatmap_norm_path = output_dir / 'consistency_heatmap_norm.png'
    if heatmap_path_v2.exists():
        import shutil
        shutil.copy2(heatmap_path_v2, heatmap_path)
        shutil.copy2(heatmap_norm_path_v2, heatmap_norm_path)
    
    # Save normalization metadata (use v2 metadata)
    if norm_meta_v2:
        meta_path = output_dir / 'consistency_norm_meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(norm_meta_v2, f, indent=2, ensure_ascii=False)
        logger.info(f"Normalization metadata saved: {meta_path}")
    
    # 3. Output summary information
    logger.info("=" * 60)
    logger.info("Stability Analysis Summary")
    logger.info("=" * 60)
    logger.info(f"PB lap number: #{pb_lap if pb_lap is not None else 'N/A'}")
    logger.info(f"Number of laps in statistics: {total_laps}")
    logger.info(f"Number of turns: {total_turns}")
    logger.info(f"Average coverage: {avg_coverage:.1%}")
    logger.info(f"Heatmap aggregation method: {heatmap_agg.upper()}")
    if heatmap_metrics:
        logger.info(f"Heatmap metrics: {', '.join(heatmap_metrics)}")
    logger.info(f"Output files:")
    logger.info(f"  - {report_path}")
    logger.info(f"  - {heatmap_path_v1} (raw values, v1 natural order)")
    logger.info(f"  - {heatmap_path_v2} (raw values, v2 composite score sorted)")
    logger.info(f"  - {heatmap_norm_path_v1} (normalized, v1 natural order)")
    logger.info(f"  - {heatmap_norm_path_v2} (normalized, v2 composite score sorted)")
    if csv_path_v1:
        logger.info(f"  - {csv_path_v1} (support table, raw values)")
    if csv_path_v2:
        logger.info(f"  - {csv_path_v2} (support table, raw values)")
    if csv_norm_path_v1:
        logger.info(f"  - {csv_norm_path_v1} (support table, normalized)")
    if csv_norm_path_v2:
        logger.info(f"  - {csv_norm_path_v2} (support table, normalized)")
    meta_path = output_dir / 'consistency_norm_meta.json'
    if meta_path.exists():
        logger.info(f"  - {meta_path} (normalization metadata)")
    logger.info("=" * 60)
    
    return report_path, heatmap_path

