"""Visualization module."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_xy_speed(
    df: pd.DataFrame,
    output_path: Path,
    x_col: str = 'x',
    y_col: str = 'y',
    v_col: str = 'v_smooth',
    benchmark_lap: Optional[int] = None,
    lap_col: str = 'lap'
) -> None:
    """
    Plot XY trajectory colored by speed.

    Args:
        df: DataFrame
        output_path: Output path
        x_col: X column name
        y_col: Y column name
        v_col: Velocity column name
        benchmark_lap: Benchmark lap number (highlighted, default None)
        lap_col: Lap number column name
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if x_col not in df.columns or y_col not in df.columns or v_col not in df.columns:
        logger.warning(f"Missing columns for xy_speed plot, skipping")
        return
    
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df[v_col],
        cmap='viridis',
        s=1,
        alpha=0.6
    )
    highlighted_benchmark = False
    if benchmark_lap is not None and lap_col in df.columns:
        benchmark_df = df[df[lap_col] == benchmark_lap].copy()
        if not benchmark_df.empty:
            if 's' in benchmark_df.columns:
                benchmark_df.sort_values(by='s', inplace=True)
            elif 't_s' in benchmark_df.columns:
                benchmark_df.sort_values(by='t_s', inplace=True)
            ax.plot(
                benchmark_df[x_col],
                benchmark_df[y_col],
                color='red',
                linewidth=2.5,
                alpha=0.9,
                label='Benchmark lap'
            )
            highlighted_benchmark = True
        else:
            logger.warning(f"Benchmark lap #{benchmark_lap} not found for xy_speed plot highlighting")
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Track Trajectory Colored by Speed', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Speed (m/s)', fontsize=10)
    
    if highlighted_benchmark:
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved xy_speed plot to {output_path}")


def plot_curvature_apex(
    s: np.ndarray,
    kappa: np.ndarray,
    apex_s: np.ndarray,
    output_path: Path,
    apex_count: Optional[int] = None,
    params: Optional[dict] = None
) -> None:
    """
    Plot curvature curve with apex annotations.

    Args:
        s: Arc length array
        kappa: Curvature array
        apex_s: Apex arc length positions
        output_path: Output path
        apex_count: Apex count (optional)
        params: Parameter dictionary (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(s, kappa, 'b-', linewidth=1.5, label='Curvature κ(s)')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Annotate apexes
    for apex in apex_s:
        # Find corresponding curvature value
        idx = np.argmin(np.abs(s - apex))
        kappa_val = kappa[idx]
        ax.plot(apex, kappa_val, 'ro', markersize=8, zorder=5)
        ax.annotate(
            f'Apex {np.where(apex_s == apex)[0][0] + 1}',
            xy=(apex, kappa_val),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    
    ax.set_xlabel('Arc Length s (m)', fontsize=12)
    ax.set_ylabel('Curvature κ(s)', fontsize=12)
    
    # Build title
    title = 'Curvature Profile with Apex Detection'
    if apex_count is not None:
        title += f' (Apex count: {apex_count}'
        if params:
            title += f" | q={params.get('q', 'N/A'):.2f}, gap={params.get('min_gap_m', 'N/A'):.1f}m"
        title += ')'
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved curvature_apex plot to {output_path}")


def plot_xy_brakes(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    corner_cards_df: pd.DataFrame,
    output_path: Path,
    x_col: str = 'x',
    y_col: str = 'y',
    s_col: str = 's',
    benchmark_lap: Optional[int] = None,
    lap_col: str = 'lap'
) -> None:
    """
    Mark heavy braking positions on trajectory.

    Args:
        df: DataFrame
        events_df: Events DataFrame
        corner_cards_df: Corner cards
        output_path: Output path
        x_col: X column name
        y_col: Y column name
        s_col: Arc length column name
        benchmark_lap: Benchmark lap number (highlighted, default None)
        lap_col: Lap number column name
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if x_col not in df.columns or y_col not in df.columns:
        logger.warning(f"Missing columns for xy_brakes plot, skipping")
        return
    
    # Plot trajectory
    ax.plot(df[x_col], df[y_col], 'k-', linewidth=0.5, alpha=0.3, label='Track')
    
    highlighted_benchmark = False
    if benchmark_lap is not None and lap_col in df.columns:
        benchmark_df = df[df[lap_col] == benchmark_lap].copy()
        if not benchmark_df.empty:
            if 's' in benchmark_df.columns:
                benchmark_df.sort_values(by='s', inplace=True)
            elif 't_s' in benchmark_df.columns:
                benchmark_df.sort_values(by='t_s', inplace=True)
            ax.plot(
                benchmark_df[x_col],
                benchmark_df[y_col],
                color='red',
                linewidth=2.5,
                alpha=0.9,
                label='Benchmark lap'
            )
            highlighted_benchmark = True
        else:
            logger.warning(f"Benchmark lap #{benchmark_lap} not found for xy_brakes plot highlighting")
    
    # Annotate heavy braking events
    heavy_brakes = events_df[events_df['type'] == 'heavy_brake']
    
    if len(heavy_brakes) > 0:
        for _, event in heavy_brakes.iterrows():
            lap = event['lap']
            turn = event['turn']
            
            # Find corresponding corner card
            card = corner_cards_df[
                (corner_cards_df['lap'] == lap) &
                (corner_cards_df['turn'] == turn)
            ]
            
            if len(card) > 0:
                s_apex = card['s_apex'].iloc[0]
                s_brake = card['s_brake_onset'].iloc[0] if pd.notna(card['s_brake_onset'].iloc[0]) else s_apex
                
                # Find corresponding xy coordinates
                apex_idx = np.argmin(np.abs(df[s_col].values - s_apex))
                brake_idx = np.argmin(np.abs(df[s_col].values - s_brake)) if pd.notna(s_brake) else apex_idx
                
                x_apex = df[x_col].iloc[apex_idx]
                y_apex = df[y_col].iloc[apex_idx]
                x_brake = df[x_col].iloc[brake_idx]
                y_brake = df[y_col].iloc[brake_idx]
                
                # Plot apex
                ax.plot(x_apex, y_apex, 'go', markersize=8, label='Apex' if turn == heavy_brakes.iloc[0]['turn'] else '')
                
                # Plot brake onset point
                ax.plot(x_brake, y_brake, 'rs', markersize=8, label='Heavy Brake' if turn == heavy_brakes.iloc[0]['turn'] else '')
                
                # Annotate
                if pd.notna(event['delta_s_to_apex_m']):
                    ax.annotate(
                        f'T{turn}\nΔs={event["delta_s_to_apex_m"]:.1f}m',
                        xy=(x_brake, y_brake),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                    )
    else:
        logger.info("No heavy brake events to plot")
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Track Trajectory with Heavy Brake Locations', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved xy_brakes plot to {output_path}")


def plot_xy_speed_by_lap(
    df: pd.DataFrame,
    output_dir: Path,
    lap_col: str = 'lap',
    x_col: str = 'x',
    y_col: str = 'y',
    v_col: str = 'v_smooth',
    benchmark_lap: Optional[int] = None
) -> None:
    """
    Plot XY trajectory colored by speed for each lap separately.

    Args:
        df: DataFrame
        output_dir: Output directory
        lap_col: Lap number column name
        x_col: X column name
        y_col: Y column name
        v_col: Velocity column name
        benchmark_lap: Benchmark lap number (highlighted, default None)
    """
    if lap_col not in df.columns:
        logger.warning(f"Lap column '{lap_col}' not found, skipping per-lap speed plots")
        return
    
    if x_col not in df.columns or y_col not in df.columns or v_col not in df.columns:
        logger.warning(f"Missing columns for per-lap xy_speed plot, skipping")
        return
    
    unique_laps = sorted(df[lap_col].unique())
    
    for lap_num in unique_laps:
        lap_df = df[df[lap_col] == lap_num].copy()
        
        if len(lap_df) < 10:  # Skip too short laps
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        is_benchmark = benchmark_lap is not None and lap_num == benchmark_lap
        
        scatter = ax.scatter(
            lap_df[x_col],
            lap_df[y_col],
            c=lap_df[v_col],
            cmap='viridis',
            s=4 if is_benchmark else 2,
            alpha=0.85 if is_benchmark else 0.7,
            linewidths=0.3 if is_benchmark else 0.0,
            edgecolors='k' if is_benchmark else 'none'
        )
        
        if is_benchmark:
            line_df = lap_df.sort_values(by='s') if 's' in lap_df.columns else lap_df.sort_values(by='t_s') if 't_s' in lap_df.columns else lap_df
            ax.plot(
                line_df[x_col],
                line_df[y_col],
                color='red',
                linewidth=2.5,
                alpha=0.9,
                label='Benchmark lap'
            )
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        title = f'Track Trajectory Colored by Speed - Lap {lap_num}'
        if is_benchmark:
            title += ' (Benchmark)'
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Speed (m/s)', fontsize=10)
        
        if is_benchmark:
            ax.legend(loc='best')
        
        plt.tight_layout()
        output_path = output_dir / f'xy_speed_lap_{lap_num}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved per-lap xy_speed plot to {output_path}")


def plot_xy_brakes_by_lap(
    df: pd.DataFrame,
    events_df: pd.DataFrame,
    corner_cards_df: pd.DataFrame,
    output_dir: Path,
    lap_col: str = 'lap',
    x_col: str = 'x',
    y_col: str = 'y',
    s_col: str = 's',
    benchmark_lap: Optional[int] = None
) -> None:
    """
    Mark heavy braking positions on trajectory for each lap separately.

    Args:
        df: DataFrame
        events_df: Events DataFrame
        corner_cards_df: Corner cards
        output_dir: Output directory
        lap_col: Lap number column name
        x_col: X column name
        y_col: Y column name
        s_col: Arc length column name
        benchmark_lap: Benchmark lap number (highlighted, default None)
    """
    if lap_col not in df.columns:
        logger.warning(f"Lap column '{lap_col}' not found, skipping per-lap brakes plots")
        return
    
    if x_col not in df.columns or y_col not in df.columns:
        logger.warning(f"Missing columns for per-lap xy_brakes plot, skipping")
        return
    
    unique_laps = sorted(df[lap_col].unique())
    
    for lap_num in unique_laps:
        lap_df = df[df[lap_col] == lap_num].copy()
        
        if len(lap_df) < 10:  # Skip too short laps
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot this lap trajectory
        is_benchmark = benchmark_lap is not None and lap_num == benchmark_lap
        line_color = 'red' if is_benchmark else 'k'
        line_width = 2.5 if is_benchmark else 1.0
        line_alpha = 0.9 if is_benchmark else 0.5
        ax.plot(
            lap_df[x_col],
            lap_df[y_col],
            color=line_color,
            linewidth=line_width,
            alpha=line_alpha,
            label='Benchmark lap' if is_benchmark else 'Track'
        )
        
        # Find heavy braking events for this lap (ensure type matching)
        lap_events = events_df[pd.notna(events_df['lap']) & (events_df['lap'].astype(int) == lap_num)]
        heavy_brakes = lap_events[lap_events['type'] == 'heavy_brake']
        
        if len(heavy_brakes) > 0:
            for _, event in heavy_brakes.iterrows():
                turn = event['turn']
                
                # Find corresponding corner card
                card = corner_cards_df[
                    (corner_cards_df['lap'] == lap_num) &
                    (corner_cards_df['turn'] == turn)
                ]
                
                if len(card) > 0:
                    s_apex = card['s_apex'].iloc[0]
                    s_brake = card['s_brake_onset'].iloc[0] if pd.notna(card['s_brake_onset'].iloc[0]) else s_apex
                    
                    # Find corresponding xy coordinates (within this lap)
                    apex_idx = np.argmin(np.abs(lap_df[s_col].values - s_apex))
                    brake_idx = np.argmin(np.abs(lap_df[s_col].values - s_brake)) if pd.notna(s_brake) else apex_idx
                    
                    x_apex = lap_df[x_col].iloc[apex_idx]
                    y_apex = lap_df[y_col].iloc[apex_idx]
                    x_brake = lap_df[x_col].iloc[brake_idx]
                    y_brake = lap_df[y_col].iloc[brake_idx]
                    
                    # Plot apex
                    ax.plot(x_apex, y_apex, 'go', markersize=10, 
                           label='Apex' if turn == heavy_brakes.iloc[0]['turn'] else '')
                    
                    # Plot brake onset point
                    ax.plot(x_brake, y_brake, 'rs', markersize=10, 
                           label='Heavy Brake' if turn == heavy_brakes.iloc[0]['turn'] else '')
                    
                    # Annotate
                    if pd.notna(event.get('delta_s_to_apex_m')):
                        ax.annotate(
                            f'T{turn}\nΔs={event["delta_s_to_apex_m"]:.1f}m',
                            xy=(x_brake, y_brake),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                        )
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        title = f'Track Trajectory with Heavy Brake Locations - Lap {lap_num}'
        if is_benchmark:
            title += ' (Benchmark)'
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        output_path = output_dir / f'xy_brakes_lap_{lap_num}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved per-lap xy_brakes plot to {output_path}")


def plot_track_with_apexes(
    df: pd.DataFrame,
    apex_s: np.ndarray,
    output_path: Path,
    lap: Optional[int] = None,
    x_col: str = 'x',
    y_col: str = 'y',
    s_col: str = 's',
    lap_col: str = 'lap',
    apex_count: Optional[int] = None,
    params: Optional[dict] = None,
    benchmark_lap: Optional[int] = None
) -> None:
    """
    Plot track layout and annotate all apex positions.

    Args:
        df: DataFrame
        apex_s: Apex arc length position array
        output_path: Output path
        lap: Lap number (if None, use reference lap or all data)
        x_col: X column name
        y_col: Y column name
        s_col: Arc length column name
        lap_col: Lap number column name
        apex_count: Apex count (optional)
        params: Parameter dictionary (optional)
        benchmark_lap: Benchmark lap number (highlighted, default None)
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if x_col not in df.columns or y_col not in df.columns:
        logger.warning(f"Missing columns for track plot, skipping")
        return
    
    # Select data
    if lap is not None and lap_col in df.columns:
        plot_df = df[df[lap_col] == lap].copy()
        title_suffix = f" - Lap {lap}"
    else:
        plot_df = df.copy()
        title_suffix = " - Reference Lap"
    
    if len(plot_df) == 0:
        logger.warning(f"No data found for lap {lap}, skipping")
        return
    
    is_direct_benchmark = benchmark_lap is not None and lap is not None and benchmark_lap == lap
    line_color = 'red' if is_direct_benchmark else 'k'
    line_width = 2.5 if is_direct_benchmark else 2
    line_alpha = 0.9 if is_direct_benchmark else 0.7
    
    # Plot track route
    ax.plot(
        plot_df[x_col],
        plot_df[y_col],
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        label='Benchmark lap' if is_direct_benchmark else 'Track'
    )
    
    # If current plot is not benchmark lap but has benchmark lap, overlay highlight
    if benchmark_lap is not None and not is_direct_benchmark and lap_col in df.columns:
        benchmark_df = df[df[lap_col] == benchmark_lap].copy()
        if not benchmark_df.empty:
            benchmark_df = benchmark_df.sort_values(by=s_col) if s_col in benchmark_df.columns else benchmark_df
            ax.plot(
                benchmark_df[x_col],
                benchmark_df[y_col],
                color='red',
                linewidth=2.5,
                alpha=0.9,
                label='Benchmark lap'
            )
        else:
            logger.warning(f"Benchmark lap #{benchmark_lap} not found for track layout highlighting")
    
    # Annotate apexes
    apex_points = []
    for idx, apex_s_val in enumerate(apex_s):
        # Find corresponding xy coordinates
        apex_idx = np.argmin(np.abs(plot_df[s_col].values - apex_s_val))
        x_apex = plot_df[x_col].iloc[apex_idx]
        y_apex = plot_df[y_col].iloc[apex_idx]
        apex_points.append((x_apex, y_apex, idx + 1))
    
    # Plot apex points
    for x_apex, y_apex, turn_num in apex_points:
        ax.plot(x_apex, y_apex, 'ro', markersize=12, zorder=5, 
               label='Apex' if turn_num == 1 else '')
        
        # Annotate turn number
        ax.annotate(
            f'T{turn_num}',
            xy=(x_apex, y_apex),
            xytext=(0, 15),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=1.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=1.5)
        )
    
    # Annotate start/finish
    start_x = plot_df[x_col].iloc[0]
    start_y = plot_df[y_col].iloc[0]
    ax.plot(start_x, start_y, 'gs', markersize=15, zorder=5, label='Start/Finish')
    ax.annotate(
        'S/F',
        xy=(start_x, start_y),
        xytext=(0, -20),
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.8),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green', lw=1.5)
    )
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    
    # Build title
    title = f'Track Layout with Apex Locations{title_suffix}'
    if apex_count is not None:
        title += f' ({apex_count} apexes'
        if params:
            title += f" | q={params.get('q', 'N/A'):.2f}, gap={params.get('min_gap_m', 'N/A'):.1f}m"
        title += ')'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved track layout with apexes to {output_path}")

