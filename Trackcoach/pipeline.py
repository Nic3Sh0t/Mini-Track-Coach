"""Main CLI entry module."""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from trackcoach.detect import (
    auto_tune_apex_params,
    detect_apexes,
    detect_apexes_by_curvature,
    detect_events,
    export_curvature_debug,
    find_reference_lap,
    process_all_laps,
    scan_apex_params,
    detect_relative_events,
)
from trackcoach.features import (
    build_reference_centerline,
    compute_curvature,
    compute_jerk,
    compute_longitudinal_acceleration,
    smooth_curvature,
    smooth_velocity,
)
from trackcoach.geom import (
    compute_arc_length,
    project_to_local_coords,
)
from trackcoach.io_utils import read_and_preprocess_csv
from trackcoach.viz import (
    plot_curvature_apex,
    plot_track_with_apexes,
    plot_xy_brakes,
    plot_xy_brakes_by_lap,
    plot_xy_speed,
    plot_xy_speed_by_lap,
)
from trackcoach.stability import generate_stability_artifacts
from trackcoach.analysis import IssueAnalyzer
from trackcoach.advice import CoachingAdvisor
from trackcoach.config import ConfigManager
from trackcoach.export import export_to_excel, export_to_pdf
from trackcoach.pb_memory import PBMemory

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='TrackCoach: Track driving data analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Input CSV file path'
    )
    
    # Output arguments
    parser.add_argument(
        '--outdir',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '--ref-lap',
        type=str,
        default='fastest',
        help='Reference lap mode: fastest, median, or specify a positive integer lap number'
    )
    
    # Column name overrides
    parser.add_argument('--lat-col', type=str, default=None, help='Latitude column name (override auto-detection)')
    parser.add_argument('--lon-col', type=str, default=None, help='Longitude column name (override auto-detection)')
    parser.add_argument('--time-col', type=str, default=None, help='Time column name (override auto-detection)')
    parser.add_argument('--speed-col', type=str, default=None, help='Speed column name (override auto-detection)')
    parser.add_argument('--lap-col', type=str, default=None, help='Lap column name (override auto-detection)')
    parser.add_argument('--along-col', type=str, default=None, help='Longitudinal acceleration column name (override auto-detection)')
    
    # Speed unit
    parser.add_argument(
        '--speed-unit',
        type=str,
        choices=['auto', 'ms', 'kmh', 'mph'],
        default='auto',
        help='Speed unit (default: auto)'
    )
    
    # Smoothing parameters
    parser.add_argument(
        '--smooth-win-sec',
        type=float,
        default=0.5,
        help='Velocity smoothing window length in seconds (default: 0.5)'
    )
    
    # Apex detection parameters
    parser.add_argument(
        '--apex-height-quantile',
        type=float,
        default=0.78,
        help='κ(s) threshold quantile (default: 0.78)'
    )
    parser.add_argument(
        '--min-turn-gap-m',
        type=float,
        default=65.0,
        help='Minimum gap between peaks in meters (default: 65)'
    )
    parser.add_argument(
        '--apex-prom-mult',
        type=float,
        default=0.60,
        help='Prominence threshold = IQR(κ)*mult (default: 0.60)'
    )
    parser.add_argument(
        '--post-merge-gap-m',
        type=float,
        default=45.0,
        help='Minimum gap for post-merge in meters (default: 45)'
    )
    parser.add_argument(
        '--apex-target-min',
        type=int,
        default=12,
        help='Target minimum number of turns (default: 12)'
    )
    parser.add_argument(
        '--apex-target-max',
        type=int,
        default=16,
        help='Target maximum number of turns (default: 16)'
    )
    parser.add_argument(
        '--apex-auto-tune',
        action='store_true',
        help='Enable auto-tuning to hit target turn count'
    )
    parser.add_argument(
        '--apex-scan',
        action='store_true',
        help='Grid scan and output suggested combinations'
    )
    parser.add_argument(
        '--debug-curvature',
        action='store_true',
        help='Output κ(s) and peak indices debug CSV'
    )
    parser.add_argument(
        '--centerline-csv',
        type=str,
        default=None,
        help='External centerline CSV path (optional, contains x,y columns)'
    )
    
    # Segment parameters
    parser.add_argument(
        '--segment-window-m',
        type=float,
        default=60.0,
        help='Segment window width in meters (default: 60)'
    )
    
    # Braking parameters
    parser.add_argument(
        '--brake-thr-g',
        type=float,
        default=-0.28,
        help='Heavy brake threshold in G (default: -0.28)'
    )
    parser.add_argument(
        '--brake-onset-thr-g',
        type=float,
        default=-0.15,
        help='Brake onset threshold in G (default: -0.15)'
    )

    # Relative event thresholds
    parser.add_argument(
        '--late-brake-thr-m',
        type=float,
        default=7.0,
        help='Late brake detection threshold in meters (default: 7.0, corresponds to d_delta_s < -threshold)'
    )
    parser.add_argument(
        '--early-brake-thr-m',
        type=float,
        default=12.0,
        help='Early brake detection threshold in meters (default: 12.0, corresponds to d_delta_s > threshold)'
    )
    parser.add_argument(
        '--overslow-thr-ms',
        type=float,
        default=1.2,
        help='Overslow detection threshold in m/s (default: 1.2, corresponds to d_Vmin < -threshold)'
    )
    parser.add_argument(
        '--weak-exit-thr-ms',
        type=float,
        default=1.5,
        help='Weak exit speed threshold in m/s (default: 1.5, corresponds to d_Vexit < -threshold)'
    )
    parser.add_argument(
        '--weak-exit-adecel-g',
        type=float,
        default=0.25,
        help='Weak exit acceleration threshold (|a_long_peak|, units g, default: 0.25)'
    )
    
    # Visualization
    parser.add_argument(
        '--plot',
        type=str,
        choices=['true', 'false'],
        default='false',
        help='Whether to generate plots (default: false)'
    )
    
    # Stability analysis parameters
    parser.add_argument(
        '--stability-heatmap-agg',
        type=str,
        choices=['median', 'iqr', 'std'],
        default='iqr',
        help='Consistency heatmap aggregation method (default: iqr)'
    )
    parser.add_argument(
        '--no-stability',
        action='store_true',
        help='Skip stability analysis (default: generate stability report and heatmap)'
    )
    parser.add_argument(
        '--keep-warmup-cooldown',
        action='store_true',
        help='Keep warmup lap (lap 0) and cooldown lap (last lap), default: auto-filter'
    )
    
    # Issue analysis parameters
    parser.add_argument(
        '--analyze-issues',
        action='store_true',
        help='Execute issue analysis and generate Top Issues report (default: do not execute)'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=8,
        help='Number of Top Issues (default: 8)'
    )
    parser.add_argument(
        '--stability-weight',
        type=float,
        default=0.3,
        help='Stability score weight (0-1, default: 0.3, i.e., stability accounts for 30%% of composite score)'
    )
    
    # Advice generation parameters
    parser.add_argument(
        '--generate-advice',
        action='store_true',
        help='Generate coaching advice report (default: do not execute)'
    )
    parser.add_argument(
        '--advice-config',
        type=str,
        default='llamacloud_config.json',
        help='LlamaCloud config file path (default: llamacloud_config.json). Create from llamacloud_config.json.example'
    )
    
    # Configuration preset parameters
    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        help='Use parameter preset (optional values: default, sows_track, high_speed_track, technical_track)'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default=None,
        help='Custom config file path (JSON format)'
    )
    parser.add_argument(
        '--save-config',
        type=str,
        default=None,
        help='Save current parameters as config file (specify output path)'
    )
    
    # Export parameters
    parser.add_argument(
        '--export-excel',
        type=str,
        default=None,
        help='Export Excel report (specify output path, e.g.: out/report.xlsx)'
    )
    parser.add_argument(
        '--export-pdf',
        type=str,
        default=None,
        help='Export PDF report (specify output path, e.g.: out/report.pdf)'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List all available parameter presets and exit'
    )
    parser.add_argument(
        '--pb-memory',
        type=str,
        default=None,
        help='PB memory file path (default: <outdir>/pb_memory.json)'
    )
    parser.add_argument(
        '--session-name',
        type=str,
        default=None,
        help='Current session name (for PB memory, default: use output directory name)'
    )
    parser.add_argument(
        '--session-date',
        type=str,
        default=None,
        help='Current session date (format: YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--no-pb-memory',
        action='store_true',
        help='Disable PB memory feature (do not use historical PB)'
    )
    
    # Debug
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable DEBUG logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration preset (if specified)
    config_manager = ConfigManager(
        config_file=Path(args.config_file) if args.config_file else None
    )
    
    # List presets and exit
    if args.list_presets:
        print("\nAvailable parameter presets:")
        print("=" * 60)
        presets = config_manager.list_presets()
        for name, desc in presets.items():
            print(f"  {name:20s} - {desc}")
        print("=" * 60)
        print("\nUsage: --preset <preset_name>")
        return 0
    
    if args.preset:
        config_manager.apply_preset_to_args(args.preset, args)
    
    # Setup logging
    setup_logging(debug=args.debug)
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Read CSV
        logger.info("=" * 60)
        logger.info("Step 1: Reading and preprocessing CSV")
        logger.info("=" * 60)
        
        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 1
        
        df, col_mapping = read_and_preprocess_csv(
            csv_path,
            lat_col=args.lat_col,
            lon_col=args.lon_col,
            time_col=args.time_col,
            speed_col=args.speed_col,
            lap_col=args.lap_col,
            along_col=args.along_col,
            speed_unit=args.speed_unit
        )
        
        # 2. Project coordinates
        logger.info("=" * 60)
        logger.info("Step 2: Projecting coordinates")
        logger.info("=" * 60)
        
        x, y, lon0, lat0 = project_to_local_coords(df['lon'], df['lat'])
        df['x'] = x
        df['y'] = y
        
        # Check and remove incomplete laps (e.g., last lap that didn't return to start/finish)
        # Also filter warmup lap (lap 0) and cooldown lap (last lap)
        if 'lap' in df.columns and df['lap'].nunique() > 1:
            # Get all lap numbers and sort
            unique_laps = sorted(df['lap'].dropna().unique())
            max_lap = max(unique_laps) if unique_laps else None
            
            # Filter lap numbers: remove lap 0 (warmup) and last lap (cooldown)
            laps_to_exclude = []
            
            # Decide whether to filter warmup and cooldown laps based on parameters
            if not args.keep_warmup_cooldown:
                # Remove lap 0 (warmup lap)
                if 0 in unique_laps:
                    laps_to_exclude.append(0)
                    logger.info(f"Filtering out lap 0 (warmup lap)")
                
                # Remove last lap (cooldown lap)
                if max_lap is not None and max_lap > 0:
                    laps_to_exclude.append(int(max_lap))
                    logger.info(f"Filtering out lap {max_lap} (cooldown lap)")
            
            if laps_to_exclude:
                before_count = len(df)
                df = df[~df['lap'].isin(laps_to_exclude)].reset_index(drop=True)
                after_count = len(df)
                logger.info(f"Filtered out laps {laps_to_exclude}: {before_count} -> {after_count} rows")
            
            # Check and remove incomplete laps (based on lap duration and displacement)
            lap_stats = []
            remaining_laps = sorted(df['lap'].dropna().unique())
            for lap_id in remaining_laps:
                lap_df = df[df['lap'] == lap_id]
                if len(lap_df) < 100:
                    continue
                first = lap_df.iloc[0]
                last = lap_df.iloc[-1]
                duration = float(last['t_s'] - first['t_s'])
                displacement = float(np.hypot(last['x'] - first['x'], last['y'] - first['y']))
                lap_stats.append({
                    'lap': lap_id,
                    'duration': duration,
                    'displacement': displacement,
                    'count': len(lap_df)
                })
            
            if lap_stats:
                durations = [s['duration'] for s in lap_stats if s['duration'] > 0]
                median_duration = float(np.median(durations)) if durations else None
                incomplete_laps = []
                for stats in lap_stats:
                    too_short = median_duration is not None and stats['duration'] < 0.9 * median_duration
                    far_from_start = stats['displacement'] > 40.0
                    if too_short or far_from_start:
                        incomplete_laps.append(stats['lap'])
                        logger.warning(
                            f"Lap {stats['lap']} flagged as incomplete: "
                            f"duration={stats['duration']:.2f}s (median={median_duration:.2f}s) "
                            f"displacement={stats['displacement']:.1f}m count={stats['count']}"
                        )
                
                if incomplete_laps:
                    df = df[~df['lap'].isin(incomplete_laps)].reset_index(drop=True)
                    logger.warning(f"Dropped incomplete laps: {sorted(incomplete_laps)}")
            
            # Record final retained lap numbers
            final_laps = sorted(df['lap'].dropna().unique())
            if final_laps:
                logger.info(f"Final lap range: {min(final_laps)} - {max(final_laps)} (total {len(final_laps)} laps)")
        
        # 3. Compute arc length
        logger.info("=" * 60)
        logger.info("Step 3: Computing arc length")
        logger.info("=" * 60)
        
        s = compute_arc_length(df['x'], df['y'], df['lap'] if 'lap' in df.columns else None)
        df['s'] = s
        
        # 4. Smooth velocity and compute acceleration
        logger.info("=" * 60)
        logger.info("Step 4: Smoothing velocity and computing acceleration")
        logger.info("=" * 60)
        
        v_smooth = smooth_velocity(df['v'], df['t_s'], smooth_win_sec=args.smooth_win_sec)
        df['v_smooth'] = v_smooth
        
        a_long, a_long_source = compute_longitudinal_acceleration(
            v_smooth,
            df['t_s'],
            df.get('longitudinal_acc_g', None)
        )
        df['a_long'] = a_long
        
        jerk = compute_jerk(a_long, df['t_s'])
        df['jerk'] = jerk
        
        logger.info(f"Longitudinal acceleration source: {a_long_source}")
        
        # 5. Reference lap and curvature
        logger.info("=" * 60)
        logger.info("Step 5: Selecting benchmark lap and computing curvature")
        logger.info("=" * 60)
        
        # Initialize PB memory system
        pb_memory = None
        if not args.no_pb_memory:
            pb_memory_file = Path(args.pb_memory) if args.pb_memory else outdir / 'pb_memory.json'
            pb_memory = PBMemory(pb_memory_file)
            
            # Get historical PB (if exists)
            historical_pb = pb_memory.get_current_pb()
            if historical_pb:
                logger.info(f"Historical PB: {pb_memory.format_pb_info(historical_pb)}")
        
        ref_lap, ref_laptime, ref_mode = find_reference_lap(df, mode=args.ref_lap)
        
        # Check if historical PB should be used
        use_historical_pb = False
        historical_pb_lap = None
        historical_pb_laptime = None
        
        if pb_memory and ref_lap is not None and ref_laptime is not None:
            historical_pb = pb_memory.get_current_pb()
            if historical_pb:
                historical_pb_laptime = historical_pb['laptime_s']
                historical_pb_lap = historical_pb['lap']
                
                # Compare current PB with historical PB
                if ref_laptime < historical_pb_laptime:
                    # Current PB is faster, update memory
                    session_name = args.session_name or outdir.name
                    session_date = args.session_date or datetime.now().strftime('%Y-%m-%d')
                    updated, old_pb = pb_memory.update_pb(
                        ref_lap, ref_laptime, session_name, session_date, outdir
                    )
                    if updated:
                        logger.info(f"✅ PB updated! New PB: Lap {ref_lap}, {ref_laptime:.2f}s")
                else:
                    # Historical PB is faster, use historical PB as benchmark
                    use_historical_pb = True
                    logger.info(
                        f"📌 Using historical PB as benchmark: Lap {historical_pb_lap}, {historical_pb_laptime:.2f}s "
                        f"(Current fastest: Lap {ref_lap}, {ref_laptime:.2f}s)"
                    )
        
        # Determine final reference lap to use
        if use_historical_pb and historical_pb_lap is not None:
            # Use historical PB
            final_ref_lap = historical_pb_lap
            final_ref_laptime = historical_pb_laptime
            final_ref_mode = 'historical_pb'
            logger.info(f"🎯 Using historical PB as analysis benchmark: Lap {final_ref_lap}, {final_ref_laptime:.2f}s")
        else:
            # Use current PB
            final_ref_lap = ref_lap
            final_ref_laptime = ref_laptime
            final_ref_mode = ref_mode
            if pb_memory and ref_lap is not None and ref_laptime is not None:
                # If first record, also update memory
                session_name = args.session_name or outdir.name
                session_date = args.session_date or datetime.now().strftime('%Y-%m-%d')
                pb_memory.update_pb(ref_lap, ref_laptime, session_name, session_date, outdir)
        
        if final_ref_lap is not None and 'lap' in df.columns:
            ref_df = df[df['lap'] == final_ref_lap].copy()
        else:
            ref_df = df.copy()
        
        if final_ref_lap is not None and final_ref_laptime is not None:
            logger.info(f"Benchmark lap: #{final_ref_lap} (T={final_ref_laptime:.2f}s, mode={final_ref_mode})")
        else:
            logger.info(f"Benchmark lap: #ALL (T=N/A, mode={final_ref_mode})")
        
        # Update ref_lap variable for subsequent use
        ref_lap = final_ref_lap
        ref_laptime = final_ref_laptime
        
        # Resample reference lap centerline
        s_ref, x_ref, y_ref = build_reference_centerline(
            df,
            ref_lap_idx=ref_lap,
            n_points=1000
        )
        
        # Compute curvature
        kappa = compute_curvature(x_ref, y_ref, s_ref)
        kappa_smooth = smooth_curvature(kappa, window_length=21)
        
        # 6. Apex detection
        logger.info("=" * 60)
        logger.info("Step 6: Detecting apexes")
        logger.info("=" * 60)
        
        apex_params = {}
        peaks_before_merge = None
        peaks_after_merge = None
        
        # Grid scan mode
        if args.apex_scan:
            scan_df = scan_apex_params(
                s_ref,
                kappa_smooth,
                target_min=args.apex_target_min,
                target_max=args.apex_target_max
            )
            scan_path = outdir / 'apex_scan.csv'
            scan_df.to_csv(scan_path, index=False)
            logger.info(f"Saved apex scan results to {scan_path}")
            
            # Continue with default parameters
            apex_s, peaks_before_merge, peaks_after_merge = detect_apexes_by_curvature(
                s_ref, kappa_smooth,
                q=args.apex_height_quantile,
                min_gap_m=args.min_turn_gap_m,
                prom_mult=args.apex_prom_mult,
                post_merge_gap_m=args.post_merge_gap_m
            )
            apex_params = {
                'q': args.apex_height_quantile,
                'min_gap_m': args.min_turn_gap_m,
                'prom_mult': args.apex_prom_mult,
                'post_merge_gap_m': args.post_merge_gap_m
            }
        
        # Auto-tuning mode
        elif args.apex_auto_tune:
            apex_params, apex_s = auto_tune_apex_params(
                s_ref,
                kappa_smooth,
                target_min=args.apex_target_min,
                target_max=args.apex_target_max,
                base_min_gap_m=args.min_turn_gap_m,
                base_q=args.apex_height_quantile
            )
            # Re-detect to get peaks information
            _, peaks_before_merge, peaks_after_merge = detect_apexes_by_curvature(
                s_ref, kappa_smooth,
                q=apex_params['q'],
                min_gap_m=apex_params['min_gap_m'],
                prom_mult=apex_params['prom_mult'],
                post_merge_gap_m=apex_params['post_merge_gap_m']
            )
            # Save parameters
            params_path = outdir / 'apex_params.json'
            with open(params_path, 'w') as f:
                json.dump(apex_params, f, indent=2)
            logger.info(f"Saved apex parameters to {params_path}")
        
        # Manual mode
        else:
            apex_s, peaks_before_merge, peaks_after_merge = detect_apexes_by_curvature(
                s_ref, kappa_smooth,
                q=args.apex_height_quantile,
                min_gap_m=args.min_turn_gap_m,
                prom_mult=args.apex_prom_mult,
                post_merge_gap_m=args.post_merge_gap_m
            )
            apex_params = {
                'q': args.apex_height_quantile,
                'min_gap_m': args.min_turn_gap_m,
                'prom_mult': args.apex_prom_mult,
                'post_merge_gap_m': args.post_merge_gap_m
            }
        
        apex_count = len(apex_s)
        logger.info(f"Apex count: {apex_count} | "
                   f"q={apex_params['q']:.2f} min_gap={apex_params['min_gap_m']:.1f}m "
                   f"prom={apex_params['prom_mult']:.2f}*IQR merge_gap={apex_params['post_merge_gap_m']:.1f}m")
        
        # DebugOutput
        if args.debug_curvature:
            debug_path = outdir / 'curvature_debug.csv'
            export_curvature_debug(
                s_ref, kappa_smooth,
                peaks_before_merge=peaks_before_merge,
                peaks_after_merge=peaks_after_merge,
                output_path=debug_path
            )
        
        # 7. Generate corner cards and events
        logger.info("=" * 60)
        logger.info("Step 7: Generating corner cards and detecting events")
        logger.info("=" * 60)
        
        corner_cards_df, events_df = process_all_laps(
            df,
            apex_s,
            segment_window_m=args.segment_window_m,
            brake_onset_thr_g=args.brake_onset_thr_g,
            brake_thr_g=args.brake_thr_g,
            conservative_entry_factor=1.3
        )

        if 'delta_s_to_apex_m' not in corner_cards_df.columns:
            corner_cards_df['delta_s_to_apex_m'] = np.where(
                corner_cards_df['s_brake_onset'].notna(),
                corner_cards_df['s_apex'] - corner_cards_df['s_brake_onset'],
                np.nan
            )

        for diff_col in ['d_delta_s', 'd_Ventry', 'd_Vmin', 'd_Vexit']:
            if diff_col not in corner_cards_df.columns:
                corner_cards_df[diff_col] = np.nan

        # Use final determined reference lap (may be historical PB)
        analysis_ref_lap = ref_lap  # Reference lap for analysis
        
        # If using historical PB, try to load data from historical session
        pb_cards = None
        if use_historical_pb and pb_memory:
            historical_pb = pb_memory.get_current_pb()
            if historical_pb and historical_pb.get('session_dir'):
                historical_session_dir = Path(historical_pb['session_dir'])
                historical_corner_cards_path = historical_session_dir / 'corner_cards.csv'
                
                if historical_corner_cards_path.exists():
                    try:
                        historical_corner_cards = pd.read_csv(historical_corner_cards_path)
                        historical_pb_lap = historical_pb['lap']
                        pb_cards = historical_corner_cards[
                            historical_corner_cards['lap'] == historical_pb_lap
                        ].copy()
                        logger.info(f"Loading PB data from historical session: {historical_session_dir}")
                    except Exception as e:
                        logger.warning(f"Failed to load historical PB data: {e}")
        
        # If historical PB data loading failed, try using current session data
        if pb_cards is None or pb_cards.empty:
            if analysis_ref_lap is not None and analysis_ref_lap in corner_cards_df['lap'].unique():
                pb_cards = corner_cards_df[corner_cards_df['lap'] == analysis_ref_lap].copy()
            else:
                pb_cards = None
        
        # Calculate deltas
        if pb_cards is not None and not pb_cards.empty:
            pb_cards = pb_cards.set_index('turn')
            for idx, card in corner_cards_df.iterrows():
                turn = card['turn']
                if turn not in pb_cards.index:
                    continue
                pb_card = pb_cards.loc[turn]
                if isinstance(pb_card, pd.DataFrame):
                    pb_card = pb_card.iloc[0]

                curr_delta = card['delta_s_to_apex_m']
                pb_delta = pb_card.get('delta_s_to_apex_m', np.nan)
                if pd.notna(curr_delta) and pd.notna(pb_delta):
                    corner_cards_df.at[idx, 'd_delta_s'] = curr_delta - pb_delta

                for metric, diff_col in [('Ventry', 'd_Ventry'), ('Vmin', 'd_Vmin'), ('Vexit', 'd_Vexit')]:
                    curr_val = card.get(metric, np.nan)
                    pb_val = pb_card.get(metric, np.nan)
                    if pd.notna(curr_val) and pd.notna(pb_val):
                        corner_cards_df.at[idx, diff_col] = curr_val - pb_val
                    elif pd.notna(curr_val):
                        corner_cards_df.at[idx, diff_col] = np.nan
        else:
            if use_historical_pb:
                logger.warning(
                    f"Unable to load PB data from historical session, will use current session reference lap {analysis_ref_lap} "
                    f"(if exists) for delta calculation"
                )
            else:
                logger.warning("Unable to compute d_* relative metrics: reference lap %s not in corner cards", analysis_ref_lap)

        relative_events_df = detect_relative_events(
            corner_cards_df,
            analysis_ref_lap,  # Use analysis benchmark lap
            late_brake_thr_m=args.late_brake_thr_m,
            early_brake_thr_m=args.early_brake_thr_m,
            overslow_thr_ms=args.overslow_thr_ms,
            weak_exit_thr_ms=args.weak_exit_thr_ms,
            weak_exit_adecel_g=args.weak_exit_adecel_g
        )

        if relative_events_df is not None and not relative_events_df.empty:
            if events_df is None or len(events_df) == 0:
                events_df = relative_events_df
            else:
                events_df = pd.concat([events_df, relative_events_df], ignore_index=True)
        
        # Count events
        if events_df is not None and len(events_df) > 0:
            event_counts = events_df['type'].value_counts().to_dict()
            logger.info("Events detected by type: %s", event_counts)
        else:
            logger.info("No events detected")
        
        # 8. SaveOutput
        logger.info("=" * 60)
        logger.info("Step 8: Saving outputs")
        logger.info("=" * 60)
        
        corner_cards_path = outdir / 'corner_cards.csv'
        events_path = outdir / 'events.csv'
        
        corner_cards_df.to_csv(corner_cards_path, index=False)
        logger.info(f"Saved corner cards: {corner_cards_path} ({len(corner_cards_df)} rows)")
        
        events_df.to_csv(events_path, index=False)
        logger.info(f"Saved events: {events_path} ({len(events_df)} rows)")
        if events_df is not None and len(events_df) > 0:
            logger.info("Event type counts: %s", events_df['type'].value_counts().to_dict())
        
        # 9. Visualization (optional)
        if args.plot == 'true':
            logger.info("=" * 60)
            logger.info("Step 9: Generating visualizations")
            logger.info("=" * 60)
            
            plot_xy_speed(
                df,
                outdir / 'xy_speed.png',
                x_col='x',
                y_col='y',
                v_col='v_smooth',
                benchmark_lap=ref_lap,
                lap_col='lap'
            )
            
            plot_curvature_apex(
                s_ref,
                kappa_smooth,
                apex_s,
                outdir / 'curvature_apex.png',
                apex_count=apex_count,
                params=apex_params
            )
            
            plot_xy_brakes(
                df,
                events_df,
                corner_cards_df,
                outdir / 'xy_brakes.png',
                x_col='x',
                y_col='y',
                s_col='s',
                benchmark_lap=ref_lap,
                lap_col='lap'
            )
            
            # Generate track layout (using reference lap)
            plot_track_with_apexes(
                ref_df if ref_lap is not None else df,
                apex_s,
                outdir / 'track_layout_apexes.png',
                lap=ref_lap,
                x_col='x',
                y_col='y',
                s_col='s',
                lap_col='lap',
                apex_count=apex_count,
                params=apex_params,
                benchmark_lap=ref_lap
            )
            
            # Generate speed and brake plots for each lap separately
            logger.info("Generating per-lap visualizations...")
            plot_xy_speed_by_lap(
                df,
                outdir,
                lap_col='lap',
                x_col='x',
                y_col='y',
                v_col='v_smooth',
                benchmark_lap=ref_lap
            )
            
            plot_xy_brakes_by_lap(
                df,
                events_df,
                corner_cards_df,
                outdir,
                lap_col='lap',
                x_col='x',
                y_col='y',
                s_col='s',
                benchmark_lap=ref_lap
            )
        
        # 10. Complete
        logger.info("=" * 60)
        logger.info("Processing completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Output files:")
        logger.info(f"  - {corner_cards_path} ({len(corner_cards_df)} rows)")
        logger.info(f"  - {events_path} ({len(events_df)} rows)")
        benchmark_path = outdir / 'benchmark_lap.json'
        benchmark_payload = {
            'lap': int(ref_lap) if ref_lap is not None else None,
            'laptime_s': float(ref_laptime) if ref_laptime is not None else None,
            'mode': final_ref_mode if 'final_ref_mode' in locals() else ref_mode,
            'is_historical_pb': use_historical_pb if 'use_historical_pb' in locals() else False
        }
        
        # If historical PB was used, add historical PB information
        if 'use_historical_pb' in locals() and use_historical_pb and pb_memory:
            historical_pb = pb_memory.get_current_pb()
            if historical_pb:
                benchmark_payload['historical_pb'] = {
                    'lap': historical_pb['lap'],
                    'laptime_s': historical_pb['laptime_s'],
                    'session_name': historical_pb['session_name'],
                    'session_date': historical_pb['session_date']
                }
        
        with open(benchmark_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_payload, f, indent=2)
        logger.info(f"  - {benchmark_path}")
        
        # Display PB memory status
        if pb_memory:
            current_pb = pb_memory.get_current_pb()
            if current_pb:
                logger.info(f"  - PB memory: {pb_memory.format_pb_info(current_pb)}")
                if use_historical_pb:
                    logger.info(f"    (This analysis uses historical PB as benchmark)")
        if args.plot == 'true':
            logger.info(f"  - {outdir / 'xy_speed.png'} (overview)")
            logger.info(f"  - {outdir / 'curvature_apex.png'}")
            logger.info(f"  - {outdir / 'xy_brakes.png'} (overview)")
            logger.info(f"  - {outdir / 'track_layout_apexes.png'} (track layout)")
            if 'lap' in df.columns and df['lap'].nunique() > 1:
                num_laps = df['lap'].nunique()
                logger.info(f"  - Per-lap speed plots: xy_speed_lap_*.png ({num_laps} files)")
                logger.info(f"  - Per-lap brake plots: xy_brakes_lap_*.png ({num_laps} files)")
        
        # 10. Issue analysis (optional)
        if args.analyze_issues:
            logger.info("=" * 60)
            logger.info("Step 10: Analyzing issues and generating Top Issues report")
            logger.info("=" * 60)
            
            try:
                analyzer = IssueAnalyzer()
                
                # Find stability report path
                stability_report_path = outdir / 'stability_report.csv'
                if not stability_report_path.exists():
                    stability_report_path = None
                    logger.info("Stability report not found, will only use issue score")
                
                analysis_result = analyzer.analyze(
                    corner_cards_df,
                    events_df if events_df is not None else pd.DataFrame(),
                    analysis_ref_lap if analysis_ref_lap is not None else 0,
                    topk=args.topk,
                    stability_report_path=stability_report_path,
                    stability_weight=args.stability_weight
                )
                
                # Save Top Issues
                top_issues_path = outdir / 'top_issues.csv'
                analyzer.save_top_issues(analysis_result['top_issues'], top_issues_path)
                logger.info(f"Saved top issues: {top_issues_path}")
                
                # If stability score is integrated, output message
                if analysis_result.get('stability_integrated', False):
                    logger.info("Top Issues sorted by composite score (issue score + stability score)")
                
                # Save plot
                plot_path = outdir / 'top_issues_bar.png'
                plot_result = analyzer.save_top_issues_plot(analysis_result['top_issues'], plot_path)
                plot_relative_path = None
                if plot_result:
                    try:
                        import os
                        plot_relative_path = os.path.relpath(plot_result, outdir)
                    except ValueError:
                        plot_relative_path = plot_path.name
                
                # Save Markdown report
                summary_path = outdir / 'events_summary.md'
                analyzer.save_summary_markdown(
                    events_df if events_df is not None else pd.DataFrame(),
                    analysis_result['turn_stats'],
                    analysis_result['top_issues'],
                    summary_path,
                    plot_relative_path=plot_relative_path
                )
                logger.info(f"Saved events summary: {summary_path}")
                
                # If deltas were recomputed, save updated corner_cards
                if analysis_result.get('recomputed', False):
                    cards_with_delta_path = outdir / 'corner_cards_with_delta.csv'
                    analysis_result['corner_cards_with_delta'].to_csv(cards_with_delta_path, index=False)
                    logger.info(f"Saved corner cards with delta: {cards_with_delta_path}")
                
                if analysis_result.get('pb_included', False):
                    logger.warning("PB lap data included in statistics, please check filter logic")
                
                logger.info(f"Top Issues analysis completed successfully")
            except Exception as e:
                logger.error(f"Error during issue analysis: {e}", exc_info=args.debug)
                if args.debug:
                    raise
        else:
            logger.debug("Issue analysis skipped (use --analyze-issues to enable)")
        
        # 10.5 Generate coaching advice (optional, requires issue analysis first)
        if args.generate_advice:
            logger.info("=" * 60)
            logger.info("Step 10.5: Generating coaching advice report")
            logger.info("=" * 60)
            
            top_issues_path = outdir / 'top_issues.csv'
            if not top_issues_path.exists():
                logger.warning(f"Top issues file not found: {top_issues_path}")
                logger.warning("Please run with --analyze-issues first, or generate top_issues.csv manually")
            else:
                try:
                    advisor = CoachingAdvisor(config_path=args.advice_config)
                    top_issues_df = pd.read_csv(top_issues_path)
                    
                    # Find stability report path
                    stability_report_path = outdir / 'stability_report.csv'
                    if not stability_report_path.exists():
                        stability_report_path = None
                        logger.info("Stability report not found, will skip stability metrics in report")
                    
                    analysis = advisor.analyze_top_issues(
                        top_issues_df,
                        events_df if events_df is not None else pd.DataFrame(),
                        stability_report_path=stability_report_path
                    )
                    
                    advice_path = outdir / 'coaching_advice_report.md'
                    advisor.generate_report(analysis, advice_path, stability_report_path=stability_report_path)
                    logger.info(f"Saved coaching advice report: {advice_path}")
                    logger.info(f"Coaching advice generation completed successfully")
                except Exception as e:
                    logger.error(f"Error during coaching advice generation: {e}", exc_info=args.debug)
                    if args.debug:
                        raise
            if not args.analyze_issues:
                logger.warning("Recommend running with --analyze-issues for better results")
        else:
            logger.debug("Coaching advice generation skipped (use --generate-advice to enable)")
        
        # 10.6 Export reports (optional)
        if args.export_excel or args.export_pdf:
            logger.info("=" * 60)
            logger.info("Step 10.6: Exporting reports")
            logger.info("=" * 60)
            
            coaching_report_path = outdir / 'coaching_advice_report.md' if args.generate_advice else None
            top_issues_path = outdir / 'top_issues.csv' if args.analyze_issues else None
            events_path = outdir / 'events.csv'
            corner_cards_path = outdir / 'corner_cards.csv'
            stability_report_path = outdir / 'stability_report.csv' if not args.no_stability else None
            events_summary_path = outdir / 'events_summary.md' if args.analyze_issues else None
            
            if args.export_excel:
                try:
                    excel_path = export_to_excel(
                        Path(args.export_excel),
                        coaching_report_path=coaching_report_path,
                        top_issues_path=top_issues_path,
                        events_path=events_path if events_path.exists() else None,
                        corner_cards_path=corner_cards_path if corner_cards_path.exists() else None,
                        stability_report_path=stability_report_path if stability_report_path and stability_report_path.exists() else None,
                        events_summary_path=events_summary_path if events_summary_path and events_summary_path.exists() else None
                    )
                    logger.info(f"Excel report exported: {excel_path}")
                except Exception as e:
                    logger.error(f"Failed to export Excel: {e}", exc_info=args.debug)
                    if args.debug:
                        raise
            
            if args.export_pdf:
                try:
                    pdf_path = export_to_pdf(
                        Path(args.export_pdf),
                        coaching_report_path=coaching_report_path if coaching_report_path and coaching_report_path.exists() else None,
                        top_issues_path=top_issues_path if top_issues_path and top_issues_path.exists() else None,
                        events_summary_path=events_summary_path if events_summary_path and events_summary_path.exists() else None
                    )
                    logger.info(f"PDF report exported: {pdf_path}")
                except Exception as e:
                    logger.error(f"Failed to export PDF: {e}", exc_info=args.debug)
                    if args.debug:
                        raise
        
        # 11. Generate stability analysis artifacts
        if not args.no_stability:
            logger.info("=" * 60)
            logger.info("Step 11: Generating stability artifacts")
            logger.info("=" * 60)
            
            try:
                report_path, heatmap_path = generate_stability_artifacts(
                    corner_cards_df,
                    analysis_ref_lap,  # Use analysis reference lap
                    outdir,
                    delta_metrics=['d_delta_s', 'd_Ventry', 'd_Vmin', 'd_Vexit', 'd_a_long_peak'],
                    heatmap_metrics=['d_delta_s', 'd_Ventry', 'd_Vmin', 'd_a_long_peak'],
                    heatmap_agg=args.stability_heatmap_agg
                )
                logger.info(f"Stability analysis completed successfully")
            except Exception as e:
                logger.error(f"Error during stability analysis: {e}", exc_info=args.debug)
                if args.debug:
                    raise

        # Save config (if specified)
        if args.save_config:
            config_manager.save_config(
                Path(args.save_config),
                preset_name=args.preset if args.preset else None
            )
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=args.debug)
        return 1


if __name__ == '__main__':
    sys.exit(main())

