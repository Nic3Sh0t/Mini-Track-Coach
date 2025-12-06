"""Apex detection, corner cards, and event detection module."""

import logging
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

G = 9.81  # Gravitational acceleration constant


def find_reference_lap(
    df: pd.DataFrame,
    mode: Optional[str] = 'fastest',
    lap_col: str = 'lap',
    time_col: str = 't_s',
    min_samples: int = 200
) -> Tuple[Optional[int], Optional[float], str]:
    """
    Find benchmark lap according to selection mode.
    
    Benchmark lap selection follows these rules:
        - Compute lap time for each lap: max(time_col) - min(time_col)
        - Filter abnormally short laps: lap time < median of all lap times * 0.5 or sample count < min_samples
        - mode = 'fastest': Select lap with shortest lap time among valid laps
        - mode = 'median': Select lap with lap time closest to median
        - mode = 'N': If specified lap exists in valid laps, use it directly; otherwise fall back to 'fastest'
    
    Args:
        df: DataFrame
        mode: Selection mode ('fastest', 'median', or specified lap number string/integer)
        lap_col: Lap number column name
        time_col: Time column name (seconds)
        min_samples: Minimum sample count for valid lap
    
    Returns:
        (Benchmark lap number, lap time (seconds), actually used mode)
    """
    if time_col not in df.columns:
        logger.warning(f"Time column '{time_col}' not found, unable to compute benchmark lap")
        return None, None, 'fastest'
    
    # If lap number column is missing, fall back to whole dataset
    if lap_col not in df.columns:
        lap_time = df[time_col].max() - df[time_col].min()
        lap_time = float(lap_time) if pd.notna(lap_time) else None
        logger.info(f"Lap column '{lap_col}' not found, using whole dataset as benchmark")
        return 0, lap_time, 'fastest'
    
    lap_groups = df.groupby(lap_col)
    lap_stats = lap_groups.agg(
        t_min=(time_col, 'min'),
        t_max=(time_col, 'max'),
        sample_count=(time_col, 'count')
    ).reset_index()
    
    lap_stats['laptime_s'] = lap_stats['t_max'] - lap_stats['t_min']
    lap_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    lap_stats.dropna(subset=['laptime_s'], inplace=True)
    
    if lap_stats.empty:
        logger.info("No valid laps found after computing lap times, using all data as reference")
        lap_time = df[time_col].max() - df[time_col].min()
        lap_time = float(lap_time) if pd.notna(lap_time) else None
        return None, lap_time, 'fastest'
    
    lap_stats.sort_values(by=lap_col, inplace=True)
    global_median = lap_stats['laptime_s'].median()
    
    filtered_stats = lap_stats.copy()
    if pd.notna(global_median) and global_median > 0:
        filtered_stats = filtered_stats[filtered_stats['laptime_s'] >= 0.5 * global_median]
    else:
        logger.warning("Unable to compute meaningful lap time median; skipping short-lap filter")
    filtered_stats = filtered_stats[filtered_stats['sample_count'] >= min_samples]
    
    if filtered_stats.empty:
        logger.warning("No laps passed quality filter; falling back to all laps for benchmark selection")
        filtered_stats = lap_stats
    
    # Parse mode
    requested_mode = 'fastest'
    requested_lap: Optional[int] = None
    
    if isinstance(mode, str):
        mode_str = mode.strip().lower()
        if mode_str in {'fastest', 'median'}:
            requested_mode = mode_str
        else:
            if mode_str.isdigit():
                requested_mode = 'N'
                requested_lap = int(mode_str)
            else:
                logger.warning(f"Unrecognized ref-lap value '{mode}', fallback to fastest")
    elif isinstance(mode, (int, np.integer)):
        requested_mode = 'N'
        requested_lap = int(mode)
    else:
        logger.warning(f"Unsupported ref-lap type '{type(mode)}', fallback to fastest")
    
    selected_row = None
    applied_mode = requested_mode
    
    if requested_mode == 'N' and requested_lap is not None:
        matches = filtered_stats[filtered_stats[lap_col] == requested_lap]
        if not matches.empty:
            selected_row = matches.iloc[0]
        else:
            logger.warning(f"Requested reference lap #{requested_lap} not available after filtering; falling back to fastest")
            applied_mode = 'fastest'
    
    if selected_row is None:
        if filtered_stats.empty:
            logger.warning("No laps available to select benchmark from after fallback")
            return None, None, 'fastest'
        
        if applied_mode == 'median':
            median_time = filtered_stats['laptime_s'].median()
            candidates = filtered_stats.assign(
                time_diff=(filtered_stats['laptime_s'] - median_time).abs()
            )
            selected_row = candidates.sort_values(by=['time_diff', lap_col]).iloc[0]
        else:
            selected_row = filtered_stats.sort_values(by=['laptime_s', lap_col]).iloc[0]
            applied_mode = 'fastest'
    
    lap_value = int(selected_row[lap_col])
    laptime_value = float(selected_row['laptime_s'])
    
    logger.debug(
        "Lap time summary (after filtering): %s",
        filtered_stats[[lap_col, 'laptime_s', 'sample_count']].to_dict(orient='records')
    )
    
    return lap_value, laptime_value, 'N' if applied_mode == 'N' else applied_mode


def merge_close_peaks(
    peaks: np.ndarray,
    kappa: np.ndarray,
    s: np.ndarray,
    min_gap_m: float
) -> np.ndarray:
    """
    Merge peaks that are too close, keeping the one with higher curvature.

    Args:
        peaks: Peak index array
        kappa: Curvature array
        s: Arc length array
        min_gap_m: Minimum gap (meters)

    Returns:
        Merged peak index array
    """
    if len(peaks) == 0:
        return peaks
    
    # Convert to arc length
    peak_s = s[peaks]
    peak_kappa = kappa[peaks]
    
    # Sort by arc length
    sorted_idx = np.argsort(peak_s)
    peak_s_sorted = peak_s[sorted_idx]
    peak_kappa_sorted = peak_kappa[sorted_idx]
    peaks_sorted = peaks[sorted_idx]
    
    merged_peaks = []
    i = 0
    
    while i < len(peaks_sorted):
        current_peak = peaks_sorted[i]
        current_s = peak_s_sorted[i]
        current_kappa = peak_kappa_sorted[i]
        
        # Find peaks that need to be merged (adjacent peaks)
        j = i + 1
        while j < len(peaks_sorted):
            gap = peak_s_sorted[j] - current_s
            if gap < min_gap_m:
                # Too close, keep the one with higher curvature
                if peak_kappa_sorted[j] > current_kappa:
                    current_peak = peaks_sorted[j]
                    current_kappa = peak_kappa_sorted[j]
                    current_s = peak_s_sorted[j]
                j += 1
            else:
                break
        
        merged_peaks.append(current_peak)
        i = j
    
    return np.array(merged_peaks)


def detect_apexes_by_curvature(
    s: np.ndarray,
    kappa: np.ndarray,
    q: float = 0.78,
    min_gap_m: float = 65.0,
    prom_mult: float = 0.60,
    post_merge_gap_m: float = 45.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect apex positions using significance threshold and secondary merging.

    Args:
        s: Arc length array (meters)
        kappa: Curvature array
        q: κ(s) threshold quantile
        min_gap_m: Minimum arc length between peaks (meters)
        prom_mult: Significance threshold multiplier = IQR(κ) * prom_mult
        post_merge_gap_m: Minimum spacing for secondary neighbor merging (meters)

    Returns:
        (Apex arc length position array, peak index array before merging, peak index array after merging)
    """
    # Compute threshold
    height = np.quantile(kappa, q)
    
    # Compute minimum spacing (in points)
    if len(s) > 1:
        ds = np.median(np.diff(s))
        distance = int(min_gap_m / ds) if ds > 0 else 10
    else:
        distance = 10
    
    # Compute significance threshold
    iqr = np.percentile(kappa, 75) - np.percentile(kappa, 25)
    prominence = prom_mult * iqr
    
    logger.debug(f"Apex detection params: height={height:.6f} (q={q}), "
                 f"distance={distance} points ({min_gap_m}m), "
                 f"prominence={prominence:.6f} (IQR*{prom_mult})")
    
    # Find peaks (using height, distance, prominence)
    peaks_before_merge, properties = find_peaks(
        kappa,
        height=height,
        distance=distance,
        prominence=prominence
    )
    
    # Secondary merging
    peaks_merged = merge_close_peaks(peaks_before_merge, kappa, s, post_merge_gap_m)
    apex_s = s[peaks_merged]
    
    logger.info(f"Detected {len(peaks_before_merge)} peaks, after merge: {len(apex_s)} apexes")
    
    return apex_s, peaks_before_merge, peaks_merged


def detect_apexes(
    kappa: np.ndarray,
    s: np.ndarray,
    height_quantile: float = 0.78,
    min_turn_gap_m: float = 65.0,
    apex_prom_mult: float = 0.60,
    post_merge_gap_m: float = 45.0
) -> np.ndarray:
    """
    Detect apex positions (compatible with old interface).

    Args:
        kappa: Curvature array
        s: Arc length array
        height_quantile: Curvature peak threshold quantile
        min_turn_gap_m: Minimum turn spacing (meters)
        apex_prom_mult: Significance threshold multiplier
        post_merge_gap_m: Secondary merge minimum spacing (meters)

    Returns:
        Apex arc length position array
    """
    apex_s, _, _ = detect_apexes_by_curvature(
        s, kappa, q=height_quantile, min_gap_m=min_turn_gap_m,
        prom_mult=apex_prom_mult, post_merge_gap_m=post_merge_gap_m
    )
    return apex_s


def extract_corner_segment(
    df: pd.DataFrame,
    apex_s: float,
    s_col: str = 's',
    segment_window_m: float = 60.0
) -> pd.DataFrame:
    """
    Extract corner segment data (centered on apex).

    Args:
        df: DataFrame
        apex_s: Apex arc length position
        s_col: Arc length column name
        segment_window_m: Window width (meters)

    Returns:
        Corner segment DataFrame
    """
    mask = (df[s_col] >= apex_s - segment_window_m) & (df[s_col] <= apex_s + segment_window_m)
    segment = df[mask].copy()
    
    return segment


def compute_corner_metrics(
    df: pd.DataFrame,
    apex_s: float,
    s_col: str = 's',
    v_col: str = 'v_smooth',
    a_long_col: str = 'a_long',
    jerk_col: Optional[str] = None,
    segment_window_m: float = 60.0,
    brake_onset_thr_g: float = -0.15
) -> dict:
    """
    Compute corner cards metrics.

    Args:
        df: DataFrame
        apex_s: Apex arc length position
        s_col: Arc length column name
        v_col: Velocity column name
        a_long_col: Longitudinal acceleration column name
        jerk_col: Jerk column name (optional)
        segment_window_m: Window width (meters)
        brake_onset_thr_g: Brake onset threshold (G)

    Returns:
        Metrics dictionary
    """
    segment = extract_corner_segment(df, apex_s, s_col, segment_window_m)
    
    if len(segment) == 0:
        return {
            'Vmin': None,
            'Ventry': None,
            'Vexit': None,
            's_brake_onset': None,
            'a_long_peak': None
        }
    
    # Vmin: Minimum velocity in window
    Vmin = segment[v_col].min()
    
    # Ventry: Median velocity in 30m range before apex
    entry_mask = (segment[s_col] >= apex_s - 30) & (segment[s_col] < apex_s)
    if entry_mask.sum() > 0:
        Ventry = segment.loc[entry_mask, v_col].median()
    else:
        # Take velocity at left end of window
        left_mask = segment[s_col] < apex_s
        if left_mask.sum() > 0:
            Ventry = segment.loc[left_mask, v_col].iloc[-1]
        else:
            Ventry = segment[v_col].iloc[0]
    
    # Vexit: Median velocity in 30m range after apex
    exit_mask = (segment[s_col] > apex_s) & (segment[s_col] <= apex_s + 30)
    if exit_mask.sum() > 0:
        Vexit = segment.loc[exit_mask, v_col].median()
    else:
        # Take velocity at right end of window
        right_mask = segment[s_col] > apex_s
        if right_mask.sum() > 0:
            Vexit = segment.loc[right_mask, v_col].iloc[0]
        else:
            Vexit = segment[v_col].iloc[-1]
    
    # Brake onset and peak (before apex)
    before_apex_mask = segment[s_col] < apex_s
    if before_apex_mask.sum() > 0:
        before_apex = segment[before_apex_mask].copy()
        
        # Brake peak (minimum a_long)
        a_long_peak = before_apex[a_long_col].min()
        
        # Brake onset: most recent point satisfying a_long < brake_onset_thr_g * g
        brake_threshold = brake_onset_thr_g * G
        brake_mask = before_apex[a_long_col] < brake_threshold
        
        if brake_mask.sum() > 0:
            # Prioritize jerk < 0 points
            if jerk_col and jerk_col in before_apex.columns:
                jerk_mask = before_apex[jerk_col] < 0
                if jerk_mask.sum() > 0:
                    brake_candidates = before_apex[brake_mask & jerk_mask]
                else:
                    brake_candidates = before_apex[brake_mask]
            else:
                brake_candidates = before_apex[brake_mask]
            
            if len(brake_candidates) > 0:
                # Take closest to apex
                brake_candidates = brake_candidates.sort_values(s_col, ascending=False)
                s_brake_onset = brake_candidates[s_col].iloc[0]
            else:
                s_brake_onset = None
        else:
            s_brake_onset = None
    else:
        s_brake_onset = None
        a_long_peak = None
    
    return {
        'Vmin': Vmin,
        'Ventry': Ventry,
        'Vexit': Vexit,
        's_brake_onset': s_brake_onset,
        'a_long_peak': a_long_peak
    }


def generate_corner_cards(
    df: pd.DataFrame,
    apex_s_list: np.ndarray,
    lap: Optional[int] = None,
    segment_window_m: float = 60.0,
    brake_onset_thr_g: float = -0.15
) -> pd.DataFrame:
    """
    Generate corner cards.

    Args:
        df: DataFrame (must contain s, v_smooth, a_long columns)
        apex_s_list: Apex arc length position array
        lap: Lap number (if None, taken from df)
        segment_window_m: Window width (meters)
        brake_onset_thr_g: Brake onset threshold (G)

    Returns:
        Corner cards DataFrame
    """
    cards = []
    
    for turn_idx, apex_s in enumerate(apex_s_list, start=1):
        metrics = compute_corner_metrics(
            df,
            apex_s,
            jerk_col='jerk' if 'jerk' in df.columns else None,
            segment_window_m=segment_window_m,
            brake_onset_thr_g=brake_onset_thr_g
        )
        
        lap_num = lap if lap is not None else (df['lap'].iloc[0] if 'lap' in df.columns else 0)
        
        card = {
            'lap': lap_num,
            'turn': turn_idx,
            's_apex': apex_s,
            's_brake_onset': metrics['s_brake_onset'],
            'Ventry': metrics['Ventry'],
            'Vmin': metrics['Vmin'],
            'Vexit': metrics['Vexit'],
            'a_long_peak': metrics['a_long_peak']
        }
        
        cards.append(card)
    
    cards_df = pd.DataFrame(cards)
    
    logger.info(f"Generated {len(cards_df)} corner cards")
    
    return cards_df


def detect_events(
    corner_cards: pd.DataFrame,
    brake_thr_g: float = -0.28,
    conservative_entry_factor: float = 1.3
) -> pd.DataFrame:
    """
    Detect events (heavy braking, conservative entry, etc.).

    Args:
        corner_cards: Corner cards
        brake_thr_g: Heavy braking threshold (G)
        conservative_entry_factor: Conservative entry factor (Ventry < factor * Vmin)

    Returns:
        Events DataFrame
    """
    events = []
    
    for _, card in corner_cards.iterrows():
        lap = card['lap']
        turn = card['turn']
        
        # Heavy braking detection
        if pd.notna(card['a_long_peak']):
            a_long_peak_g = card['a_long_peak'] / G
            if a_long_peak_g < brake_thr_g:
                delta_s = None
                if pd.notna(card['s_brake_onset']):
                    delta_s = card['s_apex'] - card['s_brake_onset']
                
                events.append({
                    'lap': lap,
                    'turn': turn,
                    'type': 'heavy_brake',
                    'a_long_peak_g': a_long_peak_g,
                    'delta_s_to_apex_m': delta_s
                })
        
        # Conservative entry detection
        if pd.notna(card['Ventry']) and pd.notna(card['Vmin']):
            if card['Ventry'] < conservative_entry_factor * card['Vmin']:
                events.append({
                    'lap': lap,
                    'turn': turn,
                    'type': 'conservative_entry',
                    'Ventry': card['Ventry'],
                    'Vmin': card['Vmin']
                })
    
    events_df = pd.DataFrame(events)
    
    if len(events_df) > 0:
        event_counts = events_df['type'].value_counts()
        logger.info(f"Detected events: {dict(event_counts)}")
    else:
        logger.info("No events detected")
    
    return events_df


def process_all_laps(
    df: pd.DataFrame,
    reference_apex_s: np.ndarray,
    segment_window_m: float = 60.0,
    brake_onset_thr_g: float = -0.15,
    brake_thr_g: float = -0.28,
    conservative_entry_factor: float = 1.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all laps, generate corner cards and events.

    Args:
        df: DataFrame
        reference_apex_s: Reference lap apex positions
        segment_window_m: Window width (meters)
        brake_onset_thr_g: Brake onset threshold (G)
        brake_thr_g: Heavy braking threshold (G)
        conservative_entry_factor: Conservative entry factor

    Returns:
        (Corner cards DataFrame, events DataFrame)
    """
    all_cards = []
    
    if 'lap' not in df.columns or df['lap'].nunique() <= 1:
        # Single lap or global
        cards = generate_corner_cards(
            df,
            reference_apex_s,
            segment_window_m=segment_window_m,
            brake_onset_thr_g=brake_onset_thr_g
        )
        all_cards.append(cards)
    else:
        # Multiple laps
        # Get all lap numbers
        unique_laps = sorted(df['lap'].dropna().unique())
        
        for lap_num in unique_laps:
            lap_df = df[df['lap'] == lap_num].copy()
            if len(lap_df) < 10:  # Skip too short laps
                continue
            
            cards = generate_corner_cards(
                lap_df,
                reference_apex_s,
                lap=lap_num,
                segment_window_m=segment_window_m,
                brake_onset_thr_g=brake_onset_thr_g
            )
            all_cards.append(cards)
    
    corner_cards_df = pd.concat(all_cards, ignore_index=True)
    
    # Detectevents
    events_df = detect_events(
        corner_cards_df,
        brake_thr_g=brake_thr_g,
        conservative_entry_factor=conservative_entry_factor
    )
    
    return corner_cards_df, events_df


def auto_tune_apex_params(
    s: np.ndarray,
    kappa: np.ndarray,
    target_min: int = 12,
    target_max: int = 16,
    base_min_gap_m: float = 65.0,
    base_q: float = 0.78
) -> Tuple[dict, np.ndarray]:
    """
    Auto-tune parameters to hit target turn count range.

    Args:
        s: Arc length array
        kappa: Curvature array
        target_min: Target minimum turn count
        target_max: Target maximum turn count
        base_min_gap_m: Base minimum spacing
        base_q: Base quantile

    Returns:
        (Parameter dictionary, apex arc length position array)
    """
    # Parameter grid
    q_values = [0.75, 0.78, 0.80]
    min_gap_values = [55.0, 65.0, 75.0]
    prom_mult_values = [0.5, 0.6, 0.7]
    merge_gap_values = [40.0, 45.0, 55.0]
    
    # Stage 1: Fix min_gap, adjust q
    logger.info(f"[AUTO-TUNE] Stage 1: Adjusting q with fixed min_gap={base_min_gap_m}m")
    for q in q_values:
        apex_s, _, _ = detect_apexes_by_curvature(
            s, kappa, q=q, min_gap_m=base_min_gap_m,
            prom_mult=0.60, post_merge_gap_m=45.0
        )
        apex_count = len(apex_s)
        logger.info(f"[AUTO] q={q:.2f}, min_gap={base_min_gap_m:.1f}m, prom=0.60, merge=45.0m -> apex_count={apex_count}")
        if target_min <= apex_count <= target_max:
            return {
                'q': q,
                'min_gap_m': base_min_gap_m,
                'prom_mult': 0.60,
                'post_merge_gap_m': 45.0
            }, apex_s
    
    # Stage 2: Fix q, adjust min_gap
    logger.info(f"[AUTO-TUNE] Stage 2: Adjusting min_gap with fixed q={base_q}")
    for min_gap in min_gap_values:
        apex_s, _, _ = detect_apexes_by_curvature(
            s, kappa, q=base_q, min_gap_m=min_gap,
            prom_mult=0.60, post_merge_gap_m=45.0
        )
        apex_count = len(apex_s)
        logger.info(f"[AUTO] q={base_q:.2f}, min_gap={min_gap:.1f}m, prom=0.60, merge=45.0m -> apex_count={apex_count}")
        if target_min <= apex_count <= target_max:
            return {
                'q': base_q,
                'min_gap_m': min_gap,
                'prom_mult': 0.60,
                'post_merge_gap_m': 45.0
            }, apex_s
    
    # Stage 3: Adjust prom_mult and merge_gap
    logger.info(f"[AUTO-TUNE] Stage 3: Adjusting prom_mult and merge_gap")
    for prom_mult in prom_mult_values:
        for merge_gap in merge_gap_values:
            apex_s, _, _ = detect_apexes_by_curvature(
                s, kappa, q=base_q, min_gap_m=base_min_gap_m,
                prom_mult=prom_mult, post_merge_gap_m=merge_gap
            )
            apex_count = len(apex_s)
            logger.info(f"[AUTO] q={base_q:.2f}, min_gap={base_min_gap_m:.1f}m, prom={prom_mult:.2f}, merge={merge_gap:.1f}m -> apex_count={apex_count}")
            if target_min <= apex_count <= target_max:
                return {
                    'q': base_q,
                    'min_gap_m': base_min_gap_m,
                    'prom_mult': prom_mult,
                    'post_merge_gap_m': merge_gap
                }, apex_s
    
    # If none hit target, return closest parameters
    logger.warning(f"[AUTO-TUNE] No combination found in target range [{target_min}, {target_max}], using default")
    apex_s, _, _ = detect_apexes_by_curvature(
        s, kappa, q=base_q, min_gap_m=base_min_gap_m,
        prom_mult=0.60, post_merge_gap_m=45.0
    )
    return {
        'q': base_q,
        'min_gap_m': base_min_gap_m,
        'prom_mult': 0.60,
        'post_merge_gap_m': 45.0
    }, apex_s


def scan_apex_params(
    s: np.ndarray,
    kappa: np.ndarray,
    target_min: int = 12,
    target_max: int = 16
) -> pd.DataFrame:
    """
    Grid scan parameter combinations and generate recommendation table.

    Args:
        s: Arc length array
        kappa: Curvature array
        target_min: Target minimum turn count
        target_max: Target maximum turn count

    Returns:
        Scan result DataFrame
    """
    q_values = [0.75, 0.78, 0.80]
    min_gap_values = [55.0, 65.0, 75.0]
    prom_mult_values = [0.5, 0.6, 0.7]
    merge_gap_values = [40.0, 45.0, 55.0]
    
    results = []
    
    logger.info(f"[SCAN] Starting parameter grid scan...")
    
    for q in q_values:
        for min_gap in min_gap_values:
            for prom_mult in prom_mult_values:
                for merge_gap in merge_gap_values:
                    apex_s, _, _ = detect_apexes_by_curvature(
                        s, kappa, q=q, min_gap_m=min_gap,
                        prom_mult=prom_mult, post_merge_gap_m=merge_gap
                    )
                    apex_count = len(apex_s)
                    in_target = target_min <= apex_count <= target_max
                    
                    results.append({
                        'q': q,
                        'min_gap_m': min_gap,
                        'prom_mult': prom_mult,
                        'merge_gap_m': merge_gap,
                        'apex_count': apex_count,
                        'in_target': in_target
                    })
    
    df = pd.DataFrame(results)
    
    # Output candidate combinations in target range
    candidates = df[df['in_target']].copy()
    if len(candidates) > 0:
        logger.info(f"[SCAN] Found {len(candidates)} parameter combinations in target range [{target_min}, {target_max}]")
        logger.info("[SCAN] Top candidates:")
        for idx, row in candidates.head(10).iterrows():
            logger.info(f"  q={row['q']:.2f}, min_gap={row['min_gap_m']:.1f}m, "
                       f"prom={row['prom_mult']:.2f}, merge={row['merge_gap_m']:.1f}m -> {int(row['apex_count'])} apexes")
    else:
        logger.warning(f"[SCAN] No combinations found in target range [{target_min}, {target_max}]")
    
    return df


def detect_relative_events(
    corner_cards: pd.DataFrame,
    ref_lap: Optional[int],
    late_brake_thr_m: float,
    early_brake_thr_m: float,
    overslow_thr_ms: float,
    weak_exit_thr_ms: float,
    weak_exit_adecel_g: float
) -> pd.DataFrame:
    """
    Detect new event types based on deltas relative to reference lap.

    Args:
        corner_cards: Corner cards (must contain d_ series delta columns)
        ref_lap: Benchmark lap number
        late_brake_thr_m: Late braking threshold (meters)
        early_brake_thr_m: Early braking threshold (meters)
        overslow_thr_ms: Apex speed difference threshold (m/s)
        weak_exit_thr_ms: Exit speed difference threshold (m/s)
        weak_exit_adecel_g: Exit longitudinal acceleration threshold (g)

    Returns:
        New events DataFrame
    """
    required_cols = {
        'lap',
        'turn',
        'd_delta_s',
        'd_Vmin',
        'd_Vexit',
        'delta_s_to_apex_m',
        'Ventry',
        'Vmin',
        'Vexit',
        'a_long_peak'
    }

    if corner_cards is None or corner_cards.empty:
        return pd.DataFrame()

    missing = required_cols.difference(corner_cards.columns)
    if missing:
        logger.warning("Corner cards missing required columns for relative events: %s", missing)
        return pd.DataFrame()

    if ref_lap is None or ref_lap not in corner_cards['lap'].unique():
        logger.warning("Reference lap %s not present in corner cards; skipping relative event detection", ref_lap)
        return pd.DataFrame()

    events = []
    type_counters: defaultdict[str, int] = defaultdict(int)

    for row in corner_cards.itertuples(index=False):
        if getattr(row, 'lap') == ref_lap:
            continue

        lap = getattr(row, 'lap')
        turn = getattr(row, 'turn')
        d_delta_s = getattr(row, 'd_delta_s')
        d_ventry = getattr(row, 'd_Ventry', np.nan) if hasattr(row, 'd_Ventry') else np.nan
        d_vmin = getattr(row, 'd_Vmin')
        d_vexit = getattr(row, 'd_Vexit')
        delta_s_to_apex_m = getattr(row, 'delta_s_to_apex_m')
        ventry = getattr(row, 'Ventry')
        vmin = getattr(row, 'Vmin')
        vexit = getattr(row, 'Vexit')
        a_long_peak = getattr(row, 'a_long_peak')
        a_long_peak_g = a_long_peak / G if pd.notna(a_long_peak) else np.nan

        def make_event(event_type: str, severity: float, ref_flags: dict, extra: dict) -> None:
            if severity <= 0 or not np.isfinite(severity):
                return
            type_counters[event_type] += 1
            event_id = f"{lap}-{turn}-{event_type}-{type_counters[event_type]}"
            payload = {
                'lap': lap,
                'turn': turn,
                'type': event_type,
                'event_id': event_id,
                'severity': severity,
                'ref_flags': json.dumps(ref_flags, ensure_ascii=False),
                'source': 'rule_v1',
                'delta_s_to_apex_m': delta_s_to_apex_m,
                'Ventry': ventry,
                'Vmin': vmin,
                'Vexit': vexit,
                'd_delta_s': d_delta_s,
                'd_Ventry': d_ventry,
                'd_Vmin': d_vmin,
                'd_Vexit': d_vexit,
                'a_long_peak_g': a_long_peak_g
            }
            payload.update(extra)
            events.append(payload)

        # late_brake
        if pd.notna(d_delta_s) and d_delta_s < -late_brake_thr_m:
            severity = abs(d_delta_s) - late_brake_thr_m
            make_event(
                'late_brake',
                severity,
                {'delta_s': 'late'},
                {}
            )

        # early_brake
        if pd.notna(d_delta_s) and d_delta_s > early_brake_thr_m:
            severity = d_delta_s - early_brake_thr_m
            make_event(
                'early_brake',
                severity,
                {'delta_s': 'early'},
                {}
            )

        # overslow
        if pd.notna(d_vmin) and d_vmin < -overslow_thr_ms:
            severity = abs(d_vmin) - overslow_thr_ms
            make_event(
                'overslow',
                severity,
                {'vmin': 'lower'},
                {}
            )

        # weak_exit
        if pd.notna(d_vexit) and d_vexit < -weak_exit_thr_ms:
            adecel_g = abs(a_long_peak_g) if pd.notna(a_long_peak_g) else np.nan
            if pd.notna(adecel_g) and adecel_g < weak_exit_adecel_g:
                speed_component = abs(d_vexit) - weak_exit_thr_ms
                adecel_component = weak_exit_adecel_g - adecel_g
                severity = max(speed_component, 0.0) + max(adecel_component, 0.0)
                make_event(
                    'weak_exit',
                    severity,
                    {'vexit': 'lower', 'a_long_peak': 'weak'},
                    {'a_long_peak_g': a_long_peak_g}
                )

    if not events:
        return pd.DataFrame()

    events_df = pd.DataFrame(events)
    return events_df


def export_curvature_debug(
    s: np.ndarray,
    kappa: np.ndarray,
    peaks_before_merge: Optional[np.ndarray] = None,
    peaks_after_merge: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Export curvature debug data.

    Args:
        s: Arc length array
        kappa: Curvature array
        peaks_before_merge: Peak indices before merging (optional)
        peaks_after_merge: Peak indices after merging (optional)
        output_path: Output path (optional)

    Returns:
        DebugDataFrame
    """
    df = pd.DataFrame({
        's': s,
        'kappa_s': kappa,
        'is_peak': np.zeros(len(s), dtype=int)
    })
    
    if peaks_after_merge is not None:
        df.loc[peaks_after_merge, 'is_peak'] = 1
    
    if peaks_before_merge is not None:
        df['peak_s_before_merge'] = np.nan
        df.loc[peaks_before_merge, 'peak_s_before_merge'] = s[peaks_before_merge]
        
        if peaks_after_merge is not None:
            df['peak_s_after_merge'] = np.nan
            df.loc[peaks_after_merge, 'peak_s_after_merge'] = s[peaks_after_merge]
    
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Exported curvature debug data to {output_path}")
    
    return df

