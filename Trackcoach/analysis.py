"""Issue analysis module: aggregate events, compute Top Issues, generate analysis reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import shutil

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import stability score computation function
try:
    from trackcoach.stability import compute_composite_stability_score
except ImportError:
    compute_composite_stability_score = None
    logger.warning("Unable to import stability score computation function")

ABS_EPS = 1e-9


def parse_turn_series(series: pd.Series) -> pd.Series:
    """Parse turn series to numeric type."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def compute_deltas(corner_cards: pd.DataFrame, benchmark_lap: int) -> Tuple[pd.DataFrame, bool]:
    """
    Compute deltas between corner cards and benchmark lap.
    
    Args:
        corner_cards: Corner cards DataFrame
        benchmark_lap: Benchmark lap number
    
    Returns:
        (Corner cards with deltas, whether deltas were recomputed)
    """
    required_cols = {"turn", "lap"}
    missing = required_cols - set(corner_cards.columns)
    if missing:
        raise ValueError(f"corner_cards missing required columns: {', '.join(sorted(missing))}")

    if pd.isna(benchmark_lap):
        raise ValueError("benchmark_lap is empty, cannot compute deltas")

    cc = corner_cards.copy()
    cc["turn"] = parse_turn_series(cc["turn"])
    cc["lap"] = pd.to_numeric(cc["lap"], errors="coerce")

    if cc["turn"].isna().all():
        raise ValueError("All turn values in corner_cards are empty, cannot align by turn number")

    delta_base = None
    if "delta_s_to_apex_m" in cc.columns:
        delta_base = cc["delta_s_to_apex_m"]
    elif {"s_apex", "s_brake_onset"}.issubset(cc.columns):
        delta_base = cc["s_apex"] - cc["s_brake_onset"]
    else:
        raise ValueError("corner_cards missing delta_s_to_apex_m or (s_apex, s_brake_onset) for computing delta_s")

    cc["_delta_s_base"] = delta_base

    pb_cards = cc[cc["lap"] == benchmark_lap].dropna(subset=["turn"]).copy()
    if pb_cards.empty:
        raise ValueError(f"PB data for lap == {benchmark_lap} not found, cannot compute deltas")

    pb_cards = (
        pb_cards.sort_values(["turn", "lap"])
        .drop_duplicates(subset=["turn"], keep="first")
        .set_index("turn")
    )

    ref_cols = {
        "_delta_s_base": "pb_delta_s",
        "Ventry": "pb_Ventry",
        "Vmin": "pb_Vmin",
        "Vexit": "pb_Vexit",
        "a_long_peak": "pb_a_long_peak",
    }

    available_cols = [c for c in ref_cols if c in pb_cards.columns]
    if "_delta_s_base" not in available_cols:
        raise ValueError("PB data missing delta_s information, cannot compute d_delta_s")

    merge_df = pb_cards[available_cols].rename(columns={c: ref_cols[c] for c in available_cols})
    cc = cc.merge(merge_df, left_on="turn", right_index=True, how="left")

    recomputed = False
    # Uniformly generate delta columns
    if "d_delta_s" not in cc.columns:
        cc["d_delta_s"] = np.nan
        recomputed = True
    cc["d_delta_s"] = cc["_delta_s_base"] - cc["pb_delta_s"]

    diff_map = {
        "Ventry": "d_Ventry",
        "Vmin": "d_Vmin",
        "Vexit": "d_Vexit",
        "a_long_peak": "d_a_long_peak",
    }
    for src_col, diff_col in diff_map.items():
        if src_col not in cc.columns or f"pb_{src_col}" not in cc.columns:
            continue
        cc[diff_col] = cc[src_col] - cc[f"pb_{src_col}"]
        if diff_col not in corner_cards.columns:
            recomputed = True

    helper_cols = [col for col in cc.columns if col.startswith("pb_")]
    cc.drop(columns=helper_cols + ["_delta_s_base"], inplace=True, errors="ignore")
    return cc, recomputed


def robust_median_abs(series: pd.Series) -> float:
    """Compute robust estimate of median absolute value."""
    if series is None:
        return 0.0
    data = series.dropna()
    if data.empty:
        return 0.0
    return float(np.median(np.abs(data.to_numpy(dtype=float))))


def robust_iqr_abs(series: pd.Series) -> float:
    """Compute robust estimate of absolute value IQR."""
    if series is None:
        return 0.0
    data = np.abs(series.dropna().to_numpy(dtype=float))
    if data.size == 0:
        return 0.0
    q75 = np.percentile(data, 75)
    q25 = np.percentile(data, 25)
    return float(q75 - q25)


def aggregate_turn_stats(
    corner_cards: pd.DataFrame,
    events: pd.DataFrame,
    benchmark_lap: int
) -> Tuple[pd.DataFrame, bool]:
    """
    Aggregate statistics by turn number.
    
    Args:
        corner_cards: Corner cards DataFrame (must include delta columns)
        events: Events DataFrame
        benchmark_lap: Benchmark lap number
    
    Returns:
        (Aggregated statistics DataFrame by turn, whether PB lap data is included)
    """
    non_pb_cards = corner_cards[corner_cards["lap"] != benchmark_lap].copy()
    pb_included = bool((non_pb_cards["lap"] == benchmark_lap).any())

    grouped = non_pb_cards.dropna(subset=["turn"]).groupby("turn", dropna=True)
    records: List[dict] = []

    events = events.copy()
    if not events.empty:
        events["turn"] = parse_turn_series(events["turn"])
        if "severity" in events.columns:
            events["severity"] = pd.to_numeric(events["severity"], errors="coerce").fillna(0.0)
        else:
            events["severity"] = 0.0
        events["event_id"] = events["event_id"].fillna("").astype(str) if "event_id" in events.columns else ""
    else:
        events = pd.DataFrame(columns=["turn", "type", "severity", "event_id"])

    events_by_turn = {turn: df for turn, df in events.groupby("turn")} if not events.empty else {}

    for turn, df in grouped:
        med_delta_s = robust_median_abs(df.get("d_delta_s"))
        med_vmin = robust_median_abs(df.get("d_Vmin"))
        iqr_delta_s = robust_iqr_abs(df.get("d_delta_s"))
        iqr_vmin = robust_iqr_abs(df.get("d_Vmin"))

        turn_events = events_by_turn.get(turn, pd.DataFrame(columns=events.columns))
        events_count = int(len(turn_events))
        if events_count > 0:
            severity_vals = turn_events["severity"].to_numpy(dtype=float)
            max_severity = float(np.max(severity_vals))
            top_types = ", ".join(
                turn_events["type"].value_counts().head(3).index.astype(str).tolist()
            ) if "type" in turn_events.columns else "-"
            sample_idx = int(np.argmax(severity_vals))
            sample_event_id = (
                turn_events.iloc[sample_idx]["event_id"]
                if "event_id" in turn_events.columns and events_count > sample_idx
                else ""
            )
        else:
            max_severity = 0.0
            top_types = "-"
            sample_event_id = ""

        records.append(
            {
                "turn": int(turn),
                "med_|d_delta_s|": med_delta_s,
                "med_|d_Vmin|": med_vmin,
                "iqr_|d_delta_s|": iqr_delta_s,
                "iqr_|d_Vmin|": iqr_vmin,
                "events_count": events_count,
                "max_severity_by_turn": max_severity,
                "top_types": top_types,
                "sample_event_id": sample_event_id,
            }
        )

    agg_df = pd.DataFrame(records)
    if not agg_df.empty:
        agg_df.sort_values(by="turn", inplace=True)
    return agg_df, pb_included


def norm95(x: pd.Series, q95: float) -> pd.Series:
    """Normalize to 95th percentile."""
    denom = max(q95, 1e-6)
    return np.minimum(x / denom, 1.0)


def safe_percentile(series: pd.Series, q: float) -> float:
    """Safely compute percentile."""
    data = series.dropna().to_numpy(dtype=float)
    if data.size == 0:
        return 0.0
    return float(np.percentile(data, q))


def compute_scores(turn_stats: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Compute issue scores.
    
    Args:
        turn_stats: Aggregated statistics DataFrame by turn
    
    Returns:
        (DataFrame with scores, quantile information dictionary)
    """
    if turn_stats.empty:
        return turn_stats.assign(
            comp_s=0.0, comp_v=0.0, comp_sev=0.0, comp_freq=0.0, score=0.0
        ), {"q95_delta_s": 0.0, "q95_vmin": 0.0, "q95_severity": 0.0}

    q95_delta_s = safe_percentile(turn_stats["med_|d_delta_s|"], 95)
    q95_vmin = safe_percentile(turn_stats["med_|d_Vmin|"], 95)
    q95_severity = safe_percentile(turn_stats["max_severity_by_turn"], 95)

    comp_s = 10.0 * norm95(turn_stats["med_|d_delta_s|"], q95_delta_s)
    comp_v = 10.0 * norm95(turn_stats["med_|d_Vmin|"], q95_vmin)
    comp_sev = 10.0 * norm95(turn_stats["max_severity_by_turn"], q95_severity)
    comp_freq = np.minimum(np.log1p(turn_stats["events_count"]), 3.0)

    scored = turn_stats.copy()
    scored["comp_s"] = comp_s.fillna(0.0)
    scored["comp_v"] = comp_v.fillna(0.0)
    scored["comp_sev"] = comp_sev.fillna(0.0)
    scored["comp_freq"] = comp_freq.fillna(0.0)
    scored["score"] = (
        1.0 * scored["comp_s"]
        + 0.8 * scored["comp_v"]
        + 0.6 * scored["comp_sev"]
        + 0.3 * scored["comp_freq"]
    )
    scored.sort_values(by="score", ascending=False, inplace=True)
    scored.reset_index(drop=True, inplace=True)

    return scored, {
        "q95_delta_s": q95_delta_s,
        "q95_vmin": q95_vmin,
        "q95_severity": q95_severity,
    }


class IssueAnalyzer:
    """Issue analyzer: aggregate events and generate Top Issues reports."""
    
    def __init__(self):
        """Initialize issue analyzer."""
        pass
    
    def load_stability_scores(
        self,
        stability_report_path: Optional[Path] = None
    ) -> Optional[Dict[int, float]]:
        """
        Load stability scores.
        
        Args:
            stability_report_path: Stability report CSV file path
        
        Returns:
            Dictionary of stability scores indexed by turn (higher values = less stable), or None if unable to load
        """
        if stability_report_path is None or not stability_report_path.exists():
            return None
        
        try:
            stability_df = pd.read_csv(stability_report_path)
            
            # Compute composite stability score
            if compute_composite_stability_score:
                stability_scores = compute_composite_stability_score(stability_df)
                # Convert to dictionary
                return {int(row['turn']): float(stability_scores.iloc[idx]) 
                       for idx, row in stability_df.iterrows()}
            else:
                # If function not imported, manually compute simplified stability score
                stability_dict = {}
                for _, row in stability_df.iterrows():
                    turn = int(row['turn'])
                    # Use weighted IQR of key metrics
                    score = 0.0
                    if pd.notna(row.get('d_delta_s_iqr', np.nan)):
                        score += 0.5 * float(row['d_delta_s_iqr'])
                    if pd.notna(row.get('d_Vmin_iqr', np.nan)):
                        score += 0.3 * float(row['d_Vmin_iqr'])
                    if pd.notna(row.get('d_Ventry_iqr', np.nan)):
                        score += 0.1 * float(row['d_Ventry_iqr'])
                    if pd.notna(row.get('d_a_long_peak_iqr', np.nan)):
                        score += 0.1 * float(row['d_a_long_peak_iqr'])
                    stability_dict[turn] = score
                
                # Normalize to [0, 1]
                if stability_dict:
                    max_score = max(stability_dict.values())
                    if max_score > 0:
                        stability_dict = {k: v / max_score for k, v in stability_dict.items()}
                
                return stability_dict
        except Exception as e:
            logger.warning(f"Failed to load stability scores: {e}")
            return None
    
    def analyze(
        self,
        corner_cards: pd.DataFrame,
        events: pd.DataFrame,
        benchmark_lap: int,
        topk: int = 5,
        stability_report_path: Optional[Path] = None,
        stability_weight: float = 0.3
    ) -> Dict:
        """
        Execute complete issue analysis workflow.
        
        Args:
            corner_cards: Corner cards DataFrame
            events: Events DataFrame
            benchmark_lap: Benchmark lap number
            topk: Number of Top Issues
            stability_report_path: Stability report CSV file path (optional)
            stability_weight: Stability score weight (0-1, default 0.3, i.e., stability accounts for 30% of composite score)
        
        Returns:
            Analysis result dictionary containing:
            - corner_cards_with_delta: Corner cards with deltas
            - turn_stats: Aggregated statistics by turn
            - top_issues: Top Issues DataFrame (with composite scores)
            - quantiles: Quantile information
            - pb_included: Whether PB lap data is included
        """
        # Compute deltas
        corner_cards_with_delta, recomputed = compute_deltas(corner_cards, benchmark_lap)
        
        # Aggregate statistics
        turn_stats, pb_included = aggregate_turn_stats(
            corner_cards_with_delta, events, benchmark_lap
        )
        
        # Compute issue scores
        scored, quantiles = compute_scores(turn_stats)
        
        # Load and integrate stability scores
        stability_scores = self.load_stability_scores(stability_report_path)
        
        if stability_scores is not None and len(stability_scores) > 0:
            logger.info(f"Loaded stability scores, will integrate into composite score (weight: {stability_weight:.0%})")
            
            # Normalize issue scores to [0, 1] range (preserve relative relationships)
            issue_scores_normalized = scored['score'].copy()
            max_issue_score = issue_scores_normalized.max()
            min_issue_score = issue_scores_normalized.min()
            if max_issue_score > min_issue_score:
                issue_scores_normalized = (issue_scores_normalized - min_issue_score) / (max_issue_score - min_issue_score)
            else:
                issue_scores_normalized = pd.Series(0.0, index=issue_scores_normalized.index)
            
            # Normalize stability scores to [0, 1] range
            if len(stability_scores) > 0:
                max_stability_score = max(stability_scores.values())
                min_stability_score = min(stability_scores.values())
                if max_stability_score > min_stability_score:
                    stability_scores_normalized = {
                        k: (v - min_stability_score) / (max_stability_score - min_stability_score)
                        for k, v in stability_scores.items()
                    }
                else:
                    stability_scores_normalized = {k: 0.0 for k in stability_scores.keys()}
            else:
                stability_scores_normalized = {}
            
            # Compute composite scores
            composite_scores = []
            stability_scores_list = []
            for idx, row in scored.iterrows():
                turn = int(row['turn'])
                issue_score = issue_scores_normalized.iloc[idx]
                stability_score_norm = stability_scores_normalized.get(turn, 0.0)
                stability_score_raw = stability_scores.get(turn, 0.0)
                
                # Composite score = (1 - stability_weight) * issue_score + stability_weight * stability_score
                composite_score_norm = (1 - stability_weight) * issue_score + stability_weight * stability_score_norm
                
                # Rescale back to original score range
                if max_issue_score > min_issue_score:
                    composite_score_rescaled = min_issue_score + composite_score_norm * (max_issue_score - min_issue_score)
                else:
                    composite_score_rescaled = scored['score'].iloc[idx]
                
                composite_scores.append(composite_score_rescaled)
                stability_scores_list.append(stability_score_raw)
            
            # Add stability score column
            scored['stability_score'] = stability_scores_list
            scored['composite_score'] = composite_scores
            
            # Re-sort by composite score
            scored.sort_values(by='composite_score', ascending=False, inplace=True)
            scored.reset_index(drop=True, inplace=True)
            
            logger.info("Computed composite scores (issue score + stability score)")
        else:
            # If no stability data, composite score equals issue score
            scored['stability_score'] = 0.0
            scored['composite_score'] = scored['score']
            logger.info("Stability data not found, composite score equals issue score")
        
        # Extract Top Issues (based on composite score)
        topk = max(1, topk)
        top_issues = scored.head(topk).copy()
        defaults = {
            "turn": 0,
            "score": 0.0,
            "stability_score": 0.0,
            "composite_score": 0.0,
            "med_|d_delta_s|": 0.0,
            "med_|d_Vmin|": 0.0,
            "events_count": 0,
            "top_types": "",
            "sample_event_id": "",
            "comp_s": 0.0,
            "comp_v": 0.0,
            "comp_sev": 0.0,
            "comp_freq": 0.0,
        }
        for col, default in defaults.items():
            if col not in top_issues.columns:
                top_issues[col] = default
        
        # Ensure column order (prioritize score-related columns)
        priority_cols = ["turn", "composite_score", "score"]
        if stability_scores is not None and len(stability_scores) > 0:
            priority_cols.append("stability_score")
        
        # Other columns
        other_cols = [
            "med_|d_delta_s|", "med_|d_Vmin|", "events_count", "top_types",
            "sample_event_id", "comp_s", "comp_v", "comp_sev", "comp_freq"
        ]
        
        # Build final column order
        final_cols = []
        for col in priority_cols + other_cols:
            if col in top_issues.columns:
                final_cols.append(col)
        
        # Add other columns that may exist
        for col in top_issues.columns:
            if col not in final_cols:
                final_cols.append(col)
        
        top_issues = top_issues[final_cols]
        
        return {
            "corner_cards_with_delta": corner_cards_with_delta,
            "turn_stats": turn_stats,
            "top_issues": top_issues,
            "quantiles": quantiles,
            "pb_included": pb_included,
            "recomputed": recomputed,
            "stability_integrated": stability_scores is not None
        }
    
    def save_top_issues(
        self,
        top_issues: pd.DataFrame,
        output_path: Path,
        backup: bool = True
    ) -> None:
        """Save Top Issues to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if backup and output_path.exists():
            bak_path = output_path.with_suffix(output_path.suffix + ".bak")
            shutil.copy2(output_path, bak_path)
        top_issues.to_csv(output_path, index=False)
        logger.info(f"Saved top issues to {output_path}")
    
    def save_top_issues_plot(
        self,
        top_issues: pd.DataFrame,
        output_path: Path
    ) -> Optional[Path]:
        """Save Top Issues component stacked bar chart."""
        if top_issues.empty:
            return None
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping top issues plot generation")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            bak_path = output_path.with_suffix(output_path.suffix + ".bak")
            shutil.copy2(output_path, bak_path)

        turns = [f"T{int(t)}" for t in top_issues["turn"]]
        components = [
            ("comp_s", "Component Δs", "#1f77b4"),
            ("comp_v", "Component ΔVmin", "#ff7f0e"),
            ("comp_sev", "Component Severity", "#d62728"),
            ("comp_freq", "Component Frequency", "#2ca02c"),
        ]

        indices = np.arange(len(turns))
        cumulative = np.zeros(len(turns), dtype=float)

        fig, ax = plt.subplots(figsize=(8, 4.2))
        for column, label, color in components:
            values = top_issues[column].to_numpy(dtype=float)
            ax.bar(indices, values, bottom=cumulative, label=label, color=color, width=0.6, edgecolor="white", linewidth=0.6)
            cumulative += values

        ax.set_xticks(indices)
        ax.set_xticklabels(turns)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Score contribution")
        ax.set_title("Top Issues Score Breakdown")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=144, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved top issues plot to {output_path}")
        return output_path
    
    def save_summary_markdown(
        self,
        events: pd.DataFrame,
        turn_stats: pd.DataFrame,
        top_issues: pd.DataFrame,
        output_path: Path,
        top_hot_turns: int = 10,
        plot_relative_path: Optional[str] = None,
        backup: bool = True
    ) -> None:
        """Save events summary Markdown report."""
        lines: List[str] = ["# Events Summary", ""]

        # Table 1
        lines.append("## Table 1: Global Event Type Counts")
        lines.append("")

        if plot_relative_path:
            lines.append(f"![Top Issues Components]({plot_relative_path})")
            lines.append("")
        if events.empty or "type" not in events.columns:
            self._format_table(lines, ["Type", "Count"], [["(None)", "0"]])
        else:
            type_counts = events["type"].value_counts().sort_values(ascending=False)
            rows = [[etype, f"{int(count)}"] for etype, count in type_counts.items()]
            self._format_table(lines, ["Type", "Count"], rows)
        lines.append("")

        # Table 2
        lines.append("## Table 2: Top 10 Hotspots by Turn")
        lines.append("")
        if turn_stats.empty:
            self._format_table(lines, ["Turn", "Events", "Top Types", "Max Severity"], [["-", "0", "-", "0.0"]])
        else:
            ranked = turn_stats.sort_values(by="events_count", ascending=False).head(top_hot_turns)
            rows = [
                [f"T{int(row.turn)}", f"{int(row.events_count)}", row.top_types or "-", f"{row.max_severity_by_turn:.2f}"]
                for row in ranked.itertuples()
            ]
            self._format_table(lines, ["Turn", "Events", "Top Types", "Max Severity"], rows)
        lines.append("")

        # Table 3
        lines.append("## Table 3: Top Issues Component Breakdown")
        lines.append("")
        if top_issues.empty:
            self._format_table(lines, ["Turn", "Score", "Comp_s", "Comp_v", "Comp_sev", "Comp_freq"], [["-", "0.0", "0.0", "0.0", "0.0", "0.0"]])
        else:
            rows = [
                [
                    f"T{int(row.turn)}",
                    f"{row.score:.2f}",
                    f"{row.comp_s:.2f}",
                    f"{row.comp_v:.2f}",
                    f"{row.comp_sev:.2f}",
                    f"{row.comp_freq:.2f}",
                ]
                for row in top_issues.itertuples()
            ]
            self._format_table(lines, ["Turn", "Score", "Comp_s", "Comp_v", "Comp_sev", "Comp_freq"], rows)
        lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if backup and output_path.exists():
            bak_path = output_path.with_suffix(output_path.suffix + ".bak")
            shutil.copy2(output_path, bak_path)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved events summary to {output_path}")
    
    def _format_table(self, lines: List[str], headers: List[str], rows: List[List[str]]) -> None:
        """Format Markdown table."""
        header_line = "| " + " | ".join(headers) + " |"
        align_line = "| " + " | ".join("---:" if h.endswith(":") else "---" for h in headers) + " |"
        lines.append(header_line)
        lines.append(align_line)
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")


