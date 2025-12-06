"""Historical comparison module: support multi-session data comparison analysis."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SessionComparison:
    """Session comparison analyzer."""
    
    def __init__(self):
        """Initialize comparison analyzer."""
        self.sessions: List[Dict[str, Any]] = []
    
    def add_session(
        self,
        session_name: str,
        session_dir: Path,
        session_date: Optional[str] = None
    ) -> bool:
        """
        Add a session to comparison list.
        
        Args:
            session_name: Session name (e.g., "Session 1", "2024-01-15")
            session_dir: Session output directory path
            session_date: Session date (optional, for sorting)
        
        Returns:
            Whether successfully added
        """
        session_dir = Path(session_dir)
        
        # Check required files
        top_issues_path = session_dir / 'top_issues.csv'
        events_path = session_dir / 'events.csv'
        corner_cards_path = session_dir / 'corner_cards.csv'
        benchmark_path = session_dir / 'benchmark_lap.json'
        
        if not top_issues_path.exists():
            logger.warning(f"Session {session_name} missing top_issues.csv, skip")
            return False
        
        session_data = {
            'name': session_name,
            'dir': session_dir,
            'date': session_date or datetime.now().strftime('%Y-%m-%d'),
            'top_issues_path': top_issues_path,
            'events_path': events_path if events_path.exists() else None,
            'corner_cards_path': corner_cards_path if corner_cards_path.exists() else None,
            'benchmark_path': benchmark_path if benchmark_path.exists() else None
        }
        
        # Load benchmark lap information
        if benchmark_path.exists():
            try:
                with open(benchmark_path, 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)
                    session_data['benchmark_lap'] = benchmark_data.get('lap')
                    session_data['benchmark_laptime'] = benchmark_data.get('laptime_s')
                    session_data['is_historical_pb'] = benchmark_data.get('is_historical_pb', False)
                    session_data['historical_pb'] = benchmark_data.get('historical_pb')
            except Exception as e:
                logger.warning(f"Failed to load benchmark lap information: {e}")
        
        self.sessions.append(session_data)
        logger.info(f"Added Session: {session_name} ({session_data['date']})")
        return True
    
    def compare_top_issues(self) -> pd.DataFrame:
        """
        Compare Top Issues across sessions.
        
        Returns:
            Comparison result DataFrame
        """
        if len(self.sessions) < 2:
            logger.warning("Need at least 2 sessions to perform comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for session in self.sessions:
            try:
                top_issues_df = pd.read_csv(session['top_issues_path'])
                
                for _, row in top_issues_df.iterrows():
                    turn = int(row['turn'])
                    composite_score = row.get('composite_score', row.get('score', 0))
                    issue_score = row.get('score', 0)
                    stability_score = row.get('stability_score', 0)
                    events_count = row.get('events_count', 0)
                    
                    comparison_data.append({
                        'session': session['name'],
                        'date': session['date'],
                        'turn': turn,
                        'composite_score': float(composite_score) if pd.notna(composite_score) else 0,
                        'issue_score': float(issue_score) if pd.notna(issue_score) else 0,
                        'stability_score': float(stability_score) if pd.notna(stability_score) else 0,
                        'events_count': int(events_count) if pd.notna(events_count) else 0
                    })
            except Exception as e:
                logger.warning(f"Failed to load session {session['name']} data: {e}")
        
        if not comparison_data:
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def compare_laptimes(self) -> pd.DataFrame:
        """
        Compare lap times across sessions.
        
        Returns:
            Lap time comparison DataFrame
        """
        laptime_data = []
        
        for session in self.sessions:
            if 'benchmark_laptime' in session and session['benchmark_laptime']:
                laptime_data.append({
                    'session': session['name'],
                    'date': session['date'],
                    'laptime_s': float(session['benchmark_laptime']),
                    'lap': session.get('benchmark_lap', None)
                })
        
        if not laptime_data:
            return pd.DataFrame()
        
        return pd.DataFrame(laptime_data)
    
    def generate_trend_summary(self) -> Dict[str, Any]:
        """
        Generate trend analysis summary.
        
        Returns:
            Dictionary containing trend analysis results
        """
        summary = {
            'laptime_trend': None,
            'top_issues_trend': {},
            'overall_improvement': None
        }
        
        sorted_sessions = sorted(self.sessions, key=lambda x: x['date'])
        if len(sorted_sessions) < 2:
            return summary
        
        # Lap time trend
        laptime_df = self.compare_laptimes()
        if not laptime_df.empty and len(laptime_df) >= 2:
            sorted_laptime = laptime_df.sort_values('date')
            first_laptime = sorted_laptime.iloc[0]['laptime_s']
            last_laptime = sorted_laptime.iloc[-1]['laptime_s']
            improvement = first_laptime - last_laptime
            
            summary['laptime_trend'] = {
                'first': float(first_laptime),
                'last': float(last_laptime),
                'improvement': float(improvement),
                'improvement_pct': float(improvement / first_laptime * 100) if first_laptime > 0 else 0,
                'trend': 'improved' if improvement > 0 else 'worsened' if improvement < 0 else 'stable'
            }
        
        # Top Issues trend
        comparison_df = self.compare_top_issues()
        if not comparison_df.empty:
            for turn in comparison_df['turn'].unique():
                turn_data = comparison_df[comparison_df['turn'] == turn].sort_values('date')
                if len(turn_data) >= 2:
                    first_score = turn_data.iloc[0]['composite_score']
                    last_score = turn_data.iloc[-1]['composite_score']
                    change = first_score - last_score
                    
                    summary['top_issues_trend'][int(turn)] = {
                        'first': float(first_score),
                        'last': float(last_score),
                        'change': float(change),
                        'trend': 'improved' if change > 0 else 'worsened' if change < 0 else 'stable'
                    }
        
        # Overall improvement status
        if summary['laptime_trend']:
            summary['overall_improvement'] = summary['laptime_trend']['trend']
        
        return summary
    
    def generate_comparison_report(
        self,
        output_path: Path,
        top_n: int = 10
    ) -> Path:
        """
        generate comparison report.
        
        Args:
            output_path: Output Markdown file path
            top_n: Show top N issue turns
        
        Returns:
            output file path
        """
        lines = ["# Session Comparison Analysis Report\n\n"]
        lines.append(f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lines.append("---\n\n")
        
        # 1. Session comparison table
        lines.append("## 📋 Session Comparison Table\n\n")
        lines.append("| Session Name | Date | Benchmark Lap | Lap Time | PB Status |\n")
        lines.append("|------------|------|--------|------|--------|\n")
        
        for session in sorted(self.sessions, key=lambda x: x['date']):
            lap_info = f"Lap {session.get('benchmark_lap', 'N/A')}" if session.get('benchmark_lap') else 'N/A'
            laptime_info = f"{session.get('benchmark_laptime', 0):.2f}s" if session.get('benchmark_laptime') else 'N/A'
            
            # PB status
            pb_status = ""
            if session.get('is_historical_pb'):
                pb_status = "📌 Using Historical PB"
            elif session.get('benchmark_laptime'):
                pb_status = "✅ Current PB"
            else:
                pb_status = "-"
            
            lines.append(f"| {session['name']} | {session['date']} | {lap_info} | {laptime_info} | {pb_status} |\n")
        
        lines.append("\n---\n\n")
        
        # 2. Lap time comparison
        laptime_df = self.compare_laptimes()
        if not laptime_df.empty:
            lines.append("## ⏱️ Lap Time Comparison\n\n")
            lines.append("| Session | Lap Time | Relative to Fastest | PB Type |\n")
            lines.append("|---------|------|----------|--------|\n")
            
            fastest = laptime_df['laptime_s'].min()
            for _, row in laptime_df.iterrows():
                diff = row['laptime_s'] - fastest
                diff_str = f"+{diff:.2f}s" if diff > 0 else f"{diff:.2f}s" if diff < 0 else "Fastest"
                
                # Find corresponding session information
                session_info = next((s for s in self.sessions if s['name'] == row['session']), None)
                pb_type = ""
                if session_info:
                    if session_info.get('is_historical_pb'):
                        pb_type = "📌 Historical PB"
                    else:
                        pb_type = "✅ Current PB"
                
                lines.append(f"| {row['session']} | {row['laptime_s']:.2f}s | {diff_str} | {pb_type} |\n")
            
            lines.append("\n---\n\n")
            
            # PB update history
            pb_updates = []
            for session in sorted(self.sessions, key=lambda x: x['date']):
                if session.get('is_historical_pb') and session.get('historical_pb'):
                    pb_updates.append({
                        'session': session['name'],
                        'date': session['date'],
                        'current_lap': session.get('benchmark_lap'),
                        'current_laptime': session.get('benchmark_laptime'),
                        'historical_pb': session['historical_pb']
                    })
            
            if pb_updates:
                lines.append("### 📌 PB Memory Note\n\n")
                lines.append("The following sessions used historical PB as analysis benchmark (because current PB did not exceed historical best):\n\n")
                for update in pb_updates:
                    lines.append(
                        f"- **{update['session']}** ({update['date']}): "
                        f"Current fastest Lap {update['current_lap']}, {update['current_laptime']:.2f}s, "
                        f"but using historical PB Lap {update['historical_pb']['lap']}, {update['historical_pb']['laptime_s']:.2f}s "
                        f"(from {update['historical_pb']['session_name']})\n"
                    )
                lines.append("\n---\n\n")
        
        # 3. Top Issues comparison
        comparison_df = self.compare_top_issues()
        if not comparison_df.empty:
            lines.append("## 🎯 Top Issues Comparison\n\n")
            
            # Group by turn
            turns = sorted(comparison_df['turn'].unique())
            
            for turn in turns[:top_n]:
                turn_data = comparison_df[comparison_df['turn'] == turn]
                
                lines.append(f"### Turn {turn}\n\n")
                lines.append("| Session | Composite Score | Issue Score | Stability Score | Event Count |\n")
                lines.append("|---------|----------|----------|------------|--------|\n")
                
                for _, row in turn_data.iterrows():
                    lines.append(
                        f"| {row['session']} | {row['composite_score']:.2f} | "
                        f"{row['issue_score']:.2f} | {row['stability_score']:.2f} | "
                        f"{int(row['events_count'])} |\n"
                    )
                
                # Compute improvement status
                if len(turn_data) >= 2:
                    scores = turn_data['composite_score'].values
                    best_score = scores.min()
                    worst_score = scores.max()
                    improvement = worst_score - best_score
                    
                    if improvement > 0:
                        lines.append(f"\n**Improvement**: {improvement:.2f} (from {worst_score:.2f} to {best_score:.2f})\n")
                
                lines.append("\n---\n\n")
        
        # 4. Trend analysis
        trend_summary = self.generate_trend_summary()
        
        if trend_summary['laptime_trend'] or trend_summary['top_issues_trend']:
            lines.append("## 📈 Trend Analysis\n\n")
            
            # Lap time trend
            if trend_summary['laptime_trend']:
                lt = trend_summary['laptime_trend']
                lines.append("### ⏱️ Lap Time Trend\n\n")
                lines.append(f"- **First**: {lt['first']:.2f}s\n")
                lines.append(f"- **Last**: {lt['last']:.2f}s\n")
                lines.append(f"- **Change**: {lt['improvement']:+.2f}s ({lt['improvement_pct']:+.1f}%)\n")
                lines.append(f"- **Trend**: {lt['trend']}\n\n")
            
            # Top Issues trend
            if trend_summary['top_issues_trend']:
                lines.append("### 🎯 Top Issues Trend\n\n")
                lines.append("| Turn | First Score | Last Score | Change | Trend |\n")
                lines.append("|------|----------|----------|------|------|\n")
                
                for turn in sorted(trend_summary['top_issues_trend'].keys()):
                    trend_data = trend_summary['top_issues_trend'][turn]
                    lines.append(
                        f"| Turn {turn} | {trend_data['first']:.2f} | "
                        f"{trend_data['last']:.2f} | {trend_data['change']:+.2f} | "
                        f"{trend_data['trend']} |\n"
                    )
                lines.append("\n")
            
            lines.append("> 💡 Tip: Check `trend_*.png` charts for more detailed trend visualization\n\n")
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(''.join(lines), encoding='utf-8')
        logger.info(f"Comparison report generated: {output_path}")
        
        return output_path
    
    def plot_comparison_charts(
        self,
        output_dir: Path
    ) -> List[Path]:
        """
        Generate comparison charts.
        
        Args:
            output_dir: Output directory
        
        Returns:
            List of generated chart file paths
        """
        output_paths = []
        
        # 1. Lap time comparison bar chart
        laptime_df = self.compare_laptimes()
        if not laptime_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_df = laptime_df.sort_values('date')
            
            bars = ax.bar(range(len(sorted_df)), sorted_df['laptime_s'], 
                         color='steelblue', alpha=0.7)
            
            # Annotate fastest lap time
            fastest_idx = sorted_df['laptime_s'].idxmin()
            fastest_val = sorted_df.loc[fastest_idx, 'laptime_s']
            bars[sorted_df.index.get_loc(fastest_idx)].set_color('gold')
            
            ax.set_xlabel('Session')
            ax.set_ylabel('Lap Time (seconds)')
            ax.set_title('Session Lap Time Comparison')
            ax.set_xticks(range(len(sorted_df)))
            ax.set_xticklabels(sorted_df['session'], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (idx, row) in enumerate(sorted_df.iterrows()):
                ax.text(i, row['laptime_s'] + 0.1, f"{row['laptime_s']:.2f}s",
                       ha='center', va='bottom')
            
            plt.tight_layout()
            laptime_path = output_dir / 'comparison_laptimes.png'
            plt.savefig(laptime_path, dpi=150, bbox_inches='tight')
            plt.close()
            output_paths.append(laptime_path)
            logger.info(f"Lap time comparison chart saved: {laptime_path}")
        
        # 2. Top Issues comparison heatmap
        comparison_df = self.compare_top_issues()
        if not comparison_df.empty:
            # Select top 10 turns
            top_turns = comparison_df.groupby('turn')['composite_score'].mean().nlargest(10).index
            
            pivot_data = comparison_df[comparison_df['turn'].isin(top_turns)].pivot(
                index='turn', columns='session', values='composite_score'
            )
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Composite Score'}, ax=ax)
            
            ax.set_title('Top Issues Comparison Heatmap')
            ax.set_xlabel('Session')
            ax.set_ylabel('Turn')
            
            plt.tight_layout()
            heatmap_path = output_dir / 'comparison_heatmap.png'
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            output_paths.append(heatmap_path)
            logger.info(f"Comparison heatmap saved: {heatmap_path}")
        
        return output_paths
    
    def plot_trend_analysis(
        self,
        output_dir: Path,
        top_n_turns: int = 5
    ) -> List[Path]:
        """
        Generate trend analysis visualization charts.
        
        Args:
            output_dir: Output directory
            top_n_turns: Show trends for top N issue turns
        
        Returns:
            List of generated chart file paths
        """
        output_paths = []
        
        # Sort sessions by date
        sorted_sessions = sorted(self.sessions, key=lambda x: x['date'])
        if len(sorted_sessions) < 2:
            logger.warning("Need at least 2 sessions to perform trend analysis")
            return output_paths
        
        # 1. Lap time trend chart (time series)
        laptime_df = self.compare_laptimes()
        if not laptime_df.empty and len(laptime_df) >= 2:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sorted_laptime = laptime_df.sort_values('date')
            dates = pd.to_datetime(sorted_laptime['date'])
            
            # Plot lap time trend line
            ax.plot(dates, sorted_laptime['laptime_s'], 
                   marker='o', linewidth=2, markersize=8, 
                   color='steelblue', label='Lap Time')
            
            # Annotate PB
            fastest_idx = sorted_laptime['laptime_s'].idxmin()
            fastest_date = pd.to_datetime(sorted_laptime.loc[fastest_idx, 'date'])
            fastest_laptime = sorted_laptime.loc[fastest_idx, 'laptime_s']
            ax.scatter([fastest_date], [fastest_laptime], 
                      s=200, color='gold', zorder=5, 
                      label='Best PB', edgecolors='black', linewidths=2)
            
            # Add trend line (linear regression)
            if len(dates) >= 2:
                from scipy import stats
                x_numeric = np.arange(len(dates))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x_numeric, sorted_laptime['laptime_s']
                )
                trend_line = intercept + slope * x_numeric
                ax.plot(dates, trend_line, '--', alpha=0.5, color='red', 
                       label=f'Trend Line (slope: {slope:.3f}s/session)')
            
            # Add improvement annotations
            for i in range(1, len(sorted_laptime)):
                prev_laptime = sorted_laptime.iloc[i-1]['laptime_s']
                curr_laptime = sorted_laptime.iloc[i]['laptime_s']
                improvement = prev_laptime - curr_laptime
                
                if abs(improvement) > 0.1:  # Only annotate significant changes
                    mid_date = (dates.iloc[i-1] + dates.iloc[i]) / 2
                    mid_laptime = (prev_laptime + curr_laptime) / 2
                    color = 'green' if improvement > 0 else 'red'
                    ax.annotate(
                        f"{improvement:+.2f}s",
                        xy=(mid_date, mid_laptime),
                        xytext=(10, 10 if improvement > 0 else -20),
                        textcoords='offset points',
                        fontsize=9,
                        color=color,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                    )
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Lap Time (seconds)', fontsize=12)
            ax.set_title('Lap Time Trend Analysis', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            trend_laptime_path = output_dir / 'trend_laptimes.png'
            plt.savefig(trend_laptime_path, dpi=150, bbox_inches='tight')
            plt.close()
            output_paths.append(trend_laptime_path)
            logger.info(f"Lap time trend chart saved: {trend_laptime_path}")
        
        # 2. Top Issues score trend (by turn)
        comparison_df = self.compare_top_issues()
        if not comparison_df.empty:
            # Select top N issue turns
            top_turns = comparison_df.groupby('turn')['composite_score'].mean().nlargest(top_n_turns).index
            
            fig, axes = plt.subplots(len(top_turns), 1, figsize=(12, 4 * len(top_turns)))
            if len(top_turns) == 1:
                axes = [axes]
            
            for idx, turn in enumerate(top_turns):
                ax = axes[idx]
                turn_data = comparison_df[comparison_df['turn'] == turn].sort_values('date')
                
                if len(turn_data) >= 2:
                    dates = pd.to_datetime(turn_data['date'])
                    
                    # Plot composite score trend
                    ax.plot(dates, turn_data['composite_score'], 
                           marker='o', linewidth=2, markersize=8,
                           color='coral', label='Composite Score')
                    
                    # Plot issue score trend
                    if 'issue_score' in turn_data.columns:
                        ax.plot(dates, turn_data['issue_score'], 
                               marker='s', linewidth=1.5, markersize=6,
                               color='steelblue', alpha=0.7, label='Issue Score')
                    
                    # Plot stability score trend
                    if 'stability_score' in turn_data.columns:
                        ax.plot(dates, turn_data['stability_score'], 
                               marker='^', linewidth=1.5, markersize=6,
                               color='green', alpha=0.7, label='Stability Score')
                    
                    # Add improvement annotations
                    for i in range(1, len(turn_data)):
                        prev_score = turn_data.iloc[i-1]['composite_score']
                        curr_score = turn_data.iloc[i]['composite_score']
                        change = curr_score - prev_score
                        
                        if abs(change) > 1.0:  # Only annotate significant changes
                            mid_date = (dates.iloc[i-1] + dates.iloc[i]) / 2
                            mid_score = (prev_score + curr_score) / 2
                            color = 'green' if change < 0 else 'red'  # Score decrease is improvement
                            ax.annotate(
                                f"{change:+.1f}",
                                xy=(mid_date, mid_score),
                                xytext=(10, 10 if change < 0 else -20),
                                textcoords='offset points',
                                fontsize=8,
                                color=color,
                                fontweight='bold'
                            )
                    
                    ax.set_title(f'Turn {turn} Score Trend', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Score', fontsize=10)
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    if idx == len(top_turns) - 1:
                        ax.set_xlabel('Date', fontsize=10)
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            trend_issues_path = output_dir / 'trend_top_issues.png'
            plt.savefig(trend_issues_path, dpi=150, bbox_inches='tight')
            plt.close()
            output_paths.append(trend_issues_path)
            logger.info(f"Top Issues trend chart saved: {trend_issues_path}")
        
        # 3. Composite score change radar chart (compare first and last session)
        if len(sorted_sessions) >= 2 and not comparison_df.empty:
            first_session = sorted_sessions[0]
            last_session = sorted_sessions[-1]
            
            first_data = comparison_df[comparison_df['session'] == first_session['name']]
            last_data = comparison_df[comparison_df['session'] == last_session['name']]
            
            if not first_data.empty and not last_data.empty:
                # Select common turns
                common_turns = set(first_data['turn'].unique()) & set(last_data['turn'].unique())
                if len(common_turns) >= 3:
                    common_turns = sorted(list(common_turns))[:8]  # Maximum 8 turns
                    
                    first_scores = [first_data[first_data['turn'] == t]['composite_score'].iloc[0] 
                                  if len(first_data[first_data['turn'] == t]) > 0 else 0 
                                  for t in common_turns]
                    last_scores = [last_data[last_data['turn'] == t]['composite_score'].iloc[0] 
                                 if len(last_data[last_data['turn'] == t]) > 0 else 0 
                                 for t in common_turns]
                    
                    # Create radar chart
                    from math import pi
                    
                    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
                    
                    angles = [n / float(len(common_turns)) * 2 * pi for n in range(len(common_turns))]
                    angles += angles[:1]  # Close the loop
                    
                    first_scores += first_scores[:1]
                    last_scores += last_scores[:1]
                    
                    ax.plot(angles, first_scores, 'o-', linewidth=2, label=first_session['name'], color='blue')
                    ax.fill(angles, first_scores, alpha=0.25, color='blue')
                    
                    ax.plot(angles, last_scores, 'o-', linewidth=2, label=last_session['name'], color='red')
                    ax.fill(angles, last_scores, alpha=0.25, color='red')
                    
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels([f'T{int(t)}' for t in common_turns])
                    ax.set_ylim(0, max(max(first_scores), max(last_scores)) * 1.2)
                    ax.set_title('Composite Score Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                    ax.grid(True)
                    
                    plt.tight_layout()
                    radar_path = output_dir / 'trend_radar.png'
                    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    output_paths.append(radar_path)
                    logger.info(f"Radar chart saved: {radar_path}")
        
        # 4. Issue improvement/worsening count
        if not comparison_df.empty and len(sorted_sessions) >= 2:
            # Compute improvement status for each turn
            improvement_data = []
            
            for turn in comparison_df['turn'].unique():
                turn_data = comparison_df[comparison_df['turn'] == turn].sort_values('date')
                if len(turn_data) >= 2:
                    first_score = turn_data.iloc[0]['composite_score']
                    last_score = turn_data.iloc[-1]['composite_score']
                    change = first_score - last_score  # Positive value indicates improvement
                    improvement_pct = (change / first_score * 100) if first_score > 0 else 0
                    
                    improvement_data.append({
                        'turn': turn,
                        'first_score': first_score,
                        'last_score': last_score,
                        'change': change,
                        'improvement_pct': improvement_pct
                    })
            
            if improvement_data:
                improvement_df = pd.DataFrame(improvement_data)
                improvement_df = improvement_df.sort_values('change', ascending=False)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Left chart: improvement magnitude bar chart
                colors = ['green' if x > 0 else 'red' for x in improvement_df['change']]
                bars = ax1.barh(range(len(improvement_df)), improvement_df['change'], color=colors, alpha=0.7)
                ax1.set_yticks(range(len(improvement_df)))
                ax1.set_yticklabels([f'T{int(t)}' for t in improvement_df['turn']])
                ax1.set_xlabel('Score Change (decrease = improvement)', fontsize=11)
                ax1.set_title('Improvement/Worsening Status by Turn', fontsize=12, fontweight='bold')
                ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
                ax1.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (idx, row) in enumerate(improvement_df.iterrows()):
                    ax1.text(row['change'], i, f"{row['change']:+.1f}",
                           va='center', ha='left' if row['change'] > 0 else 'right',
                           fontsize=9, fontweight='bold')
                
                # Right chart: improvement percentage
                colors_pct = ['green' if x > 0 else 'red' for x in improvement_df['improvement_pct']]
                bars2 = ax2.barh(range(len(improvement_df)), improvement_df['improvement_pct'], 
                               color=colors_pct, alpha=0.7)
                ax2.set_yticks(range(len(improvement_df)))
                ax2.set_yticklabels([f'T{int(t)}' for t in improvement_df['turn']])
                ax2.set_xlabel('Improvement Percentage (%)', fontsize=11)
                ax2.set_title('Improvement Percentage by Turn', fontsize=12, fontweight='bold')
                ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
                ax2.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, (idx, row) in enumerate(improvement_df.iterrows()):
                    ax2.text(row['improvement_pct'], i, f"{row['improvement_pct']:+.1f}%",
                           va='center', ha='left' if row['improvement_pct'] > 0 else 'right',
                           fontsize=9, fontweight='bold')
                
                plt.tight_layout()
                improvement_path = output_dir / 'trend_improvement.png'
                plt.savefig(improvement_path, dpi=150, bbox_inches='tight')
                plt.close()
                output_paths.append(improvement_path)
                logger.info(f"Improvement statistics chart saved: {improvement_path}")
        
        return output_paths

