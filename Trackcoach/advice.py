"""Coaching advice generation module: generate personalized advice reports based on issue analysis results."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import re

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Issue type mapping: detected event types → knowledge base issue names
ISSUE_TYPE_MAPPING = {
    'conservative_entry': 'Conservative Entry / Overslowing',
    'late_brake': 'Late Braking / Missed Apex',
    'weak_exit': 'Weak Corner Exit',
    'overslow': 'Conservative Entry / Overslowing',
    'heavy_brake': 'Late Braking / Missed Apex',  # Heavy braking may cause missed apex
    'early_brake': 'Conservative Entry / Overslowing'
}


def fix_encoding_issues(text: str) -> str:
    """
    Fix encoding issues in text and remove Chinese translations, keeping only English.
    
    Common issue: UTF-8 encoded Chinese characters being interpreted as Latin-1 or other encodings.
    Also removes Chinese translations that appear in parentheses after English labels.
    
    Args:
        text: Text with potential encoding issues and Chinese translations
        
    Returns:
        Fixed text with only English content
    """
    if not text:
        return text
    
    # First, try the general fix: re-encode as Latin-1 and decode as UTF-8
    # This handles the case where UTF-8 bytes were incorrectly interpreted as Latin-1
    try:
        # Check if text contains characters that suggest UTF-8 misinterpretation
        # Common indicators: ç, å, æ, é, etc. (which are Latin-1 representations of UTF-8 bytes)
        if any(ord(char) > 127 for char in text):
            # Try to fix by re-encoding as Latin-1 and decoding as UTF-8
            # This works when UTF-8 bytes were read as Latin-1
            try:
                fixed_bytes = text.encode('latin-1')
                fixed_text = fixed_bytes.decode('utf-8')
                # If successful, use the fixed text
                text = fixed_text
            except (UnicodeEncodeError, UnicodeDecodeError):
                # If that doesn't work, try character-by-character fix
                pass
    except Exception:
        pass
    
    # Apply specific pattern fixes for common Chinese phrases (fix encoding first)
    encoding_fixes = {
        # Common Chinese words that appear in the knowledge base (fix encoding issues)
        'ç—‡çŠ¶': '症状',  # symptom
        'å¯èƒ½åŸå› ': '可能原因',  # likely causes (with trailing space)
        'å¯èƒ½åŸå›': '可能原因',  # likely causes (without trailing space)
        'å¤„ç†æ–¹æ³•': '处理方法',  # handling approach
        'å…³é"®è€ƒè™\'': '关键考虑',  # key considerations - note: contains quote in key
        'ç±»åˆ«': '类别',  # category
        'æ•°æ®å®¡æŸ¥å·¥ä½œæµ': '数据审查工作流',  # data review workflow
    }
    
    # Apply encoding fixes
    for wrong, correct in encoding_fixes.items():
        text = text.replace(wrong, correct)
    
    # Remove Chinese translations in parentheses after English labels
    # Pattern: "English (Chinese)" -> "English"
    
    # Remove Chinese translations in parentheses after common English labels
    chinese_patterns = [
        (r'\*\*Symptom\s*\([^)]*\)\*\*:', r'**Symptom:**'),
        (r'\*\*Likely Causes\s*\([^)]*\)\*\*:', r'**Likely Causes:**'),
        (r'\*\*Handling Approach\s*\([^)]*\)\*\*:', r'**Handling Approach:**'),
        (r'\*\*Key Considerations\s*\([^)]*\)\*\*:', r'**Key Considerations:**'),
        (r'\*\*Category\s*\([^)]*\)\*\*:', r'**Category:**'),
        (r'##\s*📊\s*Data Review Workflow\s*\([^)]*\)', r'## 📊 Data Review Workflow'),
        # General pattern: remove any non-ASCII characters in parentheses (Chinese translations)
        (r'\([^\x00-\x7F]+\)', ''),  # Remove any non-ASCII characters in parentheses
    ]
    
    for pattern, replacement in chinese_patterns:
        text = re.sub(pattern, replacement, text)
    
    # Clean up any remaining Chinese characters in parentheses (more general pattern)
    text = re.sub(r'\s*\([^\x00-\x7F]+\)\s*', '', text)  # Remove parentheses with non-ASCII
    
    # Clean up extra spaces after colons in bold text
    text = re.sub(r':\s+:', ':', text)  # Fix double colons
    text = re.sub(r'\*\*([^*]+?)\s+:\*\*', r'**\1:**', text)  # Fix space before colon in bold (non-greedy match)
    
    # Fix common encoding issues for special characters
    # Fix em dash (—) that was encoded as â€"
    text = text.replace('â€"', '—')
    text = text.replace('â€"', '—')  # Alternative encoding
    text = text.replace('â€"', '—')  # Another variant
    
    # Fix emoji encoding issues (apply again after other fixes)
    # 📊 emoji (UTF-8: F0 9F 93 8A) sometimes appears as ğŸ"Š when UTF-8 is read as Latin-1
    # The pattern ğŸ"Š is the Latin-1 interpretation of the UTF-8 bytes
    emoji_fixes = {
        'ğŸ"Š': '📊',  # 📊 emoji - most common pattern
        'ğŸ"': '📊',   # Alternative pattern
    }
    for wrong, correct in emoji_fixes.items():
        text = text.replace(wrong, correct)
    
    # Fix emoji in section headers specifically (use raw string to avoid issues)
    # Pattern: ## ğŸ"Š Data Review Workflow -> ## 📊 Data Review Workflow
    text = re.sub(r'##\s*ğŸ"Š\s*Data Review Workflow', '## 📊 Data Review Workflow', text, flags=re.IGNORECASE)
    text = re.sub(r'##\s*ğŸ"\s*Data Review Workflow', '## 📊 Data Review Workflow', text, flags=re.IGNORECASE)
    
    # Fix other common UTF-8 misinterpretations
    # En dash (–) and other special characters
    text = text.replace('â€"', '–')
    text = text.replace('â€™', "'")  # Right single quotation mark
    text = text.replace('â€œ', '"')  # Left double quotation mark
    text = text.replace('â€', '"')   # Right double quotation mark
    
    # Try to fix any remaining UTF-8 misinterpretations
    # This is a more aggressive fix for any remaining issues
    try:
        # If we see patterns like â€, try to fix them
        if 'â€' in text or 'ğŸ' in text:
            # Try to re-encode problematic sequences
            # This is a heuristic approach
            pass  # Already handled above with specific replacements
    except Exception:
        pass
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Remove excessive blank lines
    text = re.sub(r' +', ' ', text)  # Remove multiple spaces
    
    return text.strip()


class CoachingAdvisor:
    """Coaching advice generator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize advice generator.
        
        Args:
            config_path: LlamaCloud config file path, if None then knowledge base will not be initialized
        """
        self.index = None
        self.retriever = None
        self.llm = None
        self.config_path = config_path
        self.config = None
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self._init_knowledge_base(config_path)
                self._init_llm()
            except Exception as e:
                logger.warning(f"Unable to initialize knowledge base or LLM: {e}, will skip advanced features")
        else:
            logger.info("Config file not provided, will skip knowledge base query functionality")
    
    def _init_knowledge_base(self, config_path: str) -> None:
        """Initialize LlamaCloud knowledge base."""
        try:
            from llama_cloud_services import LlamaCloudIndex
        except ImportError:
            raise ImportError("Please install llama-cloud-services: pip install llama-cloud-services")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.index = LlamaCloudIndex(
            name=config['name'],
            project_name=config['project_name'],
            organization_id=config['organization_id'],
            api_key=config['api_key']
        )
        self.retriever = self.index.as_retriever()
        logger.info("Knowledge base initialized successfully")
    
    def _init_llm(self) -> None:
        """Initialize LLM for generating personalized advice."""
        if not self.config or 'llm' not in self.config:
            logger.info("LLM config not found, will skip LLM-based personalized advice generation")
            return
        
        try:
            llm_config = self.config['llm']
            provider = llm_config.get('provider', '').lower()
            
            if provider == 'openrouter':
                try:
                    from llama_index.llms import OpenRouter
                except ImportError:
                    try:
                        from llama_index.llms.openrouter import OpenRouter
                    except ImportError:
                        from llama_index.core.llms import OpenRouter
                
                self.llm = OpenRouter(
                    api_key=llm_config.get('api_key', ''),
                    model=llm_config.get('model', 'deepseek/deepseek-v3.2'),
                    temperature=llm_config.get('temperature', 0.7),
                    max_tokens=llm_config.get('max_tokens', 2048),
                    context_window=llm_config.get('context_window', 4096)
                )
                logger.info(f"LLM initialized: {llm_config.get('model', 'unknown')}")
            else:
                logger.warning(f"LLM provider '{provider}' not supported, skipping LLM initialization")
        except ImportError as e:
            logger.warning(f"Failed to import LLM libraries: {e}, will skip LLM-based advice generation")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}, will skip LLM-based advice generation")
    
    def map_event_to_issue(self, event_type: str) -> Optional[str]:
        """Map event type to knowledge base issue name."""
        return ISSUE_TYPE_MAPPING.get(event_type)
    
    def query_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query knowledge base.
        
        Args:
            query: Query string
            top_k: Number of results to return
        
        Returns:
            List of query results
        """
        if self.retriever is None:
            logger.warning("Knowledge base not initialized, cannot query")
            return []
        
        try:
            nodes = self.retriever.retrieve(query)
            results = []
            for node in nodes[:top_k]:
                text = node.text if hasattr(node, 'text') else str(node)
                # Fix encoding issues immediately after retrieval
                text = fix_encoding_issues(text)
                results.append({
                    'text': text,
                    'score': getattr(node, 'score', None),
                    'metadata': getattr(node, 'metadata', {})
                })
            return results
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return []
    
    def generate_personalized_advice(
        self,
        issue_name: str,
        turn: int,
        score: float,
        events_count: int,
        stability_info: Optional[Dict[str, Any]] = None,
        knowledge_base_text: str = ""
    ) -> Optional[str]:
        """
        Generate personalized advice using LLM based on specific driving data.
        
        Args:
            issue_name: Issue name (e.g., "Conservative Entry / Overslowing")
            turn: Turn number
            score: Issue score
            events_count: Number of events detected
            stability_info: Stability assessment information
            knowledge_base_text: Retrieved knowledge base content
        
        Returns:
            Personalized advice text, or None if LLM is not available
        """
        if self.llm is None:
            return None
        
        try:
            # Build context from stability info
            stability_context = ""
            if stability_info and stability_info.get('available'):
                stability_context = f"""
Stability Assessment: {stability_info.get('severity', 'Unknown')}
Stability Ranking: {stability_info.get('rank_text', 'N/A')}
"""
                if stability_info.get('details'):
                    metrics = []
                    for detail in stability_info.get('details', [])[:3]:
                        metrics.append(f"{detail['name']}: {detail['value']:.2f} {detail['unit']}")
                    stability_context += f"Key Metrics: {', '.join(metrics)}\n"
            
            # Build prompt
            prompt = f"""You are an expert track driving coach. Based on the following information, provide personalized, actionable advice for improving performance at Turn {turn}.

Issue: {issue_name}
Turn Number: {turn}
Issue Score: {score:.2f} (higher = more critical)
Number of Events Detected: {events_count}
{stability_context}

Reference Knowledge Base Content:
{knowledge_base_text[:1000]}

WRITING STYLE REQUIREMENTS (Professional-Friendly Tone):
1. Always use second person "you" (not "the driver" or third person)
2. Maintain PROFESSIONAL but FRIENDLY tone:
   - Professional: Base advice on data analysis, use technical terms appropriately, be precise
   - Friendly: Use supportive language, show confidence in driver's potential, be encouraging but not overly enthusiastic
   - Avoid: Excessive rhetorical questions, overly casual expressions, exaggerated metaphors, excessive exclamation marks
3. Use VARIED but NATURAL sentence structures:
   - Mix declarative and imperative sentences
   - Vary sentence length for readability
   - Use natural transitions
   - Avoid repetitive patterns
   - CRITICAL: Vary your recommendation openings - avoid starting multiple recommendations with the same verb (e.g., don't always start with "Brake", "Eliminate", "Trust")
   - Use different sentence structures: questions, conditionals, comparisons, and varied imperative forms
   - Vary your closing statements - avoid repeating the same phrase like "Addressing this issue will yield..."
4. Structure (consistent format):
   - Start with "Analysis:" - 2-3 sentences describing the issue at this specific turn, reference the data
   - Then "Actionable Recommendations:" - 3 numbered items with clear, specific advice
   - End with a brief, encouraging closing (1 sentence, supportive but not overly enthusiastic)
5. Keep total response between 150-250 words
6. Be specific to Turn {turn} and reference the data provided
7. Sound like a professional coach: knowledgeable, supportive, and direct

Example of professional-friendly style with VARIED grammar:
Analysis: Turn {turn} shows a pattern of [specific issue based on data]. The data indicates [technical observation]. This is affecting [specific impact on performance].

Actionable Recommendations:
1. [Vary opening - use different verbs: "Push", "Extend", "Adjust", "Refine", "Optimize", etc.]
2. [Use different structure - maybe start with "Focus on", "Work on", "Consider", "Try", etc.]
3. [Vary again - "Build confidence in", "Rely on", "Have faith in", "Let the car show you", etc.]

[Vary closing - e.g., "These adjustments will unlock time gains at Turn {turn}." OR "Master these techniques and Turn {turn} will become a strength." OR "Implement these changes and watch your sector times improve."]

IMPORTANT: Each recommendation should use DIFFERENT opening verbs and structures. Avoid repeating the same pattern across multiple turns."""

            # Try different response formats
            response = self.llm.complete(prompt)
            if hasattr(response, 'text'):
                advice_text = response.text.strip()
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                advice_text = response.message.content.strip()
            elif isinstance(response, str):
                advice_text = response.strip()
            else:
                advice_text = str(response).strip()
            
            # Post-process to ensure consistency
            advice_text = self._normalize_advice_format(advice_text)
            return advice_text
        except Exception as e:
            logger.warning(f"Failed to generate personalized advice with LLM: {e}")
            return None
    
    def _normalize_advice_format(self, text: str) -> str:
        """
        Normalize the format and pronouns in AI-generated advice.
        Preserves natural language variety while ensuring consistency.
        
        Args:
            text: Raw AI-generated advice text
        
        Returns:
            Normalized advice text with consistent format and pronouns
        """
        if not text:
            return text
        
        # Replace third person references with second person (but preserve natural variety)
        text = re.sub(r'\bthe driver\b', 'you', text, flags=re.IGNORECASE)
        text = re.sub(r'\bthe driver\'s\b', 'your', text, flags=re.IGNORECASE)
        # Only replace standalone "driver" if it's clearly referring to the person
        text = re.sub(r'\bdriver\b(?=\s+(?:is|are|has|have|was|were|will|can|should))', 'you', text, flags=re.IGNORECASE)
        
        # Normalize section headers
        text = re.sub(r'^(\d+\.\s*)?Analysis\s*:', 'Analysis:', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^(\d+\.\s*)?Actionable\s+Recommendations?\s*:', 'Actionable Recommendations:', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^(\d+\.\s*)?Recommendations?\s*:', 'Actionable Recommendations:', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Normalize numbered lists (convert lettered sub-items to numbered)
        # Convert "a. b. c." to "1. 2. 3." if they appear after "Recommendations:"
        lines = text.split('\n')
        normalized_lines = []
        in_recommendations = False
        item_number = 1
        
        for line in lines:
            # Check if we're entering recommendations section
            if re.match(r'^Actionable\s+Recommendations?\s*:', line, re.IGNORECASE):
                in_recommendations = True
                item_number = 1
                normalized_lines.append(line)
                continue
            
            # Check if we're leaving recommendations (new section or end)
            if in_recommendations and (re.match(r'^[A-Z]', line.strip()) or line.strip().startswith('Implementing') or line.strip().startswith('Remember')):
                in_recommendations = False
                item_number = 1
            
            # Normalize list items in recommendations section
            if in_recommendations:
                # Match lettered items (a. b. c.) or numbered items
                lettered_match = re.match(r'^\s*([a-z])\.\s+(.+)', line, re.IGNORECASE)
                numbered_match = re.match(r'^\s*(\d+)\.\s+(.+)', line)
                
                if lettered_match:
                    # Convert lettered to numbered
                    content = lettered_match.group(2)
                    normalized_lines.append(f"{item_number}. {content}")
                    item_number += 1
                elif numbered_match:
                    # Keep numbered but ensure sequential
                    content = numbered_match.group(2)
                    normalized_lines.append(f"{item_number}. {content}")
                    item_number += 1
                else:
                    normalized_lines.append(line)
            else:
                normalized_lines.append(line)
        
        text = '\n'.join(normalized_lines)
        
        # Remove "Implementation:" or "3. Implementation:" sections if they exist
        # (we want to keep it simple with just Analysis and Recommendations)
        text = re.sub(r'\n\d+\.\s*Implementation\s*:.*?(?=\n\n|\n---|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\nImplementation\s*:.*?(?=\n\n|\n---|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra blank lines (but preserve intentional paragraph breaks)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'\n{3}(?=\n)', '\n\n', text)
        
        # Preserve natural variety - don't over-normalize
        # Keep varied sentence structures, questions, exclamations, etc.
        
        return text.strip()
    
    def get_advice_for_issue(self, issue_name: str) -> Dict[str, Any]:
        """
        Get advice for a specific issue.
        
        Args:
            issue_name: Issue name
        
        Returns:
            Dictionary containing advice information
        """
        # Query issue-related information
        query = issue_name
        results = self.query_knowledge_base(query, top_k=5)
        
        # Try to find more specific information
        if not results or results[0].get('score', 0) < 0.3:
            # If direct query doesn't work well, try more specific query
            query = f"{issue_name} symptom causes advice"
            results = self.query_knowledge_base(query, top_k=5)
        
        return {
            'issue_name': issue_name,
            'knowledge_base_results': results
        }
    
    def analyze_top_issues(
        self,
        top_issues_df: pd.DataFrame,
        events_df: pd.DataFrame,
        stability_report_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analyze top issues and generate advice.
        
        Args:
            top_issues_df: Top Issues DataFrame
            events_df: Events DataFrame
        
        Returns:
            Analysis result dictionary
        """
        # Count issue types
        issue_counts = defaultdict(int)
        turn_issues = defaultdict(list)
        
        # Analyze top issues
        for _, row in top_issues_df.iterrows():
            turn = int(row['turn'])
            top_types = str(row['top_types']).split(', ')
            
            for event_type in top_types:
                event_type = event_type.strip()
                if event_type and event_type != '-':
                    issue_counts[event_type] += 1
                    turn_issues[turn].append({
                        'type': event_type,
                        'score': row['score'],
                        'events_count': row['events_count']
                    })
        
        # Get advice for each issue
        advice_by_issue = {}
        unique_issues = set()
        
        for event_type, count in issue_counts.items():
            issue_name = self.map_event_to_issue(event_type)
            if issue_name and issue_name not in unique_issues:
                unique_issues.add(issue_name)
                advice_by_issue[issue_name] = self.get_advice_for_issue(issue_name)
        
        return {
            'issue_counts': dict(issue_counts),
            'turn_issues': dict(turn_issues),
            'advice_by_issue': advice_by_issue,
            'top_issues_df': top_issues_df
        }
    
    def load_stability_data(
        self,
        stability_report_path: Optional[Path] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[int, Dict[str, Any]]]]:
        """
        Load stability report data and compute stability scores.
        
        Args:
            stability_report_path: Stability report CSV file path, if None then try to find from output directory
        
        Returns:
            (Stability report DataFrame, Dictionary of stability data indexed by turn)
        """
        if stability_report_path is None:
            # Try to find from default location (assume same directory as report output)
            return None, None
        
        if not stability_report_path.exists():
            logger.warning(f"Stability report file does not exist: {stability_report_path}")
            return None, None
        
        try:
            stability_df = pd.read_csv(stability_report_path)
            
            # Compute composite stability score (higher value = less stable)
            # Use IQR of key metrics as stability indicators
            stability_scores = {}
            for _, row in stability_df.iterrows():
                turn = int(row['turn'])
                scores = []
                
                # Braking distance stability (most important)
                if pd.notna(row.get('d_delta_s_iqr', np.nan)):
                    scores.append(('Braking Distance IQR', float(row['d_delta_s_iqr'])))
                
                # Apex speed stability
                if pd.notna(row.get('d_Vmin_iqr', np.nan)):
                    scores.append(('Apex Speed IQR', float(row['d_Vmin_iqr'])))
                
                # Entry speed stability
                if pd.notna(row.get('d_Ventry_iqr', np.nan)):
                    scores.append(('Entry Speed IQR', float(row['d_Ventry_iqr'])))
                
                # Longitudinal acceleration stability
                if pd.notna(row.get('d_a_long_peak_iqr', np.nan)):
                    scores.append(('Longitudinal Acceleration IQR', float(row['d_a_long_peak_iqr'])))
                
                stability_scores[turn] = {
                    'scores': scores,
                    'row': row.to_dict()
                }
            
            # Compute stability ranking (by composite score)
            if stability_scores:
                # Compute composite score: weighted IQR values
                composite_scores = {}
                for turn, data in stability_scores.items():
                    row = data['row']
                    score = 0.0
                    weights = {
                        'd_delta_s_iqr': 0.5,
                        'd_Vmin_iqr': 0.3,
                        'd_Ventry_iqr': 0.1,
                        'd_a_long_peak_iqr': 0.1
                    }
                    for metric, weight in weights.items():
                        if pd.notna(row.get(metric, np.nan)):
                            score += weight * float(row[metric])
                    composite_scores[turn] = score
                
                # Normalize and compute ranking
                if composite_scores:
                    max_score = max(composite_scores.values())
                    if max_score > 0:
                        for turn in composite_scores:
                            normalized_score = composite_scores[turn] / max_score
                            stability_scores[turn]['composite_score'] = normalized_score
                        
                        # Compute ranking (higher score = higher rank = less stable)
                        sorted_turns = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
                        for rank, (turn, _) in enumerate(sorted_turns, 1):
                            stability_scores[turn]['rank'] = rank
                            stability_scores[turn]['total_turns'] = len(sorted_turns)
            
            return stability_df, stability_scores
        except Exception as e:
            logger.warning(f"Failed to load stability report: {e}")
            return None, None
    
    def get_stability_assessment(
        self,
        stability_data: Optional[Dict[str, Any]],
        turn: int
    ) -> Dict[str, Any]:
        """
        Get stability assessment for a specific turn.
        
        Args:
            stability_data: Stability data dictionary
            turn: Turn number
        
        Returns:
            Dictionary containing stability assessment information
        """
        if stability_data is None or turn not in stability_data:
            return {
                'available': False,
                'summary': None,
                'details': None,
                'rank_text': None,
                'severity': None
            }
        
        data = stability_data[turn]
        row = data.get('row', {})
        
        # Extract key metrics
        details = []
        key_metrics = [
            ('d_delta_s_iqr', 'Braking Distance IQR', 'm'),
            ('d_Vmin_iqr', 'Apex Speed IQR', 'm/s'),
            ('d_Ventry_iqr', 'Entry Speed IQR', 'm/s'),
            ('d_a_long_peak_iqr', 'Longitudinal Acceleration IQR', 'g')
        ]
        
        for metric_key, metric_name, unit in key_metrics:
            if pd.notna(row.get(metric_key, np.nan)):
                value = float(row[metric_key])
                details.append({
                    'name': metric_name,
                    'value': value,
                    'unit': unit
                })
        
        # Composite score and ranking
        composite_score = data.get('composite_score', 0.0)
        rank = data.get('rank', None)
        total_turns = data.get('total_turns', None)
        
        # Assess severity
        if composite_score >= 0.8:
            severity = 'Very Poor'
            severity_emoji = '🔴'
        elif composite_score >= 0.6:
            severity = 'Poor'
            severity_emoji = '🟠'
        elif composite_score >= 0.4:
            severity = 'Average'
            severity_emoji = '🟡'
        else:
            severity = 'Good'
            severity_emoji = '🟢'
        
        # Generate summary text
        if details:
            primary_metric = details[0]  # Braking distance IQR is most important
            summary = f"{primary_metric['name']}={primary_metric['value']:.2f}{primary_metric['unit']}"
        else:
            summary = "No data"
        
        # Generate ranking text
        if rank and total_turns:
            rank_text = f"Stability ranking: {rank}/{total_turns} (higher rank = less stable)"
        else:
            rank_text = None
        
        return {
            'available': True,
            'summary': summary,
            'details': details,
            'rank_text': rank_text,
            'severity': severity,
            'severity_emoji': severity_emoji,
            'composite_score': composite_score
        }
    
    def generate_report(
        self,
        analysis: Dict[str, Any],
        output_path: Path,
        stability_report_path: Optional[Path] = None
    ) -> None:
        """
        Generate coaching advice report.
        
        Args:
            analysis: Analysis result dictionary
            output_path: Output file path
            stability_report_path: Stability report CSV file path (optional)
        """
        # Try to load stability data
        _, stability_data = self.load_stability_data(stability_report_path)
        
        lines = ["# Intelligent Coaching Advice Report\n"]
        lines.append("This report is generated based on driving data analysis and knowledge base recommendations.\n\n")
        lines.append("---\n\n")
        
        # 1. Issue count
        lines.append("## Issue Count\n\n")
        issue_counts = analysis['issue_counts']
        if issue_counts:
            lines.append("| Issue Type | Count |\n")
            lines.append("|---------|---------|\n")
            for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {issue_type} | {count} |\n")
        else:
            lines.append("No issues detected.\n")
        lines.append("\n")
        
        # 2. Main issue analysis
        lines.append("## Main Issue Analysis\n\n")
        top_issues_df = analysis['top_issues_df']
        
        for _, row in top_issues_df.head(5).iterrows():
            turn = int(row['turn'])
            score = row['score']
            top_types = str(row['top_types'])
            events_count = row['events_count']
            
            # Getstability assessment
            stability = self.get_stability_assessment(stability_data, turn)
            
            # Getcomposite score（if exists）
            composite_score = row.get('composite_score', None)
            
            # Generate title (including composite score and stability assessment)
            if pd.notna(composite_score):
                composite_score = float(composite_score)
                if stability['available']:
                    stability_label = stability['severity']  # Remove emoji
                    lines.append(f"### Turn {turn} (composite score: {composite_score:.2f}, issue score: {score:.2f}, stability: {stability_label})\n\n")
                else:
                    lines.append(f"### Turn {turn} (composite score: {composite_score:.2f}, issue score: {score:.2f})\n\n")
            else:
                if stability['available']:
                    stability_label = stability['severity']  # Remove emoji
                    lines.append(f"### Turn {turn} (issue score: {score:.2f}, stability: {stability_label})\n\n")
                else:
                    lines.append(f"### Turn {turn} (issue score: {score:.2f})\n\n")
            
            lines.append(f"- **Event Count**: {events_count}\n")
            lines.append(f"- **Main Issues**: {top_types}\n\n")
            
            # Add stability metrics
            if stability['available']:
                lines.append("#### Stability Metrics\n\n")
                if stability['details']:
                    lines.append("| Metric | Value |\n")
                    lines.append("|------|------|\n")
                    for detail in stability['details'][:3]:  # Only show top 3 key metrics
                        lines.append(f"| {detail['name']} | {detail['value']:.2f} {detail['unit']} |\n")
                    lines.append("\n")
                
                if stability['rank_text']:
                    lines.append(f"- {stability['rank_text']}\n")
                
                lines.append(f"- **Stability Assessment**: {stability['severity']}\n\n")
                
                # Get composite score (if exists)
                composite_score = row.get('composite_score', None)
                if pd.notna(composite_score):
                    composite_score = float(composite_score)
                    lines.append(f"- **Composite Score**: {composite_score:.2f} (issue score: {score:.2f} + stability score)\n")
                
                # Give priority advice according to composite score or issue score
                # Prioritize composite score, if not available then use issue score
                priority_score = composite_score if pd.notna(composite_score) else score
                if priority_score >= 25:
                    priority = "Very High Priority (many issues and poor consistency, recommend prioritizing improvement)"
                elif priority_score >= 20:
                    priority = "High Priority (needs focused attention)"
                elif priority_score >= 15:
                    priority = "Medium Priority"
                else:
                    priority = "Lower Priority"
                
                lines.append(f"- **Overall Priority**: {priority}\n\n")
            
            # Generate personalized advice using LLM (skip knowledge base generic content)
            issue_types = [t.strip() for t in top_types.split(',') if t.strip() and t.strip() != '-']
            
            # Collect all advice first
            advice_items = []
            for issue_type in issue_types[:2]:  # Only process top 2 main issues
                issue_name = self.map_event_to_issue(issue_type)
                if issue_name and issue_name in analysis['advice_by_issue']:
                    advice = analysis['advice_by_issue'][issue_name]
                    results = advice.get('knowledge_base_results', [])
                    
                    # Get knowledge base text for LLM context (but don't display it)
                    knowledge_base_text = ""
                    if results:
                        best_result = results[0]
                        text = best_result.get('text', '')
                        knowledge_base_text = fix_encoding_issues(text)
                    
                    # Generate personalized advice using LLM
                    if self.llm is not None:
                        try:
                            personalized_advice = self.generate_personalized_advice(
                                issue_name=issue_name,
                                turn=turn,
                                score=score,
                                events_count=events_count,
                                stability_info=stability,
                                knowledge_base_text=knowledge_base_text
                            )
                            if personalized_advice:
                                advice_items.append({
                                    'issue_name': issue_name,
                                    'advice': personalized_advice
                                })
                                logger.info(f"Generated personalized LLM advice for Turn {turn}, Issue: {issue_name}")
                        except Exception as e:
                            logger.warning(f"Failed to generate personalized advice: {e}")
            
            # Write all advice with unified format
            if advice_items:
                lines.append("**AI-Personalized Coaching Advice**\n\n")
                for i, item in enumerate(advice_items):
                    if len(advice_items) > 1:
                        # Use subheading for multiple issues
                        lines.append(f"**{item['issue_name']}:**\n\n")
                    lines.append(f"{item['advice']}\n\n")
                    # Add spacing between multiple issues
                    if i < len(advice_items) - 1:
                        lines.append("\n")
            
            lines.append("---\n\n")
        
        # Note: Knowledge base summary section removed to avoid duplication with AI-generated advice
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Final cleanup: fix any remaining encoding issues in the entire report
        report_text = ''.join(lines)
        # Apply final encoding fixes to the entire report
        report_text = fix_encoding_issues(report_text)
        
        # Additional final cleanup for emoji and special characters
        # Fix emoji patterns that might have been missed
        # Fix the specific emoji pattern: ğŸ"Š (which is UTF-8 bytes for 📊 read as Latin-1)
        # Try multiple approaches
        report_text = report_text.replace('ğŸ"Š', '📊')
        report_text = report_text.replace('ğŸ"', '📊')
        # Use regex to catch variations
        report_text = re.sub(r'##\s*[^\x00-\x7F]+\s*Data Review Workflow', '## 📊 Data Review Workflow', report_text)
        # More specific: replace any non-ASCII characters before "Data Review Workflow"
        report_text = re.sub(r'##\s*[^#\s📊\w]+\s*Data Review Workflow', '## 📊 Data Review Workflow', report_text)
        
        output_path.write_text(report_text, encoding='utf-8')
        logger.info(f"Report generated: {output_path}")
        
        # Automatically generate PDF version
        try:
            from trackcoach.export import markdown_to_pdf
            pdf_path = output_path.with_suffix('.pdf')
            markdown_to_pdf(output_path, pdf_path)
            logger.info(f"PDF version automatically generated: {pdf_path}")
        except ImportError:
            logger.warning("reportlab not installed, skipping PDF generation. Install with: pip install reportlab")
        except Exception as e:
            logger.warning(f"Failed to generate PDF version: {e}")


