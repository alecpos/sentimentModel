#!/usr/bin/env python3
"""
Implementation Status Verification Engine

This script analyzes files marked as "IMPLEMENTED" to verify they genuinely 
represent completed implementation rather than placeholders or works-in-progress 
that may have been incorrectly classified.

Usage:
    python verify_implementation_status.py [--threshold THRESHOLD] [--path PATH] [--fix] [--report-file REPORT_FILE]
"""

import os
import re
import sys
import json
import argparse
import subprocess
import datetime
from typing import Dict, List, Set, Tuple, Optional
import csv

# Attempt to import spaCy for NLP processing if available
try:
    import spacy
    SPACY_AVAILABLE = True
    # Load a small English model for lightweight NLP
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If the model isn't installed, attempt to download it
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                         check=True, capture_output=True)
            nlp = spacy.load("en_core_web_sm")
        except:
            SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False

# Check if Git is available for temporal analysis
try:
    subprocess.run(["git", "--version"], check=True, capture_output=True)
    GIT_AVAILABLE = True
except (subprocess.SubprocessError, FileNotFoundError):
    GIT_AVAILABLE = False

class Colors:
    """Terminal color codes for formatting output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ChangeFrequencyAnalyzer:
    """
    Analyzes Git history to determine file change frequency.
    Used for identifying volatile documentation that may require more frequent validation.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            base_dir: Base directory for the Git repository
        """
        self.base_dir = base_dir
        self.is_git_repo = self._check_is_git_repo()
        self.change_frequency_cache = {}  # Cache for change frequency results
        
    def _check_is_git_repo(self) -> bool:
        """Check if the base directory is a Git repository."""
        if not GIT_AVAILABLE:
            return False
            
        try:
            result = subprocess.run(
                ["git", "-C", self.base_dir, "rev-parse", "--is-inside-work-tree"],
                check=True, capture_output=True, text=True
            )
            return result.stdout.strip().lower() == "true"
        except subprocess.SubprocessError:
            return False
    
    def get_change_frequency(self, file_path: str, months: int = 6) -> Dict[str, any]:
        """
        Calculate change frequency metrics for a file based on Git history.
        
        Args:
            file_path: Path to the file (relative to base_dir)
            months: Number of months to analyze
            
        Returns:
            Dictionary with change frequency metrics
        """
        # Return cached result if available
        cache_key = f"{file_path}:{months}"
        if cache_key in self.change_frequency_cache:
            return self.change_frequency_cache[cache_key]
            
        # Default result if Git is not available
        default_result = {
            "commits_per_month": 0,
            "last_modified": None,
            "contributors": [],
            "is_volatile": False
        }
        
        if not self.is_git_repo or not GIT_AVAILABLE:
            return default_result
        
        # Get full file path
        full_path = os.path.join(self.base_dir, file_path)
        if not os.path.exists(full_path):
            return default_result
            
        try:
            # Get the last modified date
            last_modified_cmd = [
                "git", "-C", self.base_dir, "log", "-1", "--format=%at", "--", file_path
            ]
            last_modified_result = subprocess.run(
                last_modified_cmd, check=True, capture_output=True, text=True
            )
            
            if last_modified_result.stdout.strip():
                last_modified = datetime.datetime.fromtimestamp(
                    int(last_modified_result.stdout.strip())
                )
            else:
                last_modified = None
            
            # Get commit count in the specified period
            since_date = (datetime.datetime.now() - datetime.timedelta(days=months*30)).strftime("%Y-%m-%d")
            count_cmd = [
                "git", "-C", self.base_dir, "log", 
                f"--since={since_date}", "--format=%H", "--", file_path
            ]
            count_result = subprocess.run(
                count_cmd, check=True, capture_output=True, text=True
            )
            
            commit_count = len(count_result.stdout.strip().split('\n')) if count_result.stdout.strip() else 0
            commits_per_month = commit_count / max(months, 1)
            
            # Get contributors
            contributors_cmd = [
                "git", "-C", self.base_dir, "log", 
                f"--since={since_date}", "--format=%an", "--", file_path
            ]
            contributors_result = subprocess.run(
                contributors_cmd, check=True, capture_output=True, text=True
            )
            
            contributors = list(set(
                contributor for contributor in contributors_result.stdout.strip().split('\n')
                if contributor
            ))
            
            # Determine if the file is volatile (frequent changes or recent modification)
            is_volatile = commits_per_month >= 1.0
            
            result = {
                "commits_per_month": round(commits_per_month, 2),
                "last_modified": last_modified.isoformat() if last_modified else None,
                "contributors": contributors,
                "is_volatile": is_volatile
            }
            
            # Cache the result
            self.change_frequency_cache[cache_key] = result
            return result
            
        except subprocess.SubprocessError:
            return default_result

class ImplementationVerifier:
    """
    Tool to verify if files marked as IMPLEMENTED are genuinely complete.
    Uses multiple heuristics and a weighted scoring system to detect false positives.
    """
    
    def __init__(self, base_dir: str, threshold: float, change_analyzer: ChangeFrequencyAnalyzer = None):
        """
        Initialize the verifier with the documentation base directory.
        
        Args:
            base_dir: Base directory to search for documentation files
            threshold: Score threshold below which files are considered misclassified
            change_analyzer: Optional change frequency analyzer for temporal analysis
        """
        self.base_dir = os.path.abspath(base_dir)
        self.status_pattern = re.compile(r'\*\*IMPLEMENTATION STATUS: ([\w_]+)\*\*')
        
        # Initialize change frequency analyzer for temporal analysis
        self.change_analyzer = change_analyzer or ChangeFrequencyAnalyzer(self.base_dir)
        
        # Placeholder patterns for detecting incomplete docs with weights
        self.placeholder_patterns = [
            # Explicit with high confidence
            (r"\btodo\b", 0.8),
            (r"\bfixme\b", 0.9),
            (r"\bwip\b", 0.85),
            (r"\btbd\b", 0.75),
            (r"\bxxx\b", 0.7),
            # Implicit with medium confidence
            (r"coming\s*soon", 0.7),
            (r"will\s*be\s*implemented", 0.65),
            (r"under\s*development", 0.7),
            (r"planned\s*for\s*future", 0.6),
            (r"not\s*yet\s*available", 0.7),
            (r"future\s*release", 0.6),
            (r"to\s*be\s*determined", 0.6),
            (r"needs\s*more\s*details", 0.5),
            (r"needs\s*documentation", 0.6),
            (r"requires\s*clarification", 0.5),
            (r"awaiting\s*implementation", 0.75),
            (r"will\s*be\s*provided", 0.6),
            (r"will\s*be\s*added", 0.6),
            (r"placeholder", 0.75)
        ]
        
        # Compile all patterns for efficiency
        # We'll just use the patterns without weights for backward compatibility
        self.placeholder_regex = re.compile(
            '|'.join(pattern for pattern, _ in self.placeholder_patterns), 
            re.IGNORECASE
        )
        
        self.threshold = threshold
    
    def get_implemented_files(self, path_filter: Optional[str] = None) -> List[str]:
        """
        Find all files marked as IMPLEMENTED.
        
        Args:
            path_filter: Optional filter to only include files in specific paths
            
        Returns:
            List of file paths (relative to base_dir) marked as IMPLEMENTED
        """
        # Use the filter_docs.py tool to find implemented files
        cmd = ['python', 'filter_docs.py', '--status', 'IMPLEMENTED', '--csv']
        if path_filter:
            cmd.extend(['--path', path_filter])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running filter_docs.py: {result.stderr}")
            return []
        
        # Parse the CSV output, skipping the header
        implemented_files = []
        for line in result.stdout.strip().split('\n')[1:]:
            if line.startswith('IMPLEMENTED,'):
                file_path = line.split(',', 1)[1].strip()
                implemented_files.append(file_path)
        
        return implemented_files
    
    def extract_sections(self, content: str) -> Dict[str, str]:
        """
        Extract sections from a Markdown document based on headings.
        
        Args:
            content: Markdown content to analyze
            
        Returns:
            Dictionary mapping section names to their content
        """
        # Find all level 2 headings (##)
        headings = re.findall(r'^## (.+)$', content, re.MULTILINE)
        sections = {}
        
        # Extract content for each section
        for i, heading in enumerate(headings):
            # Find the start position of this section
            section_start = content.find(f"## {heading}")
            
            # Find the end position (either the next section or the end of the document)
            if i < len(headings) - 1:
                next_section = f"## {headings[i+1]}"
                section_end = content.find(next_section, section_start)
            else:
                section_end = len(content)
            
            # Extract the section content
            section_content = content[section_start:section_end]
            sections[heading] = section_content
        
        # If no sections were found, consider the whole document as one section
        if not sections:
            sections["Main"] = content
        
        return sections
    
    def classify_section_type(self, section_name: str) -> str:
        """
        Classify the type of section based on its name.
        
        Args:
            section_name: Name of the section to classify
            
        Returns:
            Section type classification
        """
        section_types = {
            'introduction': [
                r"introduction", r"overview", r"summary", r"about", 
                r"description", r"background"
            ],
            'usage': [
                r"usage", r"getting\s*started", r"quickstart", r"how\s*to", 
                r"tutorial", r"guide"
            ],
            'api': [
                r"api", r"reference", r"endpoints", r"methods", r"functions", 
                r"parameters", r"returns", r"arguments"
            ],
            'examples': [
                r"examples?", r"sample", r"demo", r"usage\s*example", 
                r"example\s*usage", r"code\s*sample"
            ],
            'configuration': [
                r"configuration", r"settings", r"options", r"preferences", 
                r"setup", r"installation"
            ],
            'troubleshooting': [
                r"troubleshooting", r"faq", r"common\s*issues", r"problems", 
                r"debugging", r"known\s*issues"
            ],
            'related_docs': [
                r"related\s*documentation", r"references", r"see\s*also", 
                r"resources", r"external\s*links", r"further\s*reading", 
                r"additional\s*resources"
            ]
        }
        
        for section_type, patterns in section_types.items():
            if any(re.search(pattern, section_name, re.IGNORECASE) for pattern in patterns):
                return section_type
        
        return "other"
    
    def is_reference_section(self, section_name: str) -> bool:
        """
        Determine if a section is a reference section that should not be evaluated
        for implementation completeness.
        
        Args:
            section_name: Name of the section to check
            
        Returns:
            True if this is a reference section, False otherwise
        """
        return self.classify_section_type(section_name) == "related_docs"
    
    def is_meaningful_placeholder(self, text: str, section_type: str) -> Tuple[bool, float, List[str]]:
        """
        Context-aware placeholder detection combining:
        - Syntactic patterns (regex)
        - Optional semantic analysis (spaCy NLP)
        - Document section context
        
        Args:
            text: Text to check for placeholders
            section_type: Type of section the text is in
            
        Returns:
            Tuple containing:
            - Whether the text contains meaningful placeholders
            - Confidence score
            - List of detected placeholder text
        """
        # For reference sections, reduce sensitivity to placeholders
        if section_type == "related_docs":
            confidence_multiplier = 0.5
        # For introduction/overview sections, increase sensitivity
        elif section_type == "introduction":
            confidence_multiplier = 1.2
        # For API reference sections, increase sensitivity
        elif section_type == "api":
            confidence_multiplier = 1.1
        else:
            confidence_multiplier = 1.0
        
        detected_placeholders = []
        confidence_scores = []
        
        # Check for all patterns
        for pattern, base_weight in self.placeholder_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Apply context to the confidence score
                adjusted_weight = base_weight * confidence_multiplier
                confidence_scores.append(adjusted_weight)
                detected_placeholders.extend(matches)
        
        # If no placeholders found with regex, return quickly
        if not detected_placeholders:
            return False, 0.0, []
        
        # If spaCy is available, use NLP for additional context
        if SPACY_AVAILABLE and len(text) < 5000:  # Only process reasonably sized text
            try:
                doc = nlp(text)
                
                # Check for future tense around placeholder terms
                future_tense_markers = ['will', 'shall', 'going to', 'plan to']
                
                for placeholder in detected_placeholders:
                    placeholder_position = text.lower().find(placeholder.lower())
                    
                    # Skip if we couldn't find the placeholder (shouldn't happen)
                    if placeholder_position == -1:
                        continue
                    
                    # Extract context around the placeholder (100 chars before and after)
                    start = max(0, placeholder_position - 100)
                    end = min(len(text), placeholder_position + len(placeholder) + 100)
                    context = text[start:end]
                    
                    # Check if future tense is used in the context
                    if any(marker in context.lower() for marker in future_tense_markers):
                        confidence_scores = [score * 1.2 for score in confidence_scores]
                        break  # Only apply this bonus once
            except:
                # If NLP processing fails, just continue with regex results
                pass
        
        # Calculate average confidence score
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Determine if this is a meaningful placeholder
        is_meaningful = avg_confidence > 0.6
        
        return is_meaningful, avg_confidence, list(set(detected_placeholders))
    
    def calculate_length_score(self, content: str, sections: Dict[str, str]) -> float:
        """
        Calculate score based on document length and section content length.
        Weight: 20% of total score.
        
        Args:
            content: Full document content
            sections: Dictionary of sections extracted from the document
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Total content length (excluding code blocks)
        clean_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content_length = len(clean_content)
        
        # Score based on total content length
        if content_length > 5000:
            score += 0.5
        elif content_length > 2000:
            score += 0.3
        elif content_length > 1000:
            score += 0.2
        else:
            score += 0.1
        
        # Score based on section length
        non_reference_sections = {name: content for name, content in sections.items() 
                              if not self.is_reference_section(name)}
        
        if non_reference_sections:
            section_scores = []
            for section_content in non_reference_sections.values():
                clean_section = re.sub(r'```.*?```', '', section_content, flags=re.DOTALL)
                section_length = len(clean_section)
                
                if section_length > 1000:
                    section_scores.append(0.5)
                elif section_length > 500:
                    section_scores.append(0.4)
                elif section_length > 300:
                    section_scores.append(0.3)
                elif section_length > 200:
                    section_scores.append(0.2)
                else:
                    section_scores.append(0.1)
            
            # Average section score
            avg_section_score = sum(section_scores) / len(section_scores)
            score += avg_section_score
        
        # Normalize to 0-1 range
        return min(score / 2.0, 1.0)
    
    def calculate_structure_score(self, content: str, sections: Dict[str, str]) -> float:
        """
        Calculate score based on document structure.
        Weight: 30% of total score.
        
        Args:
            content: Full document content
            sections: Dictionary of sections extracted from the document
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Score based on number of sections
        if len(sections) >= 5:
            score += 0.3
        elif len(sections) >= 3:
            score += 0.2
        elif len(sections) >= 1:
            score += 0.1
        
        # Score based on heading structure
        heading_levels = {}
        for level in range(1, 5):
            pattern = r'^' + '#' * level + r'\s+.+$'
            headings = re.findall(pattern, content, re.MULTILINE)
            heading_levels[level] = len(headings)
        
        # Good structure has a mix of heading levels
        if heading_levels.get(1, 0) >= 1 and heading_levels.get(2, 0) >= 2:
            score += 0.2
        elif heading_levels.get(1, 0) >= 1:
            score += 0.1
        
        # Additional score for deeper heading structure
        if heading_levels.get(3, 0) >= 1:
            score += 0.1
        if heading_levels.get(4, 0) >= 1:
            score += 0.1
        
        # Check for balanced content across sections
        non_reference_sections = {name: content for name, content in sections.items() 
                              if not self.is_reference_section(name)}
        
        if len(non_reference_sections) >= 2:
            lengths = [len(section) for section in non_reference_sections.values()]
            avg_length = sum(lengths) / len(lengths)
            
            # Check how many sections are within 50% of average length
            balanced_sections = sum(1 for length in lengths 
                                 if length >= avg_length * 0.5)
            
            balance_ratio = balanced_sections / len(lengths)
            score += balance_ratio * 0.3
        
        # Check for expected section types
        expected_section_patterns = [
            r"introduction|overview|summary",
            r"usage|getting\s*started|quickstart",
            r"api|reference|endpoints|interface",
            r"examples?|sample|demo|tutorial",
            r"configuration|settings|options",
            r"troubleshooting|faq|common\s*issues"
        ]
        
        section_matches = 0
        for pattern in expected_section_patterns:
            if any(re.search(pattern, name, re.IGNORECASE) for name in sections.keys()):
                section_matches += 1
        
        section_coverage = min(section_matches / 3, 1.0)  # Expect at least 3 key sections
        score += section_coverage * 0.3
        
        return min(score, 1.0)
    
    def calculate_content_quality_score(self, content: str, sections: Dict[str, str]) -> float:
        """
        Calculate score based on content quality indicators.
        Weight: 50% of total score.
        
        Args:
            content: Full document content
            sections: Dictionary of sections extracted from the document
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check for code examples
        code_blocks = re.findall(r'```(?:python|json|bash|sh|yaml|typescript|javascript)?.*?```', 
                             content, re.DOTALL)
        
        if len(code_blocks) >= 3:
            score += 0.2
        elif len(code_blocks) >= 1:
            score += 0.1
        
        # Check for inline code references (parameter names, functions, etc.)
        inline_code_refs = re.findall(r'`[^`]+`', content)
        
        if len(inline_code_refs) >= 10:
            score += 0.15
        elif len(inline_code_refs) >= 5:
            score += 0.1
        elif len(inline_code_refs) >= 1:
            score += 0.05
        
        # Check for tables (structured data)
        if re.search(r'\|.*\|.*\|.*\n\|[\-:]+\|', content):
            score += 0.1
        
        # Check for images/diagrams
        images = re.findall(r'!\[.*?\]\(.*?\)', content)
        if len(images) >= 1:
            score += 0.1
        
        # Check for links to other resources
        links = re.findall(r'\[.*?\]\(.*?\)', content)
        if len(links) >= 5:
            score += 0.1
        elif len(links) >= 1:
            score += 0.05
        
        # Check for API parameter documentation
        api_params = re.findall(r'(?:param|parameter|argument|option)s?[:\s]+.*?(?:[\-\*]\s+`[^`]+`\s+(?:\(.*?\))?\s*\-\s*.*?)+', 
                            content, re.DOTALL | re.IGNORECASE)
        
        if api_params:
            score += 0.15
        
        # Check for return value documentation
        if re.search(r'returns?[:\s]+.*?`.*?`', content, re.IGNORECASE):
            score += 0.1
        
        # Check for error handling/exceptions
        if re.search(r'(?:error|exception|throws?)s?[:\s]+.*?(?:[\-\*]\s+`[^`]+`\s*\-\s*.*?)+', 
                   content, re.DOTALL | re.IGNORECASE):
            score += 0.1
        
        # Check for examples of usage
        example_patterns = [
            r'example\s+usage',
            r'usage\s+example',
            r'for\s+example',
            r'as\s+an\s+example',
            r'examples?:',
        ]
        
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in example_patterns):
            score += 0.1
        
        # Deduct score for placeholder content
        placeholder_scores = []
        for section_name, section_content in sections.items():
            section_type = self.classify_section_type(section_name)
            is_placeholder, confidence, _ = self.is_meaningful_placeholder(section_content, section_type)
            if is_placeholder:
                placeholder_scores.append(confidence)
        
        # Calculate placeholder penalty based on detected placeholders
        if placeholder_scores:
            avg_placeholder_confidence = sum(placeholder_scores) / len(placeholder_scores)
            placeholder_penalty = min(avg_placeholder_confidence * 0.6, 0.5)
            score = max(score - placeholder_penalty, 0.0)
        
        return min(score, 1.0)
    
    def detect_document_type(self, file_path: str, content: str) -> str:
        """
        Determine the type of document based on path and content analysis.
        
        Args:
            file_path: Path to the document
            content: Content of the document
            
        Returns:
            Document type classification
        """
        # Path-based classification
        path_lower = file_path.lower()
        
        if 'api' in path_lower or '/reference/' in path_lower:
            return 'api_reference'
        elif 'user_guide' in path_lower or '/guide/' in path_lower or '/tutorial/' in path_lower:
            return 'user_guide'
        elif 'architecture' in path_lower or 'design' in path_lower or 'decision' in path_lower:
            return 'architecture_decision'
        
        # Content-based classification
        sections = self.extract_sections(content)
        section_types = [self.classify_section_type(name) for name in sections.keys()]
        
        # Count the occurrence of different section types
        type_counts = {}
        for section_type in section_types:
            type_counts[section_type] = type_counts.get(section_type, 0) + 1
        
        # Determine document type based on dominant section types
        if type_counts.get('api', 0) >= 2:
            return 'api_reference'
        elif (type_counts.get('usage', 0) + type_counts.get('examples', 0)) >= 2:
            return 'user_guide'
        elif type_counts.get('introduction', 0) >= 1 and 'architecture' in content.lower():
            return 'architecture_decision'
        
        # Default to general documentation
        return 'general_documentation'
    
    def get_scoring_weights(self, document_type: str) -> Tuple[float, float, float]:
        """
        Determine appropriate scoring weights based on document type.
        
        Args:
            document_type: Type of document
            
        Returns:
            Tuple of (length_weight, structure_weight, content_weight)
        """
        # Document type-specific weights
        weights = {
            'api_reference': (0.15, 0.25, 0.60),
            'user_guide': (0.20, 0.30, 0.50),
            'architecture_decision': (0.25, 0.40, 0.35),
            'general_documentation': (0.20, 0.30, 0.50)
        }
        
        return weights.get(document_type, (0.20, 0.30, 0.50))
    
    def detect_suspicious_patterns(self, content: str) -> List[str]:
        """
        Detect suspicious patterns that might indicate incomplete implementation.
        
        Args:
            content: Document content to analyze
            
        Returns:
            List of suspicious patterns found in the document
        """
        suspicious_patterns = []
        
        # Extract sections for analysis
        sections = self.extract_sections(content)
        
        # Check for explicit placeholder text in each section
        for section_name, section_content in sections.items():
            section_type = self.classify_section_type(section_name)
            is_placeholder, confidence, detected_placeholders = self.is_meaningful_placeholder(
                section_content, section_type
            )
            
            if is_placeholder and detected_placeholders:
                for placeholder in detected_placeholders[:3]:  # Limit to 3 placeholders per section
                    suspicious_patterns.append(f"Placeholder text in '{section_name}': '{placeholder}'")
        
        # Check for suspiciously short sections
        for name, section_content in sections.items():
            if not self.is_reference_section(name):
                # Remove headings and code blocks for length calculation
                clean_section = re.sub(r'^##+ .*$', '', section_content, flags=re.MULTILINE)
                clean_section = re.sub(r'```.*?```', '', clean_section, flags=re.DOTALL)
                
                if len(clean_section.strip()) < 200:
                    suspicious_patterns.append(f"Suspiciously short section: '{name}' ({len(clean_section.strip())} chars)")
        
        # Check for excessive use of ellipses
        if content.count('...') > 3:
            suspicious_patterns.append(f"Excessive use of ellipses (found {content.count('...')} instances)")
        
        # Check for commented-out content
        commented_content = re.findall(r'<!--.*?-->', content, re.DOTALL)
        if commented_content:
            suspicious_patterns.append(f"Contains {len(commented_content)} commented-out sections")
        
        # Check for missing expected sections
        expected_sections = ["Introduction", "Usage", "Examples", "Reference"]
        section_names = sections.keys()
        
        missing_sections = []
        for expected in expected_sections:
            if not any(re.search(expected, name, re.IGNORECASE) for name in section_names):
                missing_sections.append(expected)
        
        if missing_sections:
            suspicious_patterns.append(f"Missing expected sections: {', '.join(missing_sections)}")
        
        return suspicious_patterns
    
    def calculate_total_score(self, file_path: str) -> Tuple[float, Dict[str, any], List[str]]:
        """
        Calculate total verification score for a file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Tuple containing:
            - Total score (0.0 to 1.0)
            - Component scores dictionary
            - List of suspicious patterns
        """
        try:
            full_path = os.path.join(self.base_dir, file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            sections = self.extract_sections(content)
            
            # Detect document type for adaptive scoring
            document_type = self.detect_document_type(file_path, content)
            
            # Get appropriate weights for this document type
            length_weight, structure_weight, content_weight = self.get_scoring_weights(document_type)
            
            # Calculate component scores
            length_score = self.calculate_length_score(content, sections)
            structure_score = self.calculate_structure_score(content, sections)
            quality_score = self.calculate_content_quality_score(content, sections)
            
            # Apply adaptive weights
            weighted_score = (
                length_score * length_weight +
                structure_score * structure_weight +
                quality_score * content_weight
            )
            
            # Detect suspicious patterns
            suspicious_patterns = self.detect_suspicious_patterns(content)
            
            # Get change frequency metrics
            change_frequency = self.change_analyzer.get_change_frequency(file_path)
            
            # Apply a volatility penalty for frequently changing files
            # The idea is that volatile files are more likely to become outdated
            volatility_penalty = 0.0
            if change_frequency["is_volatile"]:
                # Apply a small penalty to the score for volatile files
                volatility_penalty = min(change_frequency["commits_per_month"] * 0.02, 0.1)
                weighted_score = max(weighted_score - volatility_penalty, 0.0)
            
            component_scores = {
                "document_type": document_type,
                "length_score": length_score,
                "structure_score": structure_score,
                "quality_score": quality_score,
                "length_weight": length_weight,
                "structure_weight": structure_weight,
                "content_weight": content_weight,
                "change_frequency": change_frequency,
                "volatility_penalty": volatility_penalty,
                "weighted_score": weighted_score
            }
            
            return weighted_score, component_scores, suspicious_patterns
            
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            # Handle non-text files or other errors
            return 0.0, {
                "document_type": "unknown",
                "length_score": 0.0,
                "structure_score": 0.0,
                "quality_score": 0.0,
                "length_weight": 0.2,
                "structure_weight": 0.3,
                "content_weight": 0.5,
                "change_frequency": {
                    "commits_per_month": 0,
                    "last_modified": None,
                    "contributors": [],
                    "is_volatile": False
                },
                "volatility_penalty": 0.0,
                "weighted_score": 0.0
            }, ["File could not be analyzed (not a text file or not found)"]
    
    def get_revalidation_priority(self, score: float, change_frequency: Dict[str, any]) -> str:
        """
        Determine revalidation priority based on score and change frequency.
        
        Args:
            score: Verification score
            change_frequency: Change frequency metrics
            
        Returns:
            Priority level ("high", "medium", "low")
        """
        # High priority for revalidation if:
        # - The score is questionable (0.5-0.75) AND the file is volatile
        # - OR the score is low (< 0.5)
        if score < 0.5:
            return "high"
        elif score < 0.75 and change_frequency["is_volatile"]:
            return "high"
        # Medium priority if:
        # - The score is questionable but the file is not volatile
        # - OR the score is good but the file is very volatile (> 2 commits/month)
        elif score < 0.75:
            return "medium"
        elif change_frequency["commits_per_month"] > 2.0:
            return "medium"
        # Low priority otherwise
        else:
            return "low"
            
    def should_revalidate(self, file_path: str) -> Tuple[bool, str, Dict[str, any]]:
        """
        Determine if a file should be revalidated based on its score and change history.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple containing:
            - Whether the file should be revalidated
            - Priority reason
            - Additional metrics
        """
        # Check if we have historical data for this file
        report_dir = os.path.join(self.base_dir, "../reports")
        historical_scores = {}
        
        if os.path.isdir(report_dir):
            # Find the most recent report for this file
            report_files = [f for f in os.listdir(report_dir) if f.endswith('.json')]
            
            for report_file in sorted(report_files, reverse=True)[:5]:  # Check the 5 most recent reports
                try:
                    with open(os.path.join(report_dir, report_file), 'r') as f:
                        report_data = json.load(f)
                    
                    # Look for this file in the report
                    for file_data in report_data.get("files", []):
                        if file_data.get("file_path") == file_path:
                            score = file_data.get("score", 0.0)
                            historical_scores[report_data.get("timestamp", "unknown")] = score
                            break
                except:
                    continue
        
        # Calculate current score
        score, component_scores, _ = self.calculate_total_score(file_path)
        change_frequency = component_scores.get("change_frequency", {})
        
        # Determine revalidation priority
        priority = self.get_revalidation_priority(score, change_frequency)
        
        # Detect score trend if we have historical data
        score_trend = "stable"
        if historical_scores:
            # Get scores ordered by timestamp
            ordered_scores = [score for _, score in 
                             sorted(historical_scores.items(), reverse=True)]
            
            if ordered_scores and ordered_scores[0] < score - 0.1:
                score_trend = "improving"
            elif ordered_scores and ordered_scores[0] > score + 0.1:
                score_trend = "degrading"
        
        # Determine whether to revalidate
        should_revalidate = priority in ["high", "medium"]
        
        # Determine reason for revalidation recommendation
        if priority == "high" and score < 0.5:
            reason = "Low verification score"
        elif priority == "high":
            reason = "Questionable score with high change frequency"
        elif priority == "medium" and score < 0.75:
            reason = "Questionable score"
        elif priority == "medium":
            reason = "High change frequency"
        else:
            reason = "No immediate revalidation needed"
        
        metrics = {
            "score": score,
            "priority": priority,
            "change_frequency": change_frequency,
            "historical_scores": historical_scores,
            "score_trend": score_trend
        }
        
        return should_revalidate, reason, metrics
    
    def classify_implementation_status(self, score: float) -> str:
        """
        Classify implementation status based on score.
        
        Args:
            score: Verification score between 0.0 and 1.0
            
        Returns:
            Classification label
        """
        if score >= 0.9:
            return "Verified Implemented"
        elif score >= 0.75:
            return "Likely Implemented"
        elif score >= 0.5:
            return "Questionably Implemented"
        else:
            return "Falsely Marked"
    
    def update_status_marker(self, file_path: str, status: str) -> bool:
        """
        Update implementation status marker in a file.
        
        Args:
            file_path: Path to the file to update
            status: New status to set
            
        Returns:
            True if the file was updated, False otherwise
        """
        try:
            full_path = os.path.join(self.base_dir, file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace existing status marker
            new_content = re.sub(
                r'\*\*IMPLEMENTATION STATUS: IMPLEMENTED\*\*',
                f'**IMPLEMENTATION STATUS: {status}**',
                content
            )
            
            if new_content != content:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
            
            return False
            
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            # Handle non-text files or other errors
            return False
    
    def verify_all_implemented_files(self, path_filter: str = None) -> Dict[str, any]:
        """
        Verify all files marked as IMPLEMENTED in the base directory.
        
        Args:
            path_filter: Optional regex pattern to filter file paths
            
        Returns:
            Dictionary with verification results
        """
        implemented_files = self.get_implemented_files(path_filter)
        verification_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "base_directory": self.base_dir,
            "threshold": self.threshold,
            "path_filter": path_filter,
            "files": [],
            "document_types": {},
            "revalidation_recommendations": []
        }
        
        for file_path in implemented_files:
            score, component_scores, suspicious_patterns = self.calculate_total_score(file_path)
            classification = self.classify_implementation_status(score)
            
            # Check if file needs revalidation
            should_revalidate, reason, revalidation_metrics = self.should_revalidate(file_path)
            
            file_result = {
                "file_path": file_path,
                "score": score,
                "classification": classification,
                "component_scores": component_scores,
                "suspicious_patterns": suspicious_patterns
            }
            
            # Add revalidation information if applicable
            if should_revalidate:
                file_result["should_revalidate"] = True
                file_result["revalidation_reason"] = reason
                file_result["revalidation_priority"] = revalidation_metrics["priority"]
                
                # Add to revalidation recommendations list
                verification_results["revalidation_recommendations"].append({
                    "file_path": file_path,
                    "priority": revalidation_metrics["priority"],
                    "reason": reason,
                    "score": score,
                    "score_trend": revalidation_metrics["score_trend"],
                    "change_frequency": revalidation_metrics["change_frequency"]
                })
            else:
                file_result["should_revalidate"] = False
            
            verification_results["files"].append(file_result)
            
            # Track document types for reporting
            doc_type = component_scores["document_type"]
            if doc_type not in verification_results["document_types"]:
                verification_results["document_types"][doc_type] = {
                    "count": 0,
                    "verified": 0,
                    "likely": 0,
                    "questionable": 0,
                    "false": 0
                }
            
            verification_results["document_types"][doc_type]["count"] += 1
            
            if classification == "Verified Implemented":
                verification_results["document_types"][doc_type]["verified"] += 1
            elif classification == "Likely Implemented":
                verification_results["document_types"][doc_type]["likely"] += 1
            elif classification == "Questionably Implemented":
                verification_results["document_types"][doc_type]["questionable"] += 1
            elif classification == "Falsely Marked":
                verification_results["document_types"][doc_type]["false"] += 1
        
        # Sort revalidation recommendations by priority
        verification_results["revalidation_recommendations"].sort(
            key=lambda x: (
                0 if x["priority"] == "high" else (1 if x["priority"] == "medium" else 2),
                -x["change_frequency"]["commits_per_month"],
                -float(x["score"])
            )
        )
        
        return verification_results
        
    def generate_console_report(self, verification_results: Dict[str, any]) -> List[str]:
        """
        Generate a console-friendly report from verification results.
        
        Args:
            verification_results: Verification results dictionary
            
        Returns:
            List of report lines
        """
        report = [
            f"📋 IMPLEMENTATION STATUS VERIFICATION REPORT",
            f"Base directory: {verification_results['base_directory']}",
            f"Threshold: {verification_results['threshold']}",
            f"Path filter: {verification_results['path_filter'] or 'None'}",
            f"Timestamp: {verification_results['timestamp']}",
            ""
        ]
        
        # Count files by classification
        total_files = len(verification_results["files"])
        verified_count = sum(1 for f in verification_results["files"] if f["classification"] == "Verified Implemented")
        likely_count = sum(1 for f in verification_results["files"] if f["classification"] == "Likely Implemented")
        questionable_count = sum(1 for f in verification_results["files"] if f["classification"] == "Questionably Implemented")
        false_count = sum(1 for f in verification_results["files"] if f["classification"] == "Falsely Marked")
        
        # Summary statistics
        report.extend([
            "📊 SUMMARY:",
            f"Total files analyzed: {total_files}",
            f"Verified Implemented: {verified_count} ({verified_count/total_files*100:.1f}%)",
            f"Likely Implemented: {likely_count} ({likely_count/total_files*100:.1f}%)",
            f"Questionably Implemented: {questionable_count} ({questionable_count/total_files*100:.1f}%)",
            f"Falsely Marked: {false_count} ({false_count/total_files*100:.1f}%)",
            ""
        ])
        
        # Document type breakdown
        report.append("📚 DOCUMENT TYPES:")
        for doc_type, stats in verification_results["document_types"].items():
            report.append(f"{doc_type}: {stats['count']} files")
            report.append(f"  Verified: {stats['verified']} | Likely: {stats['likely']} | Questionable: {stats['questionable']} | False: {stats['false']}")
        
        report.append("")
        
        # Revalidation recommendations
        revalidation_recs = verification_results.get("revalidation_recommendations", [])
        if revalidation_recs:
            report.append("🔄 REVALIDATION RECOMMENDATIONS:")
            
            # High priority
            high_priority = [r for r in revalidation_recs if r["priority"] == "high"]
            if high_priority:
                report.append("HIGH PRIORITY:")
                for rec in high_priority[:5]:  # Show top 5
                    file_path = rec["file_path"]
                    reason = rec["reason"]
                    commits = rec["change_frequency"]["commits_per_month"]
                    trend = rec["score_trend"]
                    report.append(f"  {file_path}")
                    report.append(f"    Reason: {reason} | Changes: {commits:.1f}/month | Trend: {trend}")
                
                if len(high_priority) > 5:
                    report.append(f"    ... and {len(high_priority) - 5} more high priority files")
            
            # Medium priority
            medium_priority = [r for r in revalidation_recs if r["priority"] == "medium"]
            if medium_priority:
                report.append("MEDIUM PRIORITY:")
                for rec in medium_priority[:3]:  # Show top 3
                    file_path = rec["file_path"]
                    reason = rec["reason"]
                    commits = rec["change_frequency"]["commits_per_month"]
                    report.append(f"  {file_path}")
                    report.append(f"    Reason: {reason} | Changes: {commits:.1f}/month")
                
                if len(medium_priority) > 3:
                    report.append(f"    ... and {len(medium_priority) - 3} more medium priority files")
            
            report.append("")
        
        # Determine which files need attention
        issue_files = [f for f in verification_results["files"] 
                       if f["classification"] in ["Questionably Implemented", "Falsely Marked"]]
        
        if issue_files:
            report.append("⚠️ FILES REQUIRING ATTENTION:")
            
            # Sort by score (lowest first)
            issue_files.sort(key=lambda x: x["score"])
            
            for file_data in issue_files:
                file_path = file_data["file_path"]
                score = file_data["score"]
                classification = file_data["classification"]
                report.append(f"  {file_path} (Score: {score:.2f}) - {classification}")
                
                # Display suspicious patterns
                if file_data["suspicious_patterns"]:
                    pattern_text = ", ".join(file_data["suspicious_patterns"][:3])
                    if len(file_data["suspicious_patterns"]) > 3:
                        pattern_text += f", and {len(file_data['suspicious_patterns']) - 3} more issues"
                    report.append(f"    Concerns: {pattern_text}")
                
                # Show component scores that are low
                for component, value in file_data["component_scores"].items():
                    if component.endswith("_score") and isinstance(value, (int, float)) and value < 0.6:
                        nice_name = component.replace("_score", "").replace("_", " ").title()
                        report.append(f"    Low {nice_name}: {value:.2f}")
        
        report.append("")
        report.append("✅ VERIFICATION COMPLETE")
                
        return report

def main():
    """
    Main function to run the verification tool from the command line.
    """
    parser = argparse.ArgumentParser(description="Verify the implementation status of documentation files.")
    parser.add_argument("--base-dir", default=".", help="Base directory to search for markdown files")
    parser.add_argument("--threshold", type=float, default=0.75, help="Score threshold for verification (default: 0.75)")
    parser.add_argument("--path", type=str, help="Only process files matching this regex pattern")
    parser.add_argument("--fix", action="store_true", help="Fix status markers for misclassified files")
    parser.add_argument("--save-config", type=str, help="Save current configuration to specified file")
    parser.add_argument("--load-config", type=str, help="Load configuration from specified file")
    parser.add_argument("--report-file", type=str, help="Save detailed report to specified JSON file")
    parser.add_argument("--csv", type=str, help="Save CSV report to specified file")
    parser.add_argument("--revalidation-report", type=str, help="Generate a revalidation priority report to specified file")
    parser.add_argument("--check-volatility", action="store_true", help="Analyze change frequency from Git history")
    
    args = parser.parse_args()
    
    # If loading configuration
    if args.load_config:
        try:
            with open(args.load_config, 'r') as f:
                config = json.load(f)
                print(f"Loading configuration from {args.load_config}")
                
                # Override command line arguments with config file
                if "base_dir" in config:
                    args.base_dir = config["base_dir"]
                if "threshold" in config:
                    args.threshold = config["threshold"]
                if "path" in config:
                    args.path = config["path"]
                if "fix" in config:
                    args.fix = config["fix"]
                if "report_file" in config:
                    args.report_file = config["report_file"]
                if "check_volatility" in config:
                    args.check_volatility = config["check_volatility"]
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return 1
    
    # Initialize the verifier with change frequency analyzer if requested
    if args.check_volatility:
        change_analyzer = ChangeFrequencyAnalyzer(args.base_dir)
        verifier = ImplementationVerifier(
            base_dir=args.base_dir, 
            threshold=args.threshold,
            change_analyzer=change_analyzer
        )
    else:
        verifier = ImplementationVerifier(
            base_dir=args.base_dir, 
            threshold=args.threshold
        )
    
    results = verifier.verify_all_implemented_files(
        path_filter=args.path
    )
    
    # Apply fixes if requested
    if args.fix:
        updated_count = 0
        for file_data in results["files"]:
            if file_data["classification"] in ["Questionably Implemented", "Falsely Marked"]:
                # Map score ranges to appropriate status
                score = file_data["score"]
                if score < 0.5:
                    new_status = "NOT_IMPLEMENTED"
                else:
                    new_status = "PARTIALLY_IMPLEMENTED"
                
                if verifier.update_status_marker(file_data["file_path"], new_status):
                    updated_count += 1
                    file_data["status_updated"] = True
                    file_data["new_status"] = new_status
        
        print(f"Updated status markers in {updated_count} files")
    
    # Save current configuration if requested
    if args.save_config:
        try:
            config = {
                "base_dir": args.base_dir,
                "threshold": args.threshold,
                "path": args.path,
                "fix": args.fix,
                "check_volatility": args.check_volatility
            }
            
            with open(args.save_config, 'w') as f:
                json.dump(config, f, indent=2)
                print(f"Configuration saved to {args.save_config}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    # Print console report
    print("\n".join(verifier.generate_console_report(results)))
    
    # Generate revalidation report if requested
    if args.revalidation_report:
        try:
            # Generate a report with just revalidation info
            revalidation_report = {
                "timestamp": results["timestamp"],
                "base_directory": results["base_directory"],
                "recommendations": results["revalidation_recommendations"],
                "summary": {
                    "high_priority": len([r for r in results["revalidation_recommendations"] if r["priority"] == "high"]),
                    "medium_priority": len([r for r in results["revalidation_recommendations"] if r["priority"] == "medium"]),
                    "low_priority": len([r for r in results["revalidation_recommendations"] if r["priority"] == "low"])
                }
            }
            
            with open(args.revalidation_report, 'w') as f:
                json.dump(revalidation_report, f, indent=2)
                print(f"Revalidation report saved to {args.revalidation_report}")
        except Exception as e:
            print(f"Error saving revalidation report: {e}")
    
    # Write JSON report if requested
    if args.report_file:
        try:
            with open(args.report_file, 'w') as f:
                json.dump(results, f, indent=2)
                print(f"Detailed report saved to {args.report_file}")
        except Exception as e:
            print(f"Error saving report: {e}")
    
    # Write CSV report if requested
    if args.csv:
        try:
            with open(args.csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["File", "Score", "Classification", "Document Type", "Needs Revalidation", "Priority"])
                
                for file_data in results["files"]:
                    writer.writerow([
                        file_data["file_path"],
                        file_data["score"],
                        file_data["classification"],
                        file_data["component_scores"]["document_type"],
                        file_data.get("should_revalidate", False),
                        file_data.get("revalidation_priority", "N/A")
                    ])
                
                print(f"CSV report saved to {args.csv}")
        except Exception as e:
            print(f"Error saving CSV report: {e}")
    
    # Return exit code 1 if there are any questionable or falsely marked files
    if any(f["classification"] in ["Questionably Implemented", "Falsely Marked"] for f in results["files"]):
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 