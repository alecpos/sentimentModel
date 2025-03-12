#!/usr/bin/env python3
"""
Semantic Alignment Validator for the WITHIN ML Prediction System.

This script analyzes code and its corresponding docstrings to measure semantic
alignment and conceptual consistency. It uses embedding models to compute
similarity scores between code and documentation, identifying potential
misalignments where code functionality diverges from its documentation.

Usage:
    python validate_semantic_alignment.py [file_or_directory] [options]

Examples:
    # Validate semantic alignment in a file
    python validate_semantic_alignment.py app/models/ml/prediction/ad_score_predictor.py
    
    # Validate semantic alignment in all Python files in a directory with high sensitivity
    python validate_semantic_alignment.py app/models/ml/prediction --recursive --threshold 0.7
    
    # Generate a report of semantic alignment scores
    python validate_semantic_alignment.py app/models/ml/prediction --report alignment_report.json
"""

import os
import re
import ast
import sys
import json
import argparse
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from pathlib import Path

# Try to import embedding libraries, provide instructions if not available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class CodeNormalizer:
    """Normalizes code for semantic analysis."""
    
    @staticmethod
    def normalize_code(code: str) -> str:
        """Normalize code by removing comments, docstrings, and standardizing whitespace.
        
        Args:
            code: The code string to normalize
            
        Returns:
            Normalized code string
        """
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Remove docstrings (simplified approach)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        # Replace common aliases
        replacements = {
            'np': 'numpy',
            'pd': 'pandas',
            'tf': 'tensorflow',
            'plt': 'matplotlib.pyplot',
            'nn': 'neural_network'
        }
        
        for alias, full in replacements.items():
            # Only replace when it appears as a complete word
            code = re.sub(r'\b' + alias + r'\b', full, code)
        
        return code.strip()
    
    @staticmethod
    def extract_function_names(code: str) -> List[str]:
        """Extract function and method names from code.
        
        Args:
            code: The code string to analyze
            
        Returns:
            List of function and method names
        """
        pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        return re.findall(pattern, code)
    
    @staticmethod
    def extract_class_names(code: str) -> List[str]:
        """Extract class names from code.
        
        Args:
            code: The code string to analyze
            
        Returns:
            List of class names
        """
        pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]'
        return re.findall(pattern, code)
    
    @staticmethod
    def extract_variable_names(code: str) -> List[str]:
        """Extract variable names from code.
        
        Args:
            code: The code string to analyze
            
        Returns:
            List of variable names
        """
        # This is simplified and might miss some variables or include keywords
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        return re.findall(pattern, code)
    
    @staticmethod
    def extract_imports(code: str) -> List[str]:
        """Extract imported modules and packages from code.
        
        Args:
            code: The code string to analyze
            
        Returns:
            List of imported module names
        """
        # Match "import x" and "from x import y"
        import_pattern = r'import\s+([a-zA-Z0-9_.]+)'
        from_pattern = r'from\s+([a-zA-Z0-9_.]+)\s+import'
        
        imports = re.findall(import_pattern, code)
        imports.extend(re.findall(from_pattern, code))
        
        return imports
    
    @staticmethod
    def code_to_features(code: str) -> Dict[str, List[str]]:
        """Extract features from code for semantic analysis.
        
        Args:
            code: The code string to analyze
            
        Returns:
            Dictionary of code features by category
        """
        normalized_code = CodeNormalizer.normalize_code(code)
        
        return {
            'functions': CodeNormalizer.extract_function_names(normalized_code),
            'classes': CodeNormalizer.extract_class_names(normalized_code),
            'variables': CodeNormalizer.extract_variable_names(normalized_code),
            'imports': CodeNormalizer.extract_imports(normalized_code),
            'full_text': normalized_code
        }


class DocstringNormalizer:
    """Normalizes docstrings for semantic analysis."""
    
    @staticmethod
    def normalize_docstring(docstring: str) -> str:
        """Normalize docstring by removing formatting markers and standardizing whitespace.
        
        Args:
            docstring: The docstring to normalize
            
        Returns:
            Normalized docstring text
        """
        if not docstring:
            return ""
        
        # Remove triple quotes and leading/trailing whitespace
        docstring = docstring.strip().strip('"""').strip("'''").strip()
        
        # Remove section headers
        docstring = re.sub(r'(Args|Returns|Raises|Examples|Attributes|Note|Warning|Warnings|Todo|See Also|References):', '', docstring)
        
        # Remove parameter type hints
        docstring = re.sub(r'[\w_]+\s*\([^)]+\):', '', docstring)
        
        # Remove example code blocks
        docstring = re.sub(r'```.*?```', '', docstring, flags=re.DOTALL)
        
        # Remove doctest blocks
        lines = []
        for line in docstring.split('\n'):
            if not line.strip().startswith('>>>') and not line.strip().startswith('...'):
                lines.append(line)
        
        docstring = '\n'.join(lines)
        
        # Normalize whitespace
        docstring = re.sub(r'\s+', ' ', docstring)
        
        return docstring.strip()
    
    @staticmethod
    def extract_param_descriptions(docstring: str) -> Dict[str, str]:
        """Extract parameter descriptions from a docstring.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            Dictionary mapping parameter names to their descriptions
        """
        if not docstring:
            return {}
        
        params = {}
        
        # Find the Args section
        args_match = re.search(r'Args:(.*?)(?:Returns:|Raises:|Examples:|Attributes:|Note:|Warning:|Warnings:|Todo:|See Also:|References:|$)', docstring, re.DOTALL)
        if not args_match:
            return params
        
        args_section = args_match.group(1).strip()
        
        # Extract parameter descriptions
        param_pattern = r'(\w+)(?:\s*\([^)]+\))?\s*:(.*?)(?=\n\s*\w+\s*:|$)'
        for match in re.finditer(param_pattern, args_section, re.DOTALL):
            param_name = match.group(1).strip()
            param_desc = match.group(2).strip()
            params[param_name] = param_desc
        
        return params
    
    @staticmethod
    def extract_return_description(docstring: str) -> str:
        """Extract return value description from a docstring.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            Description of the return value, or empty string if not found
        """
        if not docstring:
            return ""
        
        # Find the Returns section
        returns_match = re.search(r'Returns:(.*?)(?:Raises:|Examples:|Attributes:|Note:|Warning:|Warnings:|Todo:|See Also:|References:|$)', docstring, re.DOTALL)
        if not returns_match:
            return ""
        
        returns_section = returns_match.group(1).strip()
        
        # Remove the type hint if present
        return_desc = re.sub(r'^\s*\w+(?:\s*\([^)]+\))?\s*:', '', returns_section).strip()
        
        return return_desc
    
    @staticmethod
    def docstring_to_features(docstring: str) -> Dict[str, Any]:
        """Extract features from a docstring for semantic analysis.
        
        Args:
            docstring: The docstring to analyze
            
        Returns:
            Dictionary of docstring features by category
        """
        if not docstring:
            return {
                'normalized_text': "",
                'params': {},
                'return': "",
                'has_examples': False
            }
        
        normalized_text = DocstringNormalizer.normalize_docstring(docstring)
        param_descriptions = DocstringNormalizer.extract_param_descriptions(docstring)
        return_description = DocstringNormalizer.extract_return_description(docstring)
        has_examples = '>>>' in docstring or '```' in docstring
        
        return {
            'normalized_text': normalized_text,
            'params': param_descriptions,
            'return': return_description,
            'has_examples': has_examples
        }


class EmbeddingProvider:
    """Provides embeddings for semantic similarity analysis."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding provider with the specified model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("Sentence Transformers not available. Using fallback similarity method.")
            self.model = None
        else:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                self.model = None
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
        
        if self.model is None:
            # Fallback to simple character count vector if model is not available
            return self._fallback_embedding(text)
        
        return self.model.encode(text)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        
        if self.model is None:
            return self._fallback_similarity(text1, text2)
        
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        # Compute cosine similarity
        similarity = self._cosine_similarity(embedding1, embedding2)
        
        return similarity
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Create a simple fallback embedding when the model is not available.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple numerical representation of the text
        """
        # This is an extremely simplified embedding approach for fallback only
        # It uses character counts and some basic statistics as features
        features = np.zeros(10)
        
        # Simple character-level features
        features[0] = len(text)
        features[1] = text.count(' ') + 1  # Approximate word count
        features[2] = sum(c.isupper() for c in text)
        features[3] = sum(c.isdigit() for c in text)
        features[4] = sum(c == '.' for c in text)
        features[5] = sum(not c.isalnum() for c in text)
        
        # Some basic linguistic features
        features[6] = text.count('import')
        features[7] = text.count('def ')
        features[8] = text.count('class ')
        features[9] = text.count('return')
        
        # Normalize to unit length
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Compute a simple fallback similarity score when the model is not available.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Approximate similarity score
        """
        # Use Jaccard similarity on word sets as a simple fallback
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity score
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return np.dot(v1, v2) / (norm1 * norm2)


class SemanticAlignmentAnalyzer:
    """Analyzes semantic alignment between code and documentation."""
    
    def __init__(self, threshold: float = 0.5):
        """Initialize the analyzer with specified threshold.
        
        Args:
            threshold: Similarity threshold below which items are considered misaligned
        """
        self.embedding_provider = EmbeddingProvider()
        self.threshold = threshold
    
    def analyze_function(self, func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze semantic alignment between a function and its docstring.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            Analysis results as a dictionary
        """
        func_name = func_node.name
        docstring = ast.get_docstring(func_node)
        func_code = ast.unparse(func_node)
        
        # Skip docstring and decorator lines
        func_body = "\n".join(ast.unparse(stmt) for stmt in func_node.body 
                              if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Str))
        
        # Extract features
        code_features = CodeNormalizer.code_to_features(func_body)
        docstring_features = DocstringNormalizer.docstring_to_features(docstring or "")
        
        # Compute overall semantic similarity
        code_text = code_features['full_text']
        doc_text = docstring_features['normalized_text']
        
        overall_similarity = self.embedding_provider.compute_similarity(code_text, doc_text)
        
        # Check parameter alignment
        param_alignment = {}
        param_names = {arg.arg for arg in func_node.args.args if arg.arg != 'self'}
        doc_params = set(docstring_features['params'].keys())
        
        missing_params = param_names - doc_params
        extra_params = doc_params - param_names
        
        # Compute individual parameter description alignments
        param_similarities = {}
        for param_name in param_names.intersection(doc_params):
            # Find parameter usage in code
            param_usage_pattern = r'\b' + re.escape(param_name) + r'\b'
            param_usage = re.findall(param_usage_pattern, func_body)
            param_context = []
            
            # Extract lines containing the parameter for context
            for line in func_body.split('\n'):
                if re.search(param_usage_pattern, line):
                    param_context.append(line)
            
            param_context_text = ' '.join(param_context)
            param_desc = docstring_features['params'].get(param_name, '')
            
            # Compute similarity between parameter usage and description
            similarity = self.embedding_provider.compute_similarity(param_context_text, param_desc)
            param_similarities[param_name] = similarity
        
        # Check return value alignment
        has_return = any(isinstance(stmt, ast.Return) for stmt in ast.walk(func_node))
        return_desc = docstring_features['return']
        return_statements = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value:
                return_statements.append(ast.unparse(node))
        
        return_context = ' '.join(return_statements)
        return_similarity = self.embedding_provider.compute_similarity(return_context, return_desc)
        
        # Final alignment results
        is_aligned = overall_similarity >= self.threshold
        
        return {
            'name': func_name,
            'type': 'function',
            'has_docstring': docstring is not None,
            'overall_similarity': overall_similarity,
            'is_aligned': is_aligned,
            'param_alignment': {
                'missing_params': list(missing_params),
                'extra_params': list(extra_params),
                'param_similarities': param_similarities
            },
            'return_alignment': {
                'has_return': has_return,
                'has_return_doc': bool(return_desc),
                'return_similarity': return_similarity
            }
        }
    
    def analyze_class(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze semantic alignment between a class and its docstring.
        
        Args:
            class_node: AST node for the class
            
        Returns:
            Analysis results as a dictionary
        """
        class_name = class_node.name
        docstring = ast.get_docstring(class_node)
        class_code = ast.unparse(class_node)
        
        # Skip docstring
        class_body = "\n".join(ast.unparse(stmt) for stmt in class_node.body 
                              if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Str))
        
        # Extract features
        code_features = CodeNormalizer.code_to_features(class_body)
        docstring_features = DocstringNormalizer.docstring_to_features(docstring or "")
        
        # Compute overall semantic similarity
        code_text = code_features['full_text']
        doc_text = docstring_features['normalized_text']
        
        overall_similarity = self.embedding_provider.compute_similarity(code_text, doc_text)
        
        # Analyze methods
        method_results = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_results.append(self.analyze_function(node))
        
        # Final alignment results
        is_aligned = overall_similarity >= self.threshold
        
        return {
            'name': class_name,
            'type': 'class',
            'has_docstring': docstring is not None,
            'overall_similarity': overall_similarity,
            'is_aligned': is_aligned,
            'methods': method_results
        }
    
    def analyze_module(self, module_node: ast.Module) -> Dict[str, Any]:
        """Analyze semantic alignment for an entire module.
        
        Args:
            module_node: AST node for the module
            
        Returns:
            Analysis results as a dictionary
        """
        docstring = ast.get_docstring(module_node)
        module_code = ast.unparse(module_node)
        
        # Skip docstring
        module_body = "\n".join(ast.unparse(stmt) for stmt in module_node.body 
                               if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Str))
        
        # Extract features
        code_features = CodeNormalizer.code_to_features(module_body)
        docstring_features = DocstringNormalizer.docstring_to_features(docstring or "")
        
        # Compute overall semantic similarity
        code_text = code_features['full_text']
        doc_text = docstring_features['normalized_text']
        
        overall_similarity = self.embedding_provider.compute_similarity(code_text, doc_text)
        
        # Analyze classes and functions
        class_results = []
        function_results = []
        
        for node in module_node.body:
            if isinstance(node, ast.ClassDef):
                class_results.append(self.analyze_class(node))
            elif isinstance(node, ast.FunctionDef):
                function_results.append(self.analyze_function(node))
        
        # Final alignment results
        is_aligned = overall_similarity >= self.threshold
        
        return {
            'type': 'module',
            'has_docstring': docstring is not None,
            'overall_similarity': overall_similarity,
            'is_aligned': is_aligned,
            'classes': class_results,
            'functions': function_results
        }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze semantic alignment for a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Analysis results as a dictionary
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            results = self.analyze_module(tree)
            results['file_path'] = file_path
            
            # Calculate aggregate statistics
            all_scores = []
            
            # Add module score
            if results['has_docstring']:
                all_scores.append(results['overall_similarity'])
            
            # Add class scores
            for cls in results['classes']:
                if cls['has_docstring']:
                    all_scores.append(cls['overall_similarity'])
                
                # Add method scores
                for method in cls['methods']:
                    if method['has_docstring']:
                        all_scores.append(method['overall_similarity'])
            
            # Add function scores
            for func in results['functions']:
                if func['has_docstring']:
                    all_scores.append(func['overall_similarity'])
            
            # Calculate average alignment score
            results['average_alignment'] = sum(all_scores) / len(all_scores) if all_scores else 0.0
            results['alignment_scores_count'] = len(all_scores)
            results['misaligned_count'] = sum(1 for score in all_scores if score < self.threshold)
            
            return results
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return {
                'file_path': file_path,
                'error': str(e),
                'type': 'error'
            }


def process_file(file_path: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Process a single Python file for semantic alignment.
    
    Args:
        file_path: Path to the Python file
        threshold: Similarity threshold for alignment
        
    Returns:
        Analysis results as a dictionary
    """
    analyzer = SemanticAlignmentAnalyzer(threshold=threshold)
    results = analyzer.analyze_file(file_path)
    
    # Log summary results
    if 'error' in results:
        logger.error(f"Error analyzing {file_path}: {results['error']}")
        return results
    
    logger.info(f"Semantic alignment for {file_path}:")
    logger.info(f"  Average alignment score: {results['average_alignment']:.2f}")
    logger.info(f"  Aligned items: {results['alignment_scores_count'] - results['misaligned_count']}/{results['alignment_scores_count']}")
    
    if results['misaligned_count'] > 0:
        logger.warning(f"  Found {results['misaligned_count']} potentially misaligned docstrings")
        
        # Report specific misalignments
        misaligned = []
        
        # Check module
        if results['has_docstring'] and not results['is_aligned']:
            logger.warning(f"  - Module docstring may be misaligned (score: {results['overall_similarity']:.2f})")
            misaligned.append(('module', results['overall_similarity']))
        
        # Check classes
        for cls in results['classes']:
            if cls['has_docstring'] and not cls['is_aligned']:
                logger.warning(f"  - Class {cls['name']} docstring may be misaligned (score: {cls['overall_similarity']:.2f})")
                misaligned.append((f"class {cls['name']}", cls['overall_similarity']))
            
            # Check methods
            for method in cls['methods']:
                if method['has_docstring'] and not method['is_aligned']:
                    logger.warning(f"  - Method {cls['name']}.{method['name']} docstring may be misaligned (score: {method['overall_similarity']:.2f})")
                    misaligned.append((f"method {cls['name']}.{method['name']}", method['overall_similarity']))
        
        # Check functions
        for func in results['functions']:
            if func['has_docstring'] and not func['is_aligned']:
                logger.warning(f"  - Function {func['name']} docstring may be misaligned (score: {func['overall_similarity']:.2f})")
                misaligned.append((f"function {func['name']}", func['overall_similarity']))
        
        # Sort by alignment score (worst first)
        misaligned.sort(key=lambda x: x[1])
        
        logger.warning("  Sorted misalignments (worst first):")
        for item, score in misaligned:
            logger.warning(f"  - {item}: {score:.2f}")
    
    return results


def process_directory(directory: str, threshold: float = 0.5, recursive: bool = False) -> List[Dict[str, Any]]:
    """Process all Python files in a directory for semantic alignment.
    
    Args:
        directory: Path to the directory
        threshold: Similarity threshold for alignment
        recursive: Whether to recursively process subdirectories
        
    Returns:
        List of analysis results for each file
    """
    results = []
    
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_results = process_file(file_path, threshold)
                    results.append(file_results)
    else:
        for file in os.listdir(directory):
            if file.endswith('.py'):
                file_path = os.path.join(directory, file)
                file_results = process_file(file_path, threshold)
                results.append(file_results)
    
    return results


def calculate_project_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate project-wide alignment metrics.
    
    Args:
        results: List of file analysis results
        
    Returns:
        Project-wide metrics as a dictionary
    """
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return {
            'average_alignment': 0.0,
            'total_docstrings': 0,
            'misaligned_docstrings': 0,
            'alignment_rate': 0.0,
            'files_analyzed': len(results),
            'files_with_errors': len(results) - len(valid_results)
        }
    
    # Calculate overall statistics
    total_scores = []
    total_docstrings = 0
    misaligned_docstrings = 0
    
    for file_result in valid_results:
        total_docstrings += file_result['alignment_scores_count']
        misaligned_docstrings += file_result['misaligned_count']
        
        # Collect all individual scores for a true average
        if file_result['has_docstring']:
            total_scores.append(file_result['overall_similarity'])
        
        for cls in file_result['classes']:
            if cls['has_docstring']:
                total_scores.append(cls['overall_similarity'])
            
            for method in cls['methods']:
                if method['has_docstring']:
                    total_scores.append(method['overall_similarity'])
        
        for func in file_result['functions']:
            if func['has_docstring']:
                total_scores.append(func['overall_similarity'])
    
    avg_alignment = sum(total_scores) / len(total_scores) if total_scores else 0.0
    alignment_rate = (total_docstrings - misaligned_docstrings) / total_docstrings if total_docstrings else 0.0
    
    return {
        'average_alignment': avg_alignment,
        'total_docstrings': total_docstrings,
        'misaligned_docstrings': misaligned_docstrings,
        'alignment_rate': alignment_rate * 100,  # as percentage
        'files_analyzed': len(results),
        'files_with_errors': len(results) - len(valid_results)
    }


def generate_report(results: List[Dict[str, Any]], metrics: Dict[str, Any], output_file: str) -> None:
    """Generate a detailed report of semantic alignment results.
    
    Args:
        results: List of file analysis results
        metrics: Project-wide metrics
        output_file: Path to the output JSON file
    """
    report = {
        'project_metrics': metrics,
        'file_results': results,
        'timestamp': import_time_module().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {output_file}")


def import_time_module():
    """Import time module dynamically to avoid issues with static imports."""
    import datetime
    return datetime.datetime.now()


def main():
    """Run the semantic alignment validator."""
    parser = argparse.ArgumentParser(description="Validate semantic alignment between code and docstrings")
    parser.add_argument(
        "path",
        help="Path to a Python file or directory"
    )
    parser.add_argument(
        "--recursive", 
        "-r", 
        action="store_true",
        help="Recursively process directories"
    )
    parser.add_argument(
        "--threshold", 
        "-t", 
        type=float,
        default=0.5,
        help="Similarity threshold for alignment (default: 0.5)"
    )
    parser.add_argument(
        "--report", 
        "-o", 
        type=str,
        help="Generate a JSON report and save it to the specified file"
    )
    args = parser.parse_args()
    
    if not EMBEDDINGS_AVAILABLE:
        logger.warning(
            "Sentence Transformers package not found. Using fallback similarity method. "
            "For better results, install sentence-transformers: pip install sentence-transformers"
        )
    
    path = args.path
    if not os.path.exists(path):
        logger.error(f"Path {path} does not exist")
        sys.exit(1)
    
    if os.path.isfile(path):
        if not path.endswith('.py'):
            logger.error(f"File {path} is not a Python file")
            sys.exit(1)
        results = [process_file(path, args.threshold)]
    elif os.path.isdir(path):
        results = process_directory(path, args.threshold, args.recursive)
    
    # Calculate project-wide metrics
    metrics = calculate_project_metrics(results)
    
    # Print overall summary
    logger.info("\nProject-Wide Semantic Alignment Metrics:")
    logger.info(f"  Files analyzed: {metrics['files_analyzed']}")
    logger.info(f"  Files with errors: {metrics['files_with_errors']}")
    logger.info(f"  Total docstrings: {metrics['total_docstrings']}")
    logger.info(f"  Aligned docstrings: {metrics['total_docstrings'] - metrics['misaligned_docstrings']}")
    logger.info(f"  Misaligned docstrings: {metrics['misaligned_docstrings']}")
    logger.info(f"  Average alignment score: {metrics['average_alignment']:.2f}")
    logger.info(f"  Overall alignment rate: {metrics['alignment_rate']:.2f}%")
    
    if args.report:
        generate_report(results, metrics, args.report)
    
    # Exit with non-zero status if alignment rate is below threshold
    if metrics['alignment_rate'] < args.threshold * 100:
        logger.warning(f"Alignment rate {metrics['alignment_rate']:.2f}% is below threshold {args.threshold * 100:.2f}%")
        sys.exit(1)


if __name__ == "__main__":
    main() 