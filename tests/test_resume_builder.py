"""Test suite for the ResumeBuilder class."""

import pytest
import torch
from transformers import Trainer, TrainingArguments
from app.models.ml.prediction.resume_builder import ResumeBuilder
import logging
import os
import json
from unittest.mock import Mock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def resume_builder():
    """Create a ResumeBuilder instance for testing."""
    return ResumeBuilder(model_name="t5-small")  # Use smaller model for testing

@pytest.fixture
def sample_job_experience():
    """Create sample job experience data for testing."""
    return {
        "work_experience": [
            {
                "title": "Software Engineer",
                "company": "Tech Corp",
                "start_date": "2020-01",
                "end_date": "2023-12",
                "description": "Led development of ML-powered features..."
            }
        ],
        "education": [
            {
                "degree": "BS Computer Science",
                "institution": "University of Technology",
                "start_date": "2016-09",
                "end_date": "2020-05"
            }
        ],
        "skills": ["Python", "PyTorch", "Machine Learning"]
    }

@pytest.fixture
def sample_resume_data():
    """Create sample resume data for testing."""
    return {
        "personal_info": {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "123-456-7890",
            "location": "San Francisco, CA"
        },
        "summary": "Experienced software engineer with expertise in ML...",
        "work_experience": [
            {
                "title": "Software Engineer",
                "company": "Tech Corp",
                "start_date": "2020-01",
                "end_date": "2023-12",
                "description": "Led development of ML-powered features..."
            }
        ],
        "education": [
            {
                "degree": "BS Computer Science",
                "institution": "University of Technology",
                "start_date": "2016-09",
                "end_date": "2020-05"
            }
        ],
        "skills": ["Python", "PyTorch", "Machine Learning"]
    }

def test_resume_builder_initialization(resume_builder):
    """Test ResumeBuilder initialization."""
    assert resume_builder.model_name == "t5-small"
    assert resume_builder.device in ["cuda", "cpu"]
    assert resume_builder.version == "2.1.0"
    assert resume_builder.tokenizer is not None
    assert resume_builder.model is not None
    assert resume_builder.nlp is not None
    assert resume_builder.tfidf is not None
    assert resume_builder.sentiment_analyzer is not None
    assert resume_builder.ner is not None

def test_prepare_input_text(resume_builder, sample_job_experience):
    """Test input text preparation."""
    input_text = resume_builder._prepare_input_text(sample_job_experience)
    assert isinstance(input_text, str)
    assert "Work Experience:" in input_text
    assert "Education:" in input_text
    assert "Skills:" in input_text
    assert "Software Engineer" in input_text
    assert "Python" in input_text

def test_prepare_target_text(resume_builder, sample_resume_data):
    """Test target text preparation."""
    target_text = resume_builder._prepare_target_text(sample_resume_data)
    assert isinstance(target_text, str)
    assert "Personal Information:" in target_text
    assert "Professional Summary:" in target_text
    assert "Professional Experience:" in target_text
    assert "Education:" in target_text
    assert "Skills:" in target_text

def test_parse_generated_text(resume_builder):
    """Test parsing of generated resume text."""
    sample_text = """
    Personal Information:
    name: John Doe
    email: john@example.com

    Professional Summary:
    Experienced software engineer...

    Professional Experience:
    Software Engineer at Tech Corp
    2020-01 - 2023-12
    Led development of ML features...

    Education:
    BS Computer Science
    University of Technology
    2016-09 - 2020-05

    Skills:
    Python, PyTorch, Machine Learning
    """
    
    parsed_data = resume_builder._parse_generated_text(sample_text)
    assert isinstance(parsed_data, dict)
    assert "personal_info" in parsed_data
    assert "summary" in parsed_data
    assert "work_experience" in parsed_data
    assert "education" in parsed_data
    assert "skills" in parsed_data

def test_extract_keywords(resume_builder):
    """Test keyword extraction."""
    text = "Led development of ML-powered features using Python and PyTorch at Tech Corp in San Francisco."
    keywords = resume_builder._extract_keywords(text)
    assert isinstance(keywords, list)
    assert "Python" in keywords
    assert "PyTorch" in keywords
    assert "Tech Corp" in keywords
    assert "San Francisco" in keywords

def test_calculate_keyword_match(resume_builder):
    """Test keyword match calculation."""
    resume_text = "Experienced in Python, PyTorch, and Machine Learning"
    job_description = "Looking for Python and PyTorch expertise"
    match_score = resume_builder._calculate_keyword_match(resume_text, job_description)
    assert isinstance(match_score, float)
    assert 0 <= match_score <= 1

def test_analyze_format(resume_builder, sample_resume_data):
    """Test format analysis."""
    score, suggestions = resume_builder._analyze_format(sample_resume_data)
    assert isinstance(score, float)
    assert isinstance(suggestions, list)
    assert 0 <= score <= 1

def test_analyze_content(resume_builder, sample_resume_data):
    """Test content analysis."""
    score, suggestions = resume_builder._analyze_content(sample_resume_data)
    assert isinstance(score, float)
    assert isinstance(suggestions, list)
    assert 0 <= score <= 1

def test_optimize_for_ats(resume_builder, sample_resume_data):
    """Test ATS optimization."""
    job_description = "Looking for Python and PyTorch expertise with ML experience"
    optimized_data, result = resume_builder.optimize_for_ats(
        sample_resume_data,
        job_description
    )
    assert isinstance(optimized_data, dict)
    assert isinstance(result.keyword_match_score, float)
    assert isinstance(result.format_score, float)
    assert isinstance(result.content_score, float)
    assert isinstance(result.suggestions, list)
    assert isinstance(result.missing_keywords, list)
    assert isinstance(result.keyword_density, dict)

def test_analyze_career_path(resume_builder, sample_resume_data):
    """Test career path analysis."""
    with patch('builtins.open') as mock_open:
        # Mock salary and transition data
        mock_open.side_effect = [
            Mock(read=lambda: json.dumps({
                "Software Engineer": {
                    "skills": ["Python", "PyTorch", "ML"],
                    "salary_range": {"min": 100000, "max": 150000},
                    "growth_path": ["Senior Engineer", "Tech Lead"]
                }
            })),
            Mock(read=lambda: json.dumps([
                {
                    "source_roles": ["Software Engineer"],
                    "target_industry": "AI",
                    "matching_roles": ["ML Engineer"],
                    "required_skills": ["Python", "PyTorch"]
                }
            ]))
        ]
        
        insight = resume_builder.analyze_career_path(sample_resume_data)
        assert insight is not None
        assert isinstance(insight.recommended_roles, list)
        assert isinstance(insight.skills_gap, list)
        assert isinstance(insight.salary_projection, dict)
        assert isinstance(insight.industry_transitions, list)
        assert isinstance(insight.growth_opportunities, list)

def test_generate_resume(resume_builder, sample_job_experience):
    """Test resume generation."""
    with patch.object(resume_builder.model, 'generate') as mock_generate:
        # Mock model generation
        mock_generate.return_value = torch.tensor([[1, 2, 3]])
        with patch.object(resume_builder.tokenizer, 'decode') as mock_decode:
            mock_decode.return_value = "Generated resume text"
            
            resume_text = resume_builder.generate_resume(
                sample_job_experience,
                max_length=100,
                num_beams=2,
                temperature=0.7
            )
            assert isinstance(resume_text, str)
            assert "Generated resume text" in resume_text

def test_format_resume(resume_builder, sample_resume_data):
    """Test resume formatting in different formats."""
    # Test text format
    text_format = resume_builder.format_resume(sample_resume_data, format='text')
    assert isinstance(text_format, str)
    
    # Test HTML format
    html_format = resume_builder.format_resume(sample_resume_data, format='html')
    assert isinstance(html_format, str)
    assert "<!DOCTYPE html>" in html_format
    assert "<html>" in html_format
    
    # Test Markdown format
    md_format = resume_builder.format_resume(sample_resume_data, format='markdown')
    assert isinstance(md_format, str)
    assert "# Personal Information" in md_format

def test_custom_tensorboard_callback():
    """Test CustomTensorBoardCallback implementation."""
    from app.models.ml.prediction.resume_builder import CustomTensorBoardCallback
    
    # Create a mock writer
    mock_writer = Mock()
    
    # Initialize callback
    callback = CustomTensorBoardCallback(writer=mock_writer)
    
    # Test callback methods
    args = Mock()
    state = Mock()
    control = Mock()
    logs = {"loss": 0.5}
    
    # Test on_train_begin
    callback.on_train_begin(args, state, control)
    mock_writer.add_text.assert_called_once()
    
    # Test on_log
    callback.on_log(args, state, control, logs=logs)
    mock_writer.add_scalar.assert_called_once()
    
    # Test on_train_end
    callback.on_train_end(args, state, control)
    mock_writer.close.assert_called_once()

def test_training_error_handling(resume_builder):
    """Test error handling during training."""
    with pytest.raises(Exception):
        # Test with invalid dataset
        resume_builder.train(None, None, None)
    
    with pytest.raises(Exception):
        # Test with invalid model path
        resume_builder.load_model("invalid/path")

def test_memory_management(resume_builder):
    """Test memory management during large dataset processing."""
    # Create a large dataset
    large_dataset = [{"input_ids": torch.randn(512)} for _ in range(10000)]
    
    # Test memory usage during processing
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process dataset in batches
    batch_size = 1000
    for i in range(0, len(large_dataset), batch_size):
        batch = large_dataset[i:i + batch_size]
        # Process batch
        _ = [item["input_ids"] for item in batch]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Assert memory increase is reasonable (less than 2GB)
        assert memory_increase < 2048, f"Memory increase too high: {memory_increase}MB"

def test_gpu_utilization(resume_builder):
    """Test GPU utilization and memory management."""
    if torch.cuda.is_available():
        # Get initial GPU memory usage
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Create test data
        test_data = torch.randn(1000, 512).cuda()
        
        # Process data
        _ = test_data.mean()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Check final memory usage
        final_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Assert memory was properly cleared
        assert abs(final_memory - initial_memory) < 100, "GPU memory not properly cleared" 