"""Resume Builder Model Implementation."""

from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from dataclasses import dataclass
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import Counter
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class ResumeSection:
    """Data class for resume sections."""
    title: str
    content: str
    order: int

@dataclass
class ATSOptimizationResult:
    """Data class for ATS optimization results."""
    keyword_match_score: float
    format_score: float
    content_score: float
    suggestions: List[str]
    missing_keywords: List[str]
    keyword_density: Dict[str, float]

@dataclass
class CareerInsight:
    """Data class for career path insights."""
    recommended_roles: List[str]
    skills_gap: List[str]
    salary_projection: Dict[str, float]
    industry_transitions: List[Dict[str, Any]]
    growth_opportunities: List[str]

class ResumeBuilder:
    """Enhanced resume builder with AI/ML capabilities for 2025."""
    
    def __init__(
        self,
        model_name: str = "t5-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the resume builder with advanced NLP components."""
        self.model_name = model_name
        self.device = device
        self.version = "2.1.0"  # Updated version
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize TF-IDF vectorizer for keyword matching
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            analyzer='word',
            min_df=2,
            max_df=0.7
        )
        
        # Initialize sentiment analysis for content optimization
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if device == "cuda" else -1
        )
        
        # Initialize NER for skill extraction
        self.ner = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            device=0 if device == "cuda" else -1
        )
        
        # Initialize section templates
        self.section_templates = {
            'personal_info': {
                'title': 'Personal Information',
                'order': 1,
                'required_fields': ['name', 'email', 'phone', 'location']
            },
            'summary': {
                'title': 'Professional Summary',
                'order': 2,
                'max_length': 150
            },
            'work_experience': {
                'title': 'Professional Experience',
                'order': 3,
                'required_fields': ['title', 'company', 'start_date', 'end_date', 'description']
            },
            'education': {
                'title': 'Education',
                'order': 4,
                'required_fields': ['degree', 'institution', 'start_date', 'end_date']
            },
            'skills': {
                'title': 'Skills',
                'order': 5,
                'max_items': 15
            }
        }
        
        # Initialize ATS optimization parameters
        self.ats_params = {
            'min_keyword_match': 0.7,
            'max_keyword_density': 0.15,
            'required_sections': ['personal_info', 'summary', 'work_experience', 'education', 'skills'],
            'format_requirements': {
                'max_pages': 2,
                'font_size': 11,
                'margin_size': 1,
                'line_spacing': 1.15
            }
        }
        
        # Initialize career path analysis parameters
        self.career_params = {
            'min_confidence': 0.8,
            'max_recommendations': 5,
            'salary_data_path': 'data/salary_projections.json',
            'industry_transitions_path': 'data/industry_transitions.json'
        }
        
        logger.info(f"Initialized ResumeBuilder v{self.version} with advanced AI/ML capabilities")
    
    def load_model(self, model_path: str) -> bool:
        """Load a pre-trained model from the specified path."""
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords using NLP techniques."""
        doc = self.nlp(text)
        keywords = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'DATE']:
                keywords.append(ent.text)
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Avoid long phrases
                keywords.append(chunk.text)
        
        # Extract technical terms
        technical_terms = re.findall(r'\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\b', text)
        keywords.extend(technical_terms)
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_keyword_match(self, resume_text: str, job_description: str) -> float:
        """Calculate keyword match score between resume and job description."""
        # Extract keywords from both texts
        resume_keywords = self._extract_keywords(resume_text)
        job_keywords = self._extract_keywords(job_description)
        
        # Convert to TF-IDF vectors
        tfidf_matrix = self.tfidf.fit_transform([resume_text, job_description])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    
    def _analyze_format(self, resume_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze resume format and return score and suggestions."""
        score = 1.0
        suggestions = []
        
        # Check required sections
        for section in self.ats_params['required_sections']:
            if section not in resume_data:
                score -= 0.2
                suggestions.append(f"Missing required section: {section}")
        
        # Check section content
        for section, data in resume_data.items():
            if section in self.section_templates:
                template = self.section_templates[section]
                
                # Check required fields
                if 'required_fields' in template:
                    for field in template['required_fields']:
                        if field not in data:
                            score -= 0.1
                            suggestions.append(f"Missing required field '{field}' in {section}")
                
                # Check length limits
                if 'max_length' in template and isinstance(data, str):
                    if len(data) > template['max_length']:
                        score -= 0.1
                        suggestions.append(f"{section} exceeds maximum length of {template['max_length']} characters")
        
        return max(0.0, score), suggestions
    
    def _analyze_content(self, resume_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze resume content quality and return score and suggestions."""
        score = 1.0
        suggestions = []
        
        # Analyze work experience descriptions
        if 'work_experience' in resume_data:
            for exp in resume_data['work_experience']:
                if 'description' in exp:
                    # Check for action verbs
                    if not any(verb in exp['description'].lower() for verb in ['led', 'managed', 'developed', 'created', 'improved']):
                        score -= 0.1
                        suggestions.append("Use more action verbs in work experience descriptions")
                    
                    # Check for quantifiable achievements
                    if not re.search(r'\d+%|\d+\s*%|\d+\s*M|\d+\s*K', exp['description']):
                        score -= 0.1
                        suggestions.append("Include quantifiable achievements in work experience")
        
        # Analyze skills section
        if 'skills' in resume_data:
            skills = resume_data['skills']
            if len(skills) > self.section_templates['skills']['max_items']:
                score -= 0.1
                suggestions.append(f"Reduce skills list to top {self.section_templates['skills']['max_items']} most relevant skills")
        
        return max(0.0, score), suggestions
    
    def _generate_optimization_suggestions(
        self,
        keyword_match: float,
        format_score: float,
        content_score: float,
        missing_keywords: List[str],
        format_suggestions: List[str],
        content_suggestions: List[str]
    ) -> List[str]:
        """Generate comprehensive optimization suggestions."""
        suggestions = []
        
        # Keyword match suggestions
        if keyword_match < self.ats_params['min_keyword_match']:
            suggestions.append(f"Add more relevant keywords: {', '.join(missing_keywords)}")
        
        # Format suggestions
        if format_score < 0.8:
            suggestions.extend(format_suggestions)
        
        # Content suggestions
        if content_score < 0.8:
            suggestions.extend(content_suggestions)
        
        return suggestions
    
    def _optimize_content(self, resume_data: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """Optimize resume content based on job description."""
        optimized_data = resume_data.copy()
        
        # Extract keywords from job description
        job_keywords = self._extract_keywords(job_description)
        
        # Optimize work experience descriptions
        if 'work_experience' in optimized_data:
            for exp in optimized_data['work_experience']:
                if 'description' in exp:
                    # Add relevant keywords naturally
                    for keyword in job_keywords:
                        if keyword.lower() not in exp['description'].lower():
                            exp['description'] = f"{exp['description']} {keyword}"
        
        # Optimize skills section
        if 'skills' in optimized_data:
            current_skills = set(optimized_data['skills'])
            missing_skills = [skill for skill in job_keywords if skill not in current_skills]
            optimized_data['skills'].extend(missing_skills[:3])  # Add top 3 missing skills
        
        return optimized_data
    
    def optimize_for_ats(
        self,
        resume_data: Dict[str, Any],
        job_description: str,
        format: str = 'text'
    ) -> Tuple[Dict[str, Any], ATSOptimizationResult]:
        """Optimize resume for ATS compatibility."""
        # Convert resume data to text for analysis
        resume_text = self._prepare_target_text(resume_data)
        
        # Calculate keyword match
        keyword_match = self._calculate_keyword_match(resume_text, job_description)
        missing_keywords = [
            keyword for keyword in self._extract_keywords(job_description)
            if keyword.lower() not in resume_text.lower()
        ]
        
        # Analyze format
        format_score, format_suggestions = self._analyze_format(resume_data)
        
        # Analyze content
        content_score, content_suggestions = self._analyze_content(resume_data)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(
            keyword_match,
            format_score,
            content_score,
            missing_keywords,
            format_suggestions,
            content_suggestions
        )
        
        # Optimize content
        optimized_data = self._optimize_content(resume_data, job_description)
        
        # Calculate keyword density
        keyword_density = {}
        for keyword in self._extract_keywords(job_description):
            count = resume_text.lower().count(keyword.lower())
            density = count / len(resume_text.split())
            keyword_density[keyword] = density
        
        # Create optimization result
        result = ATSOptimizationResult(
            keyword_match_score=keyword_match,
            format_score=format_score,
            content_score=content_score,
            suggestions=suggestions,
            missing_keywords=missing_keywords,
            keyword_density=keyword_density
        )
        
        return optimized_data, result
    
    def analyze_career_path(
        self,
        resume_data: Dict[str, Any],
        target_role: Optional[str] = None
    ) -> CareerInsight:
        """Analyze career path and provide insights."""
        # Extract skills and experience
        skills = resume_data.get('skills', [])
        experience = resume_data.get('work_experience', [])
        
        # Load salary and industry transition data
        try:
            with open(self.career_params['salary_data_path']) as f:
                salary_data = json.load(f)
            with open(self.career_params['industry_transitions_path']) as f:
                transition_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading career data: {str(e)}")
            return None
        
        # Analyze current role and skills
        current_role = experience[-1]['title'] if experience else None
        
        # Find recommended roles based on skills and experience
        recommended_roles = []
        for role, requirements in salary_data.items():
            skill_match = sum(1 for skill in skills if skill in requirements['skills'])
            if skill_match / len(requirements['skills']) >= 0.7:
                recommended_roles.append(role)
        
        # Identify skills gap
        skills_gap = []
        if target_role and target_role in salary_data:
            required_skills = salary_data[target_role]['skills']
            skills_gap = [skill for skill in required_skills if skill not in skills]
        
        # Calculate salary projections
        salary_projection = {}
        for role in recommended_roles[:3]:
            if role in salary_data:
                salary_projection[role] = salary_data[role]['salary_range']
        
        # Find industry transition opportunities
        industry_transitions = []
        if current_role:
            for transition in transition_data:
                if current_role in transition['source_roles']:
                    industry_transitions.append({
                        'target_industry': transition['target_industry'],
                        'matching_roles': transition['matching_roles'],
                        'required_skills': transition['required_skills']
                    })
        
        # Identify growth opportunities
        growth_opportunities = []
        for role in recommended_roles:
            if role in salary_data:
                growth_opportunities.extend(salary_data[role]['growth_path'])
        
        return CareerInsight(
            recommended_roles=recommended_roles[:self.career_params['max_recommendations']],
            skills_gap=skills_gap,
            salary_projection=salary_projection,
            industry_transitions=industry_transitions,
            growth_opportunities=growth_opportunities
        )
    
    def generate_resume(
        self,
        job_experience: Dict[str, Any],
        output_file: Optional[str] = None,
        max_length: int = 1024,
        num_beams: int = 4,
        temperature: float = 0.7
    ) -> str:
        """Generate a resume from job experience with advanced optimizations."""
        try:
            # Prepare input text
            input_text = self._prepare_input_text(job_experience)
            input_encoding = self.tokenizer(
                input_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate resume with advanced parameters
            outputs = self.model.generate(
                input_encoding['input_ids'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            # Decode output
            resume_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse generated text into structured data
            resume_data = self._parse_generated_text(resume_text)
            
            # Save to file if specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(resume_data, f, indent=2)
                logger.info(f"Resume saved to {output_file}")
            
            return resume_text
            
        except Exception as e:
            logger.error(f"Error generating resume: {str(e)}")
            return None
    
    def _prepare_input_text(self, job_experience: Dict[str, Any]) -> str:
        """Prepare input text from job experience."""
        sections = []
        
        # Add work experience
        if 'work_experience' in job_experience:
            sections.append("Work Experience:")
            for exp in job_experience['work_experience']:
                sections.append(f"- {exp['title']} at {exp['company']}")
                sections.append(f"  Duration: {exp['start_date']} to {exp['end_date']}")
                sections.append(f"  Description: {exp['description']}")
        
        # Add education
        if 'education' in job_experience:
            sections.append("\nEducation:")
            for edu in job_experience['education']:
                sections.append(f"- {edu['degree']} from {edu['institution']}")
                sections.append(f"  Duration: {edu['start_date']} to {edu['end_date']}")
        
        # Add skills
        if 'skills' in job_experience:
            sections.append("\nSkills:")
            sections.append(", ".join(job_experience['skills']))
        
        return "\n".join(sections)
    
    def _prepare_target_text(self, resume_data: Dict[str, Any]) -> str:
        """Prepare target text (resume) from structured data."""
        sections = []
        
        # Add personal info
        if 'personal_info' in resume_data:
            sections.append("Personal Information:")
            for key, value in resume_data['personal_info'].items():
                sections.append(f"{key}: {value}")
        
        # Add professional summary
        if 'summary' in resume_data:
            sections.append("\nProfessional Summary:")
            sections.append(resume_data['summary'])
        
        # Add work experience
        if 'work_experience' in resume_data:
            sections.append("\nProfessional Experience:")
            for exp in resume_data['work_experience']:
                sections.append(f"{exp['title']} at {exp['company']}")
                sections.append(f"{exp['start_date']} - {exp['end_date']}")
                sections.append(exp['description'])
        
        # Add education
        if 'education' in resume_data:
            sections.append("\nEducation:")
            for edu in resume_data['education']:
                sections.append(f"{edu['degree']}")
                sections.append(f"{edu['institution']}")
                sections.append(f"{edu['start_date']} - {edu['end_date']}")
        
        # Add skills
        if 'skills' in resume_data:
            sections.append("\nSkills:")
            sections.append(", ".join(resume_data['skills']))
        
        return "\n".join(sections)
    
    def _parse_generated_text(self, text: str) -> Dict[str, Any]:
        """Parse generated resume text into structured data."""
        sections = text.split("\n\n")
        resume_data = {}
        
        current_section = None
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.split("\n")
            title = lines[0].strip(": ")
            
            # Identify section type
            if title.lower() == "personal information":
                current_section = "personal_info"
                resume_data[current_section] = {}
                for line in lines[1:]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        resume_data[current_section][key.strip()] = value.strip()
            
            elif title.lower() == "professional summary":
                current_section = "summary"
                resume_data[current_section] = " ".join(lines[1:]).strip()
            
            elif title.lower() == "professional experience":
                current_section = "work_experience"
                resume_data[current_section] = []
                current_exp = {}
                
                for line in lines[1:]:
                    if line.startswith("-"):
                        if current_exp:
                            resume_data[current_section].append(current_exp)
                        current_exp = {"description": line[2:].strip()}
                    elif " at " in line:
                        title, company = line.split(" at ")
                        current_exp["title"] = title.strip()
                        current_exp["company"] = company.strip()
                    elif " - " in line:
                        start_date, end_date = line.split(" - ")
                        current_exp["start_date"] = start_date.strip()
                        current_exp["end_date"] = end_date.strip()
                
                if current_exp:
                    resume_data[current_section].append(current_exp)
            
            elif title.lower() == "education":
                current_section = "education"
                resume_data[current_section] = []
                current_edu = {}
                
                for line in lines[1:]:
                    if line.startswith("-"):
                        if current_edu:
                            resume_data[current_section].append(current_edu)
                        current_edu = {"degree": line[2:].strip()}
                    elif " from " in line:
                        degree, institution = line.split(" from ")
                        current_edu["degree"] = degree.strip()
                        current_edu["institution"] = institution.strip()
                    elif " - " in line:
                        start_date, end_date = line.split(" - ")
                        current_edu["start_date"] = start_date.strip()
                        current_edu["end_date"] = end_date.strip()
                
                if current_edu:
                    resume_data[current_section].append(current_edu)
            
            elif title.lower() == "skills":
                current_section = "skills"
                skills_text = " ".join(lines[1:])
                resume_data[current_section] = [skill.strip() for skill in skills_text.split(",")]
        
        return resume_data
    
    def format_resume(
        self,
        resume_data: Dict[str, Any],
        format: str = 'text',
        output_file: Optional[str] = None
    ) -> str:
        """Format resume in specified format (text, HTML, or Markdown)."""
        if format == 'text':
            return self._format_text(resume_data)
        elif format == 'html':
            return self._format_html(resume_data)
        elif format == 'markdown':
            return self._format_markdown(resume_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_text(self, resume_data: Dict[str, Any]) -> str:
        """Format resume as plain text."""
        return self._prepare_target_text(resume_data)
    
    def _format_html(self, resume_data: Dict[str, Any]) -> str:
        """Format resume as HTML."""
        html = ['<!DOCTYPE html>', '<html>', '<head>', '<style>',
                'body { font-family: Arial, sans-serif; line-height: 1.6; }',
                'h1, h2 { color: #2c3e50; }',
                '.section { margin-bottom: 20px; }',
                '.experience-item { margin-bottom: 15px; }',
                '</style>', '</head>', '<body>']
        
        # Add personal info
        if 'personal_info' in resume_data:
            html.append('<div class="section">')
            html.append('<h1>Personal Information</h1>')
            for key, value in resume_data['personal_info'].items():
                html.append(f'<p><strong>{key}:</strong> {value}</p>')
            html.append('</div>')
        
        # Add summary
        if 'summary' in resume_data:
            html.append('<div class="section">')
            html.append('<h2>Professional Summary</h2>')
            html.append(f'<p>{resume_data["summary"]}</p>')
            html.append('</div>')
        
        # Add work experience
        if 'work_experience' in resume_data:
            html.append('<div class="section">')
            html.append('<h2>Professional Experience</h2>')
            for exp in resume_data['work_experience']:
                html.append('<div class="experience-item">')
                html.append(f'<h3>{exp["title"]} at {exp["company"]}</h3>')
                html.append(f'<p>{exp["start_date"]} - {exp["end_date"]}</p>')
                html.append(f'<p>{exp["description"]}</p>')
                html.append('</div>')
            html.append('</div>')
        
        # Add education
        if 'education' in resume_data:
            html.append('<div class="section">')
            html.append('<h2>Education</h2>')
            for edu in resume_data['education']:
                html.append('<div class="experience-item">')
                html.append(f'<h3>{edu["degree"]}</h3>')
                html.append(f'<p>{edu["institution"]}</p>')
                html.append(f'<p>{edu["start_date"]} - {edu["end_date"]}</p>')
                html.append('</div>')
            html.append('</div>')
        
        # Add skills
        if 'skills' in resume_data:
            html.append('<div class="section">')
            html.append('<h2>Skills</h2>')
            html.append('<p>' + ', '.join(resume_data['skills']) + '</p>')
            html.append('</div>')
        
        html.extend(['</body>', '</html>'])
        return '\n'.join(html)
    
    def _format_markdown(self, resume_data: Dict[str, Any]) -> str:
        """Format resume as Markdown."""
        md = []
        
        # Add personal info
        if 'personal_info' in resume_data:
            md.append('# Personal Information\n')
            for key, value in resume_data['personal_info'].items():
                md.append(f'**{key}:** {value}')
            md.append('')
        
        # Add summary
        if 'summary' in resume_data:
            md.append('## Professional Summary\n')
            md.append(resume_data['summary'])
            md.append('')
        
        # Add work experience
        if 'work_experience' in resume_data:
            md.append('## Professional Experience\n')
            for exp in resume_data['work_experience']:
                md.append(f'### {exp["title"]} at {exp["company"]}')
                md.append(f'*{exp["start_date"]} - {exp["end_date"]}*')
                md.append(exp['description'])
                md.append('')
        
        # Add education
        if 'education' in resume_data:
            md.append('## Education\n')
            for edu in resume_data['education']:
                md.append(f'### {edu["degree"]}')
                md.append(f'*{edu["institution"]}*')
                md.append(f'*{edu["start_date"]} - {edu["end_date"]}*')
                md.append('')
        
        # Add skills
        if 'skills' in resume_data:
            md.append('## Skills\n')
            md.append(', '.join(resume_data['skills']))
        
        return '\n'.join(md) 
        return keyword_analysis 