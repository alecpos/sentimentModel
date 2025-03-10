# Cursor Rules Documentation

This directory contains customized rules and guidelines for the WITHIN Ad Score & Account Health Predictor project. These rules help ensure consistent, high-quality code and standardized approaches to ML development across the project.

## Purpose

The Cursor Rules establish project-specific standards for:

1. Machine learning model implementation
2. Data pipeline development
3. Testing methodologies
4. API integration patterns
5. Natural language processing
6. Cross-platform compatibility
7. Data visualization
8. Model monitoring
9. Ethical AI and privacy compliance

These rules are used by AI coding assistants to provide guidance aligned with project standards and best practices.

## Rule Types

### Global Rules

- **ai_rules.md**: Project-wide ML guidelines that apply to all interactions

### Component-Specific Rules (.mdc files)

Each `.mdc` file targets specific components or patterns in the codebase:

| Rule File | Purpose | Target Files |
|-----------|---------|--------------|
| `data_pipelines.mdc` | Standards for data ingestion and processing | `app/pipelines/**/*.py`, `app/etl/**/*.py` |
| `ml_models.mdc` | Standards for implementing ML models | `app/models/**/*.py`, `app/ml/**/*.py` |
| `api_integration.mdc` | Standards for API implementations | `app/api/**/*.py`, `app/routes/**/*.py` |
| `testing.mdc` | Standards for testing ML components | `tests/**/*.py` |
| `nlp_processing.mdc` | Standards for NLP components | `app/nlp/**/*.py`, `app/ml/text/**/*.py` |
| `cross_platform.mdc` | Standards for cross-platform compatibility | `app/platforms/**/*.py`, `app/integrations/**/*.py` |
| `visualization.mdc` | Standards for data visualization | `app/visualization/**/*.py`, `app/reports/**/*.py` |
| `model_monitoring.mdc` | Standards for model monitoring | `app/monitoring/**/*.py`, `app/ml/monitoring/**/*.py` |
| `ethics_privacy.mdc` | Standards for ethical ML and privacy | `app/**/*.py` |

## Rule Structure

Each rule file follows a consistent structure:

1. **Description**: Brief explanation of the rule's purpose
2. **Globs**: File patterns these rules apply to
3. **Component-Specific Standards**: Guidelines for that component type
4. **Examples**: Code examples demonstrating best practices

## Usage

These rules are automatically applied when working with AI coding assistants in the Cursor editor. The rules help the assistant:

1. Generate code that follows project standards
2. Provide suggestions aligned with project architecture
3. Maintain consistency across the codebase
4. Implement ML best practices specific to ad scoring and account health prediction

## Maintenance

The rules should be reviewed and updated:

- When adding new components or technologies
- When refining ML methodologies
- When addressing recurring issues or patterns
- Quarterly to reflect evolving ML practices

To update a rule, edit the corresponding `.mdc` file and ensure it follows the established format.

## Examples

Each rule file contains practical examples showing the implementation of best practices. These serve as reference implementations for developers and AI assistants.

For instance, the ML Models rule includes example code for a well-structured `AdScoreModel` class, while the Ethics & Privacy rule demonstrates fairness assessment implementation. 