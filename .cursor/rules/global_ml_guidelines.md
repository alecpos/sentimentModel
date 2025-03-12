# WITHIN ML PROJECT GUIDELINES

## 1. CODE QUALITY STANDARDS
   - Follow PEP 8 for Python code with strict type hints
   - Use comprehensive docstrings in Google docstring format
   - Implement error handling for data processing edge cases
   - Organize imports: standard library, third-party, local modules
   - Keep function complexity below 15 (cyclomatic complexity)

## 2. ML DEVELOPMENT PRINCIPLES
   - Prioritize model reproducibility through seed setting
   - Document all hyperparameters and architecture decisions
   - Include performance metrics in model documentation
   - Separate data processing from model implementation
   - Always include bias and fairness considerations
   - Use data validation before processing

## 3. PROJECT ARCHITECTURE
   - Maintain clear boundaries between system components
   - Follow modular design for pipelines, models, and APIs
   - Implement consistent interfaces between components
   - Use appropriate design patterns for each ML component
   - Track data quality metrics and alert on drift

## 4. DOCUMENTATION REQUIREMENTS
   - Document feature engineering clearly with rationale
   - Include model cards for all production models
   - Describe assumptions and limitations explicitly
   - Provide usage examples for all components
   - Document model version history and changes 