### 4. Data Pipeline Template

Save as `.cursor/prompt-templates/data-pipeline.md`:

```markdown
# Data Pipeline Development Request

## Context
- Project: Python ML backend with PyTorch/Scikit-learn
- Pipeline Purpose: ${Describe the pipeline purpose}
- Input Data Sources: ${List data sources}
- Output Requirements: ${Describe expected outputs}

## Requirements
- Data Validation:
  - Schema validation
  - Quality checks
  - Outlier detection
- Transformations:
  - Feature engineering
  - Normalization/scaling
  - Encoding categorical variables
- Performance:
  - Parallel processing
  - Memory efficiency for large datasets
  - Proper resource management

## Pipeline Sketch
```python
# Conceptual pipeline structureˆ
${Provide a basic outline of the pipeline if available}
