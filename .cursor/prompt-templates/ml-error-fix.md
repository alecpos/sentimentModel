### 2. Error Resolution Template

Save as `.cursor/prompt-templates/ml-error-fix.md`:

```markdown
# ML Error Resolution Request

## Context
- Project: Python ML backend with PyTorch/Scikit-learn
- File: ${Specify the file with the error}
- Error Type: ${Specify error type: runtime, training, inference, etc.}

## Current Code
```python
${Paste the problematic code here}
