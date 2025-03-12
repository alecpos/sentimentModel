# Data Validation Components

This directory contains data validation components for the WITHIN ML Prediction System.

## Purpose

The validation system provides mechanisms for:
- Ensuring data quality and integrity before processing
- Validating inputs against predefined schemas
- Enforcing business rules and domain constraints
- Detecting and handling data anomalies
- Providing detailed error reporting for data issues

## Key Components

### Schema Validation

Components for validating data structure:
- JSON schema validation for API payloads
- DataFrame schema validation for tabular data
- Type checking and coercion
- Required field validation
- Format and pattern validation

### Business Rule Validation

Components for enforcing domain-specific rules:
- Cross-field validation rules
- Conditional validation logic
- Value range and constraint checking
- Relationship validation between entities
- Complex rule evaluation

### Data Quality Assessment

Components for measuring and ensuring data quality:
- Completeness checks (missing value detection)
- Consistency validation across data fields
- Accuracy checks against reference data
- Statistical distribution validation
- Outlier detection and handling

### Validation Pipeline

Components for managing the validation process:
- Validation rule chains and composability
- Error aggregation and reporting
- Validation context management
- Custom validator registration
- Partial validation capabilities

## Usage Example

```python
from app.core.validation import SchemaValidator, RuleValidator, ValidationPipeline
from app.core.validation.rules import RangeRule, RequiredFieldRule, FormatRule

# Create validators
schema_validator = SchemaValidator.from_json_schema("schemas/ad_data_schema.json")
rule_validator = RuleValidator([
    RequiredFieldRule(fields=["campaign_id", "creative_id", "platform"]),
    RangeRule(field="bid_amount", min_value=0.01, max_value=100.0),
    FormatRule(field="campaign_id", pattern=r"^[A-Z]{2}-\d{6}$")
])

# Create validation pipeline
validation_pipeline = ValidationPipeline(
    validators=[schema_validator, rule_validator],
    mode="strict"
)

# Validate data
validation_result = validation_pipeline.validate(input_data)

if validation_result.is_valid:
    print("Data is valid! Proceeding with processing.")
else:
    print(f"Validation failed with {len(validation_result.errors)} errors:")
    for error in validation_result.errors:
        print(f"- {error.field}: {error.message} (severity: {error.severity})")
```

## Integration Points

- **API Layer**: Validates incoming request payloads
- **ETL Pipeline**: Validates data during extraction and loading
- **ML Training**: Ensures training data meets quality requirements
- **Inference Service**: Validates prediction inputs before processing

## Dependencies

- JSON Schema libraries for schema validation
- Pandas validators for DataFrame validation
- Regular expression utilities for pattern validation
- Statistics packages for distribution validation
- Type annotation and validation libraries 