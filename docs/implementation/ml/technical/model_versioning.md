# Model Versioning Protocol

**IMPLEMENTATION STATUS: IMPLEMENTED**


## Overview

This document outlines the versioning protocol for machine learning models in the WITHIN platform. It provides guidelines for model version management, deployment, compatibility, and lifecycle management to ensure consistency and reliability across the ML ecosystem.

## Table of Contents

1. [Version Numbering](#version-numbering)
2. [Model Registry](#model-registry)
3. [Version Transitions](#version-transitions)
4. [Compatibility Management](#compatibility-management)
5. [Artifact Management](#artifact-management)
6. [Deployment Strategy](#deployment-strategy)
7. [Rollback Procedures](#rollback-procedures)
8. [Documentation Requirements](#documentation-requirements)
9. [Testing Protocol](#testing-protocol)
10. [Governance](#governance)

## Version Numbering

### Semantic Versioning

All ML models follow semantic versioning with format `MAJOR.MINOR.PATCH`:

- **MAJOR**: Incompatible API changes, significant architecture changes, or retraining with substantially different data
- **MINOR**: Functionality added in a backward-compatible manner, hyperparameter tuning, or feature additions
- **PATCH**: Backward-compatible bug fixes, minor optimizations, or threshold adjustments

### Version Qualifiers

Additional qualifiers may be appended to version numbers:

- `-alpha`: Internal testing (not for production)
- `-beta`: External testing with selected customers
- `-rc`: Release candidate undergoing final validation
- `-experiment`: Experimental variant (e.g., `2.1.0-experiment-attention`)

### Examples

- `1.0.0`: Initial production release
- `1.0.1`: Bug fix release
- `1.1.0`: Feature addition or enhancement
- `2.0.0`: Major architecture change
- `1.1.0-beta`: Beta version of feature release
- `2.0.0-experiment-quantized`: Experimental quantized variant of version 2.0.0

## Model Registry

### Central Registry

All models are registered in the central model registry located at `s3://within-ml-models/registry/` with metadata stored in DynamoDB.

### Registry Structure

```
model-registry/
├── ad-score-predictor/
│   ├── 1.0.0/
│   │   ├── model.pkl
│   │   ├── config.json
│   │   ├── metadata.json
│   │   └── metrics.json
│   ├── 1.1.0/
│   │   └── ...
│   └── latest -> 1.1.0/
├── account-health-predictor/
│   └── ...
└── ad-sentiment-analyzer/
    └── ...
```

### Metadata Format

Each model version includes a `metadata.json` file with the following information:

```json
{
  "model_name": "ad-score-predictor",
  "version": "1.1.0",
  "created_at": "2023-04-15T14:32:00Z",
  "created_by": "jane.doe@within.co",
  "framework": "pytorch",
  "framework_version": "1.11.0",
  "python_version": "3.9.10",
  "training_dataset": "ad_performance_q1_2023",
  "training_dataset_hash": "sha256:a1b2c3...",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 20,
    "architecture": "hybrid-transformer-xgboost"
  },
  "performance_metrics": {
    "validation_rmse": 5.82,
    "validation_r2": 0.78,
    "validation_mae": 4.21
  },
  "dependencies": [
    {"name": "feature_extractor", "version": "1.2.0"},
    {"name": "text_encoder", "version": "2.0.1"}
  ],
  "tags": ["production", "approved"],
  "approval_status": "approved",
  "approval_date": "2023-04-18T09:15:00Z",
  "approved_by": "ml-review-board",
  "deployment_environments": ["staging", "production"],
  "changelog": "Improved text representation with transformer encoder",
  "documentation_url": "https://docs.within.co/ml/models/ad-score-predictor/v1.1.0"
}
```

### Registry API

The model registry provides a Python API for registration and retrieval:

```python
from within.ml.registry import ModelRegistry

# Initialize registry
registry = ModelRegistry()

# Register a model
registry.register_model(
    model_name="ad-score-predictor",
    version="1.1.0",
    model_path="/path/to/model.pkl",
    metadata={...}
)

# Retrieve a model
model = registry.get_model(
    model_name="ad-score-predictor",
    version="1.1.0"  # or "latest" for the latest version
)

# List versions
versions = registry.list_versions("ad-score-predictor")

# Get metadata
metadata = registry.get_metadata(
    model_name="ad-score-predictor",
    version="1.1.0"
)
```

## Version Transitions

### Lifecycle States

Models progress through these lifecycle states:

1. **Development**: Initial development and internal testing
2. **Alpha**: Internal testing with synthetic or limited real data
3. **Beta**: Testing with selected customers or internal stakeholders
4. **RC (Release Candidate)**: Final validation before production
5. **Production**: Deployed to production environment
6. **Deprecated**: Still available but scheduled for removal
7. **Archived**: No longer available for new deployments

### State Transitions

```
Development → Alpha → Beta → RC → Production → Deprecated → Archived
```

### Transition Criteria

#### To Alpha
- All unit tests pass
- Initial validation metrics meet minimum thresholds
- Code review completed

#### To Beta
- Extended validation on test datasets
- Performance metrics within 5% of previous production version
- Security review completed

#### To RC
- Full integration testing completed
- A/B test plan developed
- Documentation completed

#### To Production
- Successful A/B testing
- Final approval from ML Review Board
- Deployment runbook completed and tested

#### To Deprecated
- Successor version in production for 30+ days
- Migration path documented
- Deprecation notice distributed

#### To Archived
- No active users for 90+ days
- All clients migrated to newer versions

## Compatibility Management

### API Compatibility

- **Major versions** may introduce breaking API changes
- **Minor and patch versions** must maintain API compatibility
- All versions must document input/output schema changes

### Backward Compatibility

For minor version changes:

- Input format must be backward compatible
- Output format should be backward compatible or provide transformation utilities
- Configuration format should be backward compatible

### Feature Stability

Features are classified as:

- **Stable**: Will not change in minor versions
- **Beta**: May change in minor versions with notice
- **Experimental**: May change without notice

## Artifact Management

### Required Artifacts

Each model version must include:

1. **Model Artifacts**: Serialized model files
2. **Configuration**: Model parameters and settings
3. **Metadata**: Version information and dependencies
4. **Metrics**: Performance measurements
5. **Documentation**: Usage documentation and model card
6. **Tests**: Validation test suite

### Storage Location

Artifacts are stored in the following locations:

- **Model Files**: S3 bucket with versioned paths
- **Configuration**: Both in S3 and source control
- **Metadata**: DynamoDB model registry table
- **Metrics**: Both in S3 and monitoring database
- **Documentation**: In source control and documentation site
- **Tests**: In source control

### Artifact Naming Convention

```
<model_name>-<version>[-<qualifier>].<extension>
```

Examples:
- `ad-score-predictor-1.1.0.pt`
- `account-health-predictor-2.0.0-beta.pkl`

## Deployment Strategy

### Deployment Environments

Models are deployed to these environments in sequence:

1. **Development**: For active development and testing
2. **Staging**: For integration testing and validation
3. **Shadow**: Running alongside production but not serving traffic
4. **Production**: Serving live traffic

### Canary Deployments

For major version changes:

1. Deploy to 5% of traffic initially
2. Monitor metrics for 24 hours
3. If successful, increase to 20% for 48 hours
4. If successful, increase to 50% for 48 hours
5. If successful, deploy to 100%

### Deployment Verification

Before proceeding to higher traffic percentages:

1. Key performance metrics within acceptable range
2. No increase in error rates
3. Latency within target thresholds
4. No unexpected behaviors reported

### Automated Deployment

Deployments are managed via CI/CD pipeline:

```yaml
deployment:
  environments:
    - name: development
      requirements:
        - unit_tests_passed
        - integration_tests_passed
    - name: staging
      requirements:
        - development_deployment_successful
        - performance_tests_passed
    - name: shadow
      requirements:
        - staging_deployment_successful
        - compatibility_tests_passed
    - name: production
      requirements:
        - shadow_deployment_successful
        - ml_review_board_approval
```

## Rollback Procedures

### Automatic Rollback Triggers

Automatic rollbacks are triggered by:

1. Error rate exceeding 5% for 5 minutes
2. Latency p95 exceeding 500ms for 10 minutes
3. Prediction quality metrics deviating by >10% from baseline

### Manual Rollback

Manual rollback process:

1. Identify issue requiring rollback
2. Create rollback ticket with justification
3. Get approval from service owner
4. Execute rollback to previous version
5. Monitor system metrics post-rollback
6. Document incident and resolution

### Version Restoration

To restore a previous version:

```python
from within.ml.registry import ModelRegistry
from within.ml.deploy import ModelDeployer

# Initialize components
registry = ModelRegistry()
deployer = ModelDeployer()

# Restore previous version
previous_version = "1.0.0"
model = registry.get_model(
    model_name="ad-score-predictor",
    version=previous_version
)

# Deploy previous version
deployment = deployer.deploy(
    model=model,
    environment="production",
    is_rollback=True,
    rollback_reason="Performance degradation in v1.1.0"
)
```

## Documentation Requirements

### Required Documentation

Each model version must include:

1. **Model Card**: Overview of model purpose, capabilities, and limitations
2. **API Documentation**: Input/output specifications and examples
3. **Version History**: Changes from previous versions
4. **Performance Report**: Metrics on test datasets
5. **Limitations Document**: Known limitations and edge cases
6. **Integration Guide**: How to integrate with this model version

### Documentation Template

```markdown
# Model: {model_name}
## Version: {version}

### Overview
{model_description}

### Changes from Previous Version
{changelog}

### API Specification
{api_details}

### Performance Metrics
{metrics_table}

### Known Limitations
{limitations}

### Integration Examples
{code_examples}
```

### Documentation Versioning

- Documentation is versioned alongside model versions
- Previous documentation versions remain accessible
- All documentation changes tracked in version control

## Testing Protocol

### Required Tests

Each model version must pass these tests:

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test interoperability with other components
3. **Performance Tests**: Measure throughput, latency, and resource usage
4. **Regression Tests**: Compare with previous versions
5. **Edge Case Tests**: Test behavior with unusual inputs
6. **A/B Tests**: Compare with current production version

### Test Metrics

Test results must include:

- Accuracy metrics (e.g., RMSE, F1, AUC)
- Latency measurements (p50, p95, p99)
- Resource usage (memory, CPU, GPU)
- Throughput measurements
- Error rates

### Automated Test Pipeline

```python
from within.ml.testing import ModelTestSuite

# Initialize test suite
test_suite = ModelTestSuite(
    model_name="ad-score-predictor",
    version="1.1.0"
)

# Run tests
results = test_suite.run_all()

# Generate report
report = test_suite.generate_report()

# Determine if tests passed
if results.passed:
    print("Tests passed, proceeding with deployment")
else:
    print("Tests failed, blocking deployment")
    print(results.failure_summary)
```

## Governance

### Approval Process

Model versions require approvals at these stages:

1. **Design Approval**: Planned architecture and approach
2. **Alpha Approval**: Release to internal testing
3. **Beta Approval**: Release to selected customers
4. **Production Approval**: Release to all users

### Approval Authority

| Stage | Approver |
|-------|----------|
| Design | ML Architect |
| Alpha | ML Team Lead |
| Beta | Product Manager |
| Production | ML Review Board |

### Approval Criteria

Production approval requires:

1. Documentation complete and accurate
2. All tests passing
3. Performance metrics meet or exceed previous version
4. Security review completed
5. Privacy impact assessment completed
6. Ethics review completed (if applicable)

### Versioning Audit Trail

All version changes are logged with:

- Timestamp
- User who made the change
- Reason for the change
- Approvers
- Related tickets/issues

---

**Document Revision History**:
- v1.0 (2023-01-10): Initial version
- v1.1 (2023-03-25): Added deployment verification section
- v1.2 (2023-06-18): Updated artifact management guidelines
- v2.0 (2023-09-30): Major revision with governance and compatibility sections 