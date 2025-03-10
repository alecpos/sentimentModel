#!/bin/bash

# Documentation generation script for WITHIN ML projects
# Usage: ./create_doc.sh [type] [id/name] "[title]"
# Example: ./create_doc.sh epic 2 "Ad Account Health Monitoring"
# Example: ./create_doc.sh reasoning account_health "Account Health Prediction Model"
# Example: ./create_doc.sh story ML-7 "Implement NLP Pipeline for Content Analysis"
# Example: ./create_doc.sh model ad_sentiment "Ad Sentiment Analyzer"

set -e

# Check if enough arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: ./create_doc.sh [type] [id/name] \"[title]\""
    echo "Types: epic, reasoning, story, model"
    exit 1
fi

# Get arguments
DOC_TYPE=$1
ID_OR_NAME=$2
TITLE=$3
DATE=$(date +"%Y-%m-%d")
BASE_DIR="docs/implementation/ml"

# Create docs directory if it doesn't exist
mkdir -p "$BASE_DIR/epics" "$BASE_DIR/reasoning" "$BASE_DIR/stories" "$BASE_DIR/models"

# Function to create epic summary document
create_epic_summary() {
    local epic_num=$1
    local epic_title=$2
    local file_path="$BASE_DIR/epics/epic_${epic_num}_summary.md"
    
    echo "Creating epic summary document at $file_path"
    
    cat > "$file_path" << EOF
# EPIC ${epic_num}: ${epic_title} - Implementation Summary

## Epic Overview
[3-5 sentences describing the epic's purpose, goals, and significance to the overall WITHIN ML Prediction System]

## Core Components Implemented

### 1. [Component Name]
- **[Feature 1]**: [1-2 sentences describing implementation details]
- **[Feature 2]**: [1-2 sentences describing implementation details]
- **[Feature 3]**: [1-2 sentences describing implementation details]
- **[Feature 4]**: [1-2 sentences describing implementation details]

### 2. [Component Name]
- **[Feature 1]**: [1-2 sentences describing implementation details]
- **[Feature 2]**: [1-2 sentences describing implementation details]
- **[Feature 3]**: [1-2 sentences describing implementation details]
- **[Feature 4]**: [1-2 sentences describing implementation details]

### 3. [Component Name]
- **[Feature 1]**: [1-2 sentences describing implementation details]
- **[Feature 2]**: [1-2 sentences describing implementation details]
- **[Feature 3]**: [1-2 sentences describing implementation details]
- **[Feature 4]**: [1-2 sentences describing implementation details]

## Implementation Details

### Data Architecture
[4-6 sentences describing the data flow, storage solutions, and schema design]

Key technical decisions:
- [Decision 1 with brief rationale]
- [Decision 2 with brief rationale]
- [Decision 3 with brief rationale]
- [Decision 4 with brief rationale]

### Machine Learning Implementation
[4-6 sentences describing ML models, algorithms, and approaches used]

Model performance:
- [Performance metric 1]: [Value] ([context/comparison])
- [Performance metric 2]: [Value] ([context/comparison])
- [Performance metric 3]: [Value] ([context/comparison])

### Technology Stack
- **Backend**: [Technologies used]
- **Data Processing**: [Technologies used]
- **ML Framework**: [Technologies used]
- **Visualization**: [Technologies used]
- **Monitoring**: [Technologies used]
- **Storage**: [Technologies used]

## Integration Points

### External Systems
- **[System 1]**: [How integration works, key interfaces]
- **[System 2]**: [How integration works, key interfaces]
- **[System 3]**: [How integration works, key interfaces]

### Internal Components
- **[Component 1]**: [How integration works, key interfaces]
- **[Component 2]**: [How integration works, key interfaces]
- **[Component 3]**: [How integration works, key interfaces]

## Performance Metrics

| Metric | Target | Achieved | Variance |
|--------|--------|----------|----------|
| [Metric 1] | [Target value] | [Actual value] | [+/-XX%] |
| [Metric 2] | [Target value] | [Actual value] | [+/-XX%] |
| [Metric 3] | [Target value] | [Actual value] | [+/-XX%] |
| [Metric 4] | [Target value] | [Actual value] | [+/-XX%] |

## Testing Results
- **Unit Tests**: [Number] tests, [Pass rate]% passing
- **Integration Tests**: [Number] tests, [Pass rate]% passing
- **Performance Tests**: [Brief summary of results]
- **User Acceptance**: [Brief summary of feedback]

### Edge Cases Tested
| Edge Case | Test Scenario | Result |
|-----------|---------------|--------|
| [Case 1] | [Description of scenario] | [Outcome and metrics] |
| [Case 2] | [Description of scenario] | [Outcome and metrics] |
| [Case 3] | [Description of scenario] | [Outcome and metrics] |

## Ethical and Privacy Considerations
- **[Consideration 1]**: [Implementation details and approach]
- **[Consideration 2]**: [Implementation details and approach]
- **[Consideration 3]**: [Implementation details and approach]
- **[Consideration 4]**: [Implementation details and approach]

## Known Limitations and Future Work

| Limitation | Impact | Mitigation | Future Plan |
|------------|--------|------------|-------------|
| [Limitation 1] | [Business/user impact] | [Current workarounds] | [Planned improvements] |
| [Limitation 2] | [Business/user impact] | [Current workarounds] | [Planned improvements] |
| [Limitation 3] | [Business/user impact] | [Current workarounds] | [Planned improvements] |

## Documentation
- **[Doc Type 1]**: [Status and location]
- **[Doc Type 2]**: [Status and location]
- **[Doc Type 3]**: [Status and location]
- **[Doc Type 4]**: [Status and location]

## Completion Verification
- [ ] All code reviewed and approved
- [ ] Documentation complete and up-to-date
- [ ] All tests passing
- [ ] Performance metrics meet or exceed targets
- [ ] Privacy and security measures verified
- [ ] Edge cases and error handling tested
- [ ] Technical debt documented
- [ ] Integration with other epics verified

## Sign-off
- **Developer:** [Name/Team] - [Date]
- **Reviewer:** [Name/Role] - [Date]
- **Product Owner:** [Name/Role] - [Date]
EOF

    echo "Epic summary document created successfully!"
}

# Function to create chain of reasoning document
create_reasoning_document() {
    local component_name=$1
    local component_title=$2
    local file_path="$BASE_DIR/reasoning/${component_name}_implementation_chain_of_reasoning.md"
    
    echo "Creating chain of reasoning document at $file_path"
    
    cat > "$file_path" << EOF
# ML IMPLEMENTATION CHAIN OF REASONING: ${component_title}

## INITIAL ANALYSIS

1. Analyze the ML task: [1 sentence description] for WITHIN [Project/System].
2. Examine available code components and their suitability for this task.
3. Cross-reference ML best practices with WITHIN's specific needs.
4. Identify potential challenges and opportunities for optimization.

## SOLUTION DESIGN

### Key Components Required

1. **[Component 1]**
   * [Key functionality 1]
   * [Key functionality 2]
   * [Key functionality 3]
   * [Key functionality 4]

2. **[Component 2]**
   * [Key functionality 1]
   * [Key functionality 2]
   * [Key functionality 3]
   * [Key functionality 4]

3. **[Component 3]**
   * [Key functionality 1]
   * [Key functionality 2]
   * [Key functionality 3]
   * [Key functionality 4]

### ML Algorithm Selection

1. **[Algorithm/Approach 1]**
   * **Selected Approach**: [Specific implementation details]
   * **Rationale**: [Why this approach was chosen]
   * **Alternative Considered**: [Alternative approach and why it wasn't chosen]

2. **[Algorithm/Approach 2]**
   * **Selected Approach**: [Specific implementation details]
   * **Rationale**: [Why this approach was chosen]
   * **Alternative Considered**: [Alternative approach and why it wasn't chosen]

3. **[Algorithm/Approach 3]**
   * **Selected Approach**: [Specific implementation details]
   * **Rationale**: [Why this approach was chosen]
   * **Alternative Considered**: [Alternative approach and why it wasn't chosen]

### Data Pipeline Architecture

1. **Data Flow Design**
   \`\`\`
   [Visual representation of data flow]
   \`\`\`

2. **Integration with Existing System**
   * [Integration point 1]
   * [Integration point 2]
   * [Integration point 3]
   * [Integration point 4]

3. **API Design**
   * [Endpoint 1] - [Purpose and key parameters]
   * [Endpoint 2] - [Purpose and key parameters]
   * [Endpoint 3] - [Purpose and key parameters]

## IMPLEMENTATION APPROACH

### [Component 1] Implementation

\`\`\`python
# Key code snippet showing core implementation
class ComponentName:
    """Docstring describing purpose."""
    
    def __init__(self, param1, param2):
        """Initialize with key parameters."""
        self.param1 = param1
        self.param2 = param2
        
    def core_method(self, input_data):
        """
        Core method implementing the key functionality.
        
        Detailed explanation of approach.
        """
        # Implementation details
        result = process(input_data)
        return result
\`\`\`

### [Component 2] Implementation

\`\`\`python
# Key code snippet showing core implementation
\`\`\`

### [Component 3] Implementation

\`\`\`python
# Key code snippet showing core implementation
\`\`\`

## OPTIMIZATION STRATEGY

### Identified Bottlenecks

1. **[Bottleneck 1]**
   * [Details of the bottleneck]
   * [Impact on performance/functionality]

2. **[Bottleneck 2]**
   * [Details of the bottleneck]
   * [Impact on performance/functionality]

3. **[Bottleneck 3]**
   * [Details of the bottleneck]
   * [Impact on performance/functionality]

### Applied Optimizations

1. **[Optimization 1]**
   \`\`\`python
   # Code snippet showing optimization
   \`\`\`

2. **[Optimization 2]**
   \`\`\`python
   # Code snippet showing optimization
   \`\`\`

3. **[Optimization 3]**
   \`\`\`python
   # Code snippet showing optimization
   \`\`\`

## VERIFICATION PROCESS

### Requirement Validation

1. **[Requirement 1]**
   * [Implementation details]
   * [Verification approach]
   * [Metrics achieved]

2. **[Requirement 2]**
   * [Implementation details]
   * [Verification approach]
   * [Metrics achieved]

3. **[Requirement 3]**
   * [Implementation details]
   * [Verification approach]
   * [Metrics achieved]

### Testing Approach

1. **Unit Testing**
   \`\`\`python
   # Sample test code
   \`\`\`

2. **Integration Testing**
   \`\`\`python
   # Sample test code
   \`\`\`

3. **Performance Testing**
   \`\`\`python
   # Sample test code
   \`\`\`

## ETHICAL CONSIDERATIONS

1. **[Consideration 1]**
   * [Implementation details]
   * [Mitigation strategy]

2. **[Consideration 2]**
   * [Implementation details]
   * [Mitigation strategy]

3. **[Consideration 3]**
   * [Implementation details]
   * [Mitigation strategy]

## EXECUTION

This implementation of [Component/System Name] provides [summary of key benefits]. The system is designed to be:

1. **[Quality 1]**: [Brief explanation of how this is achieved]
2. **[Quality 2]**: [Brief explanation of how this is achieved]
3. **[Quality 3]**: [Brief explanation of how this is achieved]
4. **[Quality 4]**: [Brief explanation of how this is achieved]
5. **[Quality 5]**: [Brief explanation of how this is achieved]

The implementation leverages [key existing components], while adding [key new capabilities]. The modular design allows for [future benefits/extensions].
EOF

    echo "Chain of reasoning document created successfully!"
}

# Function to create story completion document
create_story_document() {
    local story_id=$1
    local story_title=$2
    local file_path="$BASE_DIR/stories/story_completion_${story_id,,}_${story_title,,// /_}.md"
    
    echo "Creating story completion document at $file_path"
    
    cat > "$file_path" << EOF
# Story Completion Report

## Story Information
- **Story ID:** ${story_id}
- **Epic:** [Epic Name]
- **Title:** ${story_title}
- **Story Points:** [Points]
- **Assigned To:** [Name/Team]
- **Reviewers:** [Names/Roles]
- **Completion Date:** ${DATE}

## Implementation Summary
### Core Components
- **[Component 1]**: [Brief description of implementation]
- **[Component 2]**: [Brief description of implementation]
- **[Component 3]**: [Brief description of implementation]

### Implementation Details
- **Techniques Implemented:**
  - **[Technique 1]**: [Description and rationale]
  - **[Technique 2]**: [Description and rationale]
  - **[Technique 3]**: [Description and rationale]

### Algorithms
- **Key Algorithms:**
  - **[Algorithm 1]**: [Purpose and implementation details]
  - **[Algorithm 2]**: [Purpose and implementation details]

## Acceptance Criteria Fulfillment
| Criterion | Implementation | Evidence | Status |
|-----------|----------------|----------|--------|
| [Criterion 1] | [How implemented] | [Proof/metrics] | ✅ |
| [Criterion 2] | [How implemented] | [Proof/metrics] | ✅ |
| [Criterion 3] | [How implemented] | [Proof/metrics] | ✅ |

## Technical Implementation Details

### Data Architecture
- **Data Flow:**
  \`\`\`
  [Visual representation of data flow]
  \`\`\`
- **Data Persistence:**
  - [Persistence mechanism 1]
  - [Persistence mechanism 2]
  - [Persistence mechanism 3]

### Architecture
- **Key Components:**
  - [Component 1]: [Function]
  - [Component 2]: [Function]
  - [Component 3]: [Function]

- **Key Parameters:**
  - [Parameter 1]: [Value] ([Justification])
  - [Parameter 2]: [Value] ([Justification])
  - [Parameter 3]: [Value] ([Justification])

### Integration Points
- **External Systems:**
  - [System 1]: [Integration method]
  - [System 2]: [Integration method]

- **Internal Components:**
  - [Component 1]: [Integration method]
  - [Component 2]: [Integration method]

### Performance Metrics
| Metric | Target | Achieved | Variance |
|--------|--------|----------|----------|
| [Metric 1] | [Target] | [Actual] | [Variance]% |
| [Metric 2] | [Target] | [Actual] | [Variance]% |
| [Metric 3] | [Target] | [Actual] | [Variance]% |

## Testing Results
- **Test Coverage:** [Percentage]%
- **Unit Tests:** [Number] tests, [Pass rate]% passing
- **Integration Tests:** [Number] tests, [Pass rate]% passing
- **Performance Tests:** [Brief summary]

### Edge Cases Tested
| Edge Case | Test Scenario | Result |
|-----------|---------------|--------|
| [Case 1] | [Description] | [Outcome] |
| [Case 2] | [Description] | [Outcome] |
| [Case 3] | [Description] | [Outcome] |

## Known Limitations and Future Work
| Limitation | Impact | Mitigation | Future Plan |
|------------|--------|------------|-------------|
| [Limitation 1] | [Impact] | [Mitigation] | [Future work] |
| [Limitation 2] | [Impact] | [Mitigation] | [Future work] |
| [Limitation 3] | [Impact] | [Mitigation] | [Future work] |

## Documentation
- **API Documentation:** [Status/Location]
- **User Guide:** [Status/Location]
- **Technical Reference:** [Status/Location]

## Completion Verification Checklist
- [ ] All code reviewed and approved
- [ ] Documentation complete and up-to-date
- [ ] All tests passing
- [ ] Performance metrics meet or exceed targets
- [ ] Edge cases and error handling tested
- [ ] Ethical considerations addressed and documented
- [ ] Integration with other components verified
- [ ] Technical debt documented

## Sign-off
- **Developer:** [Name] - ${DATE}
- **Reviewer:** [Name] - ${DATE}
- **Product Owner:** [Name] - ${DATE}
EOF

    echo "Story completion document created successfully!"
}

# Function to create model card document
create_model_card() {
    local model_name=$1
    local model_title=$2
    local file_path="$BASE_DIR/models/model_card_${model_name}.md"
    
    echo "Creating model card document at $file_path"
    
    cat > "$file_path" << EOF
# ML Model Card: ${model_title}

## Model Overview
- **Model Name:** ${model_title}
- **Version:** 1.0.0
- **Type:** [e.g., Classification, Regression, NLP, etc.]
- **Purpose:** [Brief description of what the model does]
- **Created Date:** ${DATE}
- **Last Updated:** ${DATE}

## Intended Use
- **Primary Use Cases:** 
  - [Use case 1]
  - [Use case 2]
  - [Use case 3]
  - [Use case 4]

- **Out-of-Scope Uses:**
  - [Use case 1]
  - [Use case 2]
  - [Use case 3]
  - [Use case 4]

- **Target Users:**
  - [User type 1]
  - [User type 2]
  - [User type 3]
  - [User type 4]

## Training Data
- **Dataset Sources:**
  - [Source 1] ([details])
  - [Source 2] ([details])
  - [Source 3] ([details])

- **Dataset Size:** [Number] examples

- **Feature Distribution:**
  - [Distribution category 1]: [statistics]
  - [Distribution category 2]: [statistics]
  - [Distribution category 3]: [statistics]

- **Data Preparation:**
  - [Preparation step 1]
  - [Preparation step 2]
  - [Preparation step 3]

## Model Architecture
- **Algorithm Type:** [Algorithm name]
- **Architecture Details:**
  - [Detail 1]
  - [Detail 2]
  - [Detail 3]
  - [Detail 4]

- **Feature Inputs:**
  - [Input type 1]
  - [Input type 2]
  - [Input type 3]

- **Output Format:**
  - [Output type 1]
  - [Output type 2]
  - [Output type 3]

## Performance Metrics
| Metric | Value | Notes |
|--------|-------|-------|
| [Metric 1] | [Value] | [Context] |
| [Metric 2] | [Value] | [Context] |
| [Metric 3] | [Value] | [Context] |
| [Metric 4] | [Value] | [Context] |
| [Metric 5] | [Value] | [Context] |

## Limitations and Biases
- **Known Limitations:**
  - [Limitation 1]
  - [Limitation 2]
  - [Limitation 3]
  - [Limitation 4]

- **Potential Biases:**
  - [Bias 1]
  - [Bias 2]
  - [Bias 3]

- **Evaluation Results by Segment:**
  | Segment | [Metric 1] | [Metric 2] | Notes |
  |---------|------------|------------|-------|
  | [Segment 1] | [Value] | [Value] | [Notes] |
  | [Segment 2] | [Value] | [Value] | [Notes] |
  | [Segment 3] | [Value] | [Value] | [Notes] |
  | [Segment 4] | [Value] | [Value] | [Notes] |
  | [Segment 5] | [Value] | [Value] | [Notes] |

## Ethical Considerations
- **Data Privacy:**
  - [Consideration 1]
  - [Consideration 2]
  - [Consideration 3]

- **Fairness Assessment:**
  - [Assessment 1]
  - [Assessment 2]
  - [Assessment 3]

- **Potential Risks:**
  - [Risk 1]
  - [Risk 2]
  - [Risk 3]

## Usage Instructions
- **Required Environment:**
  - [Requirement 1]
  - [Requirement 2]
  - [Requirement 3]
  - [Requirement 4]

- **Setup Steps:**
  \`\`\`python
  # Code example showing initialization
  from path.to.module import ModelClass
  
  model = ModelClass(param1, param2)
  \`\`\`

- **Inference Examples:**
  \`\`\`python
  # Code example showing usage
  result = model.predict(input_data)
  
  # Processing result
  processed_result = process_output(result)
  \`\`\`

- **API Reference:** [Link or path to detailed API documentation]

## Maintenance
- **Owner:** [Team or individual responsible]
- **Update Frequency:** [How often the model is updated]
- **Monitoring Plan:**
  - [Monitoring aspect 1]
  - [Monitoring aspect 2]
  - [Monitoring aspect 3]

- **Retraining Triggers:**
  - [Trigger 1]
  - [Trigger 2]
  - [Trigger 3]
  - [Trigger 4]

## Version History
| Version | Date | Changes | Performance Delta |
|---------|------|---------|-------------------|
| 1.0.0 | ${DATE} | Initial release | Baseline |
| [Version] | [Date] | [Key changes] | [Performance change] |
| [Version] | [Date] | [Key changes] | [Performance change] |
| [Version] | [Date] | [Key changes] | [Performance change] |
EOF

    echo "Model card document created successfully!"
}

# Execute based on document type
case $DOC_TYPE in
    epic)
        create_epic_summary "$ID_OR_NAME" "$TITLE"
        ;;
    reasoning)
        create_reasoning_document "$ID_OR_NAME" "$TITLE"
        ;;
    story)
        create_story_document "$ID_OR_NAME" "$TITLE"
        ;;
    model)
        create_model_card "$ID_OR_NAME" "$TITLE"
        ;;
    *)
        echo "Unknown document type: $DOC_TYPE"
        echo "Valid types are: epic, reasoning, story, model"
        exit 1
        ;;
esac

echo "Documentation file created successfully at: $BASE_DIR"
echo "Remember to fill in the template placeholders with actual content." 