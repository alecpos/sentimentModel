# WITHIN ML Documentation Style Guide

**IMPLEMENTATION STATUS: PARTIALLY_IMPLEMENTED**


## Overview

This style guide establishes standards for all documentation in the WITHIN ML Prediction System project. Consistent documentation improves readability, maintainability, and knowledge transfer among team members.

## Documentation Types

### 1. Epic Summaries
- **Purpose**: Provide high-level overview of multi-story implementations
- **Location**: `docs/implementation/ml/epics/`
- **Naming Convention**: `epic_[number]_summary.md`
- **Target Audience**: Project managers, architects, stakeholders

### 2. Implementation Chain of Reasoning
- **Purpose**: Explain technical decisions and implementation approach
- **Location**: `docs/implementation/ml/reasoning/`
- **Naming Convention**: `[component_name]_implementation_chain_of_reasoning.md`
- **Target Audience**: ML engineers, developers

### 3. Story Completion Reports
- **Purpose**: Document completed user stories with implementation details
- **Location**: `docs/implementation/ml/stories/`
- **Naming Convention**: `story_completion_[story_id]_[story_name].md`
- **Target Audience**: Team leads, product owners, QA engineers

### 4. Model Cards
- **Purpose**: Standardize model information, usage, and performance metrics
- **Location**: `docs/implementation/ml/models/`
- **Naming Convention**: `model_card_[model_name].md`
- **Target Audience**: Data scientists, ML engineers, model users

### 5. API Documentation
- **Purpose**: Document API endpoints, request/response formats
- **Location**: `docs/implementation/api/`
- **Naming Convention**: Various, based on component
- **Target Audience**: Developers, integration engineers

### 6. User Documentation
- **Purpose**: Explain system usage for end users
- **Location**: `docs/implementation/user/`
- **Naming Convention**: Various, based on component
- **Target Audience**: End users, customer support

## Formatting Guidelines

### Markdown Standards

1. **Headers**
   - Use ATX-style headers (`#` for level 1, `##` for level 2, etc.)
   - Include one space after the `#` characters
   - Capitalize first word and proper nouns only

2. **Lists**
   - Use `-` for unordered lists
   - Use `1.` for ordered lists (the actual numbers don't matter for rendering)
   - Indent nested lists with 2 spaces

3. **Code Blocks**
   - Use triple backticks with language specifier: \```python
   - For inline code, use single backticks: \`code\`

4. **Tables**
   - Use standard Markdown table format
   - Include header row and separator row
   - Align columns as appropriate (default left-align)

### Content Standards

1. **Technical Depth**
   - Match technical depth to target audience
   - Include more detail for technical audiences
   - Focus on business impact for stakeholder audiences

2. **Completeness**
   - All required sections must be completed
   - Use "N/A" or "None" rather than leaving sections blank
   - Mark draft sections with "TODO" and assignee

3. **Language**
   - Use present tense for current functionality
   - Use future tense only for planned features
   - Be concise and direct
   - Avoid jargon unless necessary for technical accuracy

4. **Examples**
   - Include concrete examples for complex concepts
   - Ensure code examples are tested and working
   - Use realistic data in examples

## Document Structure

### Standard Sections
All documents should include:

1. **Title**
   - Clear, descriptive title
   - Include relevant identifiers (Epic/Story ID, version, etc.)

2. **Overview/Purpose**
   - Brief explanation of the document's subject
   - Relevance to the overall system

3. **Main Content**
   - Follows template for the specific document type
   - All required sections completed

4. **References**
   - Links to related documents
   - Links to source code when relevant
   - Citations for external resources

5. **Sign-off**
   - Names and roles of authors and reviewers
   - Date of last update

## Review Process

1. **Before Submission**
   - Self-review against this style guide
   - Run spell check and fix any issues
   - Verify all links work

2. **Review Criteria**
   - Technical accuracy
   - Completeness
   - Compliance with style guide
   - Appropriate technical depth for audience

3. **Approval Process**
   - Technical review by subject matter expert
   - Quality review by documentation lead
   - Final approval by project manager or product owner

## Versioning

1. **Document Versioning**
   - Update date and version number with significant changes
   - Include changelog for major revisions
   - Maintain old versions when appropriate

2. **Compatible Product Versions**
   - Clearly indicate which product versions the document applies to
   - Update documentation when product changes affect accuracy

## Examples

See the `docs/standards/examples/` directory for exemplary documentation that follows these guidelines:

- Epic Summary: `examples/epic_summary_example.md`
- Chain of Reasoning: `examples/chain_of_reasoning_example.md`
- Story Completion: `examples/story_completion_example.md`
- Model Card: `examples/model_card_example.md` 