# Search Components

This directory contains search functionality components for the WITHIN ML Prediction System.

## Purpose

The search system provides capabilities to efficiently find and retrieve information across the system:
- Enable quick lookup of models, predictions, and related data
- Support semantic search across ML model descriptions and metadata
- Facilitate discovery of similar items and patterns
- Provide flexible querying with filters and relevance ranking

## Key Components

### Indexing

Components for building and maintaining search indices:
- Document and field indexing
- Index update and maintenance
- Incremental indexing for large datasets
- Schema management for indexed fields

### Query Processing

Components for interpreting and optimizing search queries:
- Query parsing and validation
- Query expansion and refinement
- Filter application and optimization
- Field boosting and relevance configuration

### Search Execution

Components for performing search operations:
- Full-text search for text content
- Vector similarity search for embeddings
- Hybrid search combining multiple approaches
- Distributed search across partitioned indices

### Result Handling

Components for processing and enhancing search results:
- Relevance scoring and ranking
- Result pagination and sorting
- Faceting and aggregation
- Result highlighting and formatting

## Usage Example

```python
from app.core.search import SearchEngine, QueryBuilder

# Initialize the search engine
search_engine = SearchEngine(index_name="ad_campaigns")

# Build a search query
query = (QueryBuilder()
    .with_text_query("high performing ads")
    .with_filter(field="performance_score", op="gt", value=0.8)
    .with_filter(field="created_at", op="range", start="2023-01-01", end="2023-12-31")
    .with_sort(field="relevance", direction="desc")
    .with_pagination(page=1, size=20)
    .build())

# Execute search
results = search_engine.search(query)

# Process results
for item in results.items:
    print(f"ID: {item.id}, Score: {item.score}, Title: {item.title}")
    
# Get result metadata
print(f"Total matches: {results.total_count}")
print(f"Search time: {results.search_time_ms}ms")
```

## Integration Points

- **ML Models**: Search is used to find similar models and predictions
- **API Layer**: Search endpoints expose search capabilities via REST API
- **Data Storage**: Search indexes are built from underlying data stores
- **Analytics**: Search patterns provide insights into user interests

## Dependencies

- Text processing utilities for tokenization and analysis
- Vector operations for similarity calculations
- Data storage interfaces for retrieving indexed content
- Caching mechanisms for high-performance queries 