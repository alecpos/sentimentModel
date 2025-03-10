# Natural Language Processing Components

This directory contains the NLP components and text analysis modules used throughout the WITHIN Ad Score & Account Health Predictor system. These components extract insights from ad copy, user comments, reviews, and other text data to power prediction models.

## Directory Structure

```
nlp/
├── __init__.py                     # NLP package initialization
├── preprocessing/                  # Text preprocessing modules
│   ├── __init__.py                 # Preprocessing package initialization
│   ├── cleaner.py                  # Text cleaning utilities
│   ├── normalizer.py               # Text normalization functions
│   ├── tokenizer.py                # Tokenization implementations
│   └── stopwords.py                # Stopwords handling (domain-specific)
├── feature_extraction/             # Feature extraction from text
│   ├── __init__.py                 # Feature extraction package initialization
│   ├── embeddings.py               # Text embedding generators
│   ├── tfidf.py                    # TF-IDF feature extraction
│   └── transformers.py             # Transformer model integrations
├── sentiment/                      # Sentiment analysis components
│   ├── __init__.py                 # Sentiment package initialization
│   ├── classifier.py               # Sentiment classification models
│   ├── aspect_based.py             # Aspect-based sentiment analysis
│   └── calibration.py              # Domain-specific sentiment calibration
├── entities/                       # Entity recognition components
│   ├── __init__.py                 # Entities package initialization
│   ├── extractor.py                # Named entity recognition 
│   ├── ad_entities.py              # Ad-specific entity types
│   └── linker.py                   # Entity linking and resolution
├── topics/                         # Topic modeling components
│   ├── __init__.py                 # Topics package initialization
│   ├── lda.py                      # Latent Dirichlet Allocation models
│   ├── bertopic.py                 # BERT-based topic modeling
│   └── coherence.py                # Topic coherence evaluation
└── pipeline.py                     # Integrated NLP pipeline
```

## Core Components

### Text Processor

The `AdTextProcessor` is the primary entry point for text analysis, providing an integrated pipeline for feature extraction from ad text.

```python
from app.nlp import AdTextProcessor

# Initialize processor
processor = AdTextProcessor(tokenizer="wordpiece", embedding_model="ad-optimized-bert")

# Process ad text
features = processor.process(
    text="Limited Time Offer: 20% Off All Products. Shop Now!",
    extract_entities=True,
    include_sentiment=True
)

# Access extracted features
embedding = features["embedding"]
sentiment = features["sentiment"]
entities = features["entities"]
token_count = features["token_count"]
```

### Preprocessing Pipeline

The preprocessing pipeline handles text cleaning, normalization, and tokenization:

1. **Cleaning**: Removes HTML, excess whitespace, special characters
2. **Normalization**: Lowercases text, normalizes Unicode, expands contractions
3. **Tokenization**: Splits text into tokens using appropriate tokenizers

### Embedding Models

Multiple embedding models are available:

| Model | Description | Dimension | Usage |
|-------|-------------|-----------|-------|
| `ad-optimized-bert` | BERT fine-tuned on advertising text | 768 | General ad text analysis |
| `product-bert` | BERT fine-tuned on product descriptions | 768 | Product-focused ads |
| `fasttext-ads` | FastText model trained on ad corpus | 300 | Lightweight analysis |
| `tfidf-vectorizer` | TF-IDF based vectors | Configurable | Traditional NLP approaches |
| `ad-sentiment-roberta` | RoBERTa fine-tuned for ad sentiment | 768 | Detailed sentiment analysis |

### Sentiment Analysis

The sentiment analyzer provides both general and advertising-specific sentiment analysis:

```python
from app.nlp.sentiment import AdSentimentAnalyzer

# Initialize analyzer
analyzer = AdSentimentAnalyzer(model="ad-sentiment-roberta")

# Analyze ad sentiment
sentiment = analyzer.analyze("Limited Time Offer: 20% Off All Products. Shop Now!")

# Returns:
# {
#     "score": 0.78,          # Positive sentiment score (0-1)
#     "label": "positive",    # Sentiment classification
#     "confidence": 0.92,     # Confidence in the prediction
#     "aspects": {            # Aspect-based sentiment
#         "offer": 0.85,
#         "price": 0.92,
#         "urgency": 0.75
#     }
# }
```

### Entity Recognition

The entity extraction system identifies ad-specific entities:

```python
from app.nlp.entities import AdEntityExtractor

# Initialize extractor
extractor = AdEntityExtractor(domain="advertising")

# Extract entities
entities = extractor.extract("Limited Time Offer: 20% Off All Products. Shop Now!")

# Returns:
# [
#     {"type": "DISCOUNT", "text": "20% Off", "start": 22, "end": 29, "value": 0.2},
#     {"type": "OFFER_TYPE", "text": "Limited Time Offer", "start": 0, "end": 19},
#     {"type": "PRODUCT", "text": "All Products", "start": 30, "end": 42},
#     {"type": "CTA", "text": "Shop Now", "start": 44, "end": 52}
# ]
```

### Topic Modeling

The topic modeling system identifies key themes in ad content:

```python
from app.nlp.topics import AdTopicModeler

# Initialize topic modeler
topic_modeler = AdTopicModeler(model="bertopic")

# Extract topics
topics = topic_modeler.extract_topics(["Limited Time Offer: 20% Off All Products. Shop Now!"])

# Returns:
# {
#     "topics": [
#         {"id": 3, "label": "sales_promotions", "score": 0.78},
#         {"id": 12, "label": "limited_time_offers", "score": 0.65}
#     ],
#     "coherence": 0.82
# }
```

## Integration with Ad Scoring

The NLP components provide essential features for the ad scoring system:

1. **Content Quality Assessment**: Evaluates writing quality, clarity, and persuasiveness
2. **Sentiment Prediction**: Predicts audience sentiment response to ad copy
3. **Engagement Estimation**: Estimates likelihood of engagement based on language patterns
4. **Conversion Potential**: Analyzes call-to-action effectiveness and purchase intent signals
5. **Platform Appropriateness**: Evaluates text suitability for specific ad platforms

## Performance Optimization

NLP components implement several optimizations:

1. **Caching**: Frequently processed text is cached to avoid recomputation
2. **Batch Processing**: Multiple texts can be processed in batches for efficiency
3. **Quantization**: Models use INT8 quantization where possible
4. **Lazy Loading**: Models are loaded on demand to reduce memory footprint
5. **Embedding Persistence**: Embeddings are stored for reuse across analyses

## Model Management

Pre-trained models are versioned and managed through a model registry:

```python
from app.nlp import model_registry

# List available models
models = model_registry.list_models(category="sentiment")

# Get model info
model_info = model_registry.get_model_info("ad-sentiment-roberta")

# Load specific model version
model = model_registry.load_model("ad-sentiment-roberta", version="1.2.0")
```

## Development Guidelines

When enhancing or adding NLP components:

1. **Consistent Tokenization**: Use standardized tokenization across components
2. **Text Standardization**: Apply consistent cleaning procedures
3. **Domain-Specific Stopwords**: Use advertising-specific stopword lists
4. **Document Normalization**: Document all text processing steps
5. **Embedding Techniques**: Implement both traditional and modern embedding methods
6. **Versioned Models**: Use explicit versioning for all pre-trained models
7. **Performance Optimization**: Cache embeddings and other expensive computations
8. **Advertising Calibration**: Calibrate sentiment models for advertising language
9. **Custom Entity Types**: Implement ad-specific entity types
10. **Entity Disambiguation**: Handle ambiguous entities properly 