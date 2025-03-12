# Enhanced Sentiment Analysis with Fairness Evaluation

This project implements an advanced sentiment analysis system with comprehensive fairness evaluation and bias mitigation techniques. It builds on the original Sentiment140 sentiment analyzer with significant improvements to model architecture, robustness, and fairness across demographic intersections.

## Project Overview

The enhanced sentiment analysis system addresses key limitations identified in our previous implementation:

1. **Improved Model Architecture**: Incorporates transformer-based models (BERT, RoBERTa, DistilBERT) that better understand context and nuance in text.

2. **Advanced Text Preprocessing**: Implements Twitter-specific text preprocessing that handles emojis, hashtags, mentions, contractions, and negations.

3. **Intersectional Fairness Analysis**: Evaluates model performance across demographic intersections, visualizing biases with heatmaps and detecting problematic demographic groups.

4. **Bias Mitigation Techniques**: Applies various bias mitigation strategies including data reweighting, counterfactual data augmentation, and adversarial training.

## Components

### Core Modules

- **`transformer_sentiment_analysis.py`**: Main script for training transformer-based models on Sentiment140
- **`fairness_evaluation.py`**: Comprehensive fairness evaluation across demographic intersections
- **`bias_mitigation.py`**: Techniques for reducing bias in sentiment analysis models

### Key Features

- **Advanced Twitter Text Processing**: Handles Twitter-specific elements with advanced preprocessing techniques
- **Intersectional Bias Analysis**: Identifies and visualizes bias patterns across demographic group intersections
- **Transformer Model Integration**: Leverages pre-trained transformers for improved language understanding
- **Fairness Metrics**: Calculates disparate impact, equalized odds, and other fairness measures
- **Bias Visualization**: Generates heatmaps and reports to understand bias patterns
- **Mitigation Techniques**: Incorporates pre-processing, in-processing, and post-processing techniques

## Usage

### Installation

1. Install required dependencies:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn emoji contractions
```

2. Download the Sentiment140 dataset:

```bash
# Using kagglehub
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('kazanova/sentiment140')"
```

### Training a Transformer-Based Model

Train a transformer model with fairness evaluation:

```bash
python transformer_sentiment_analysis.py \
  --dataset_path /path/to/sentiment140.csv \
  --model_type distilbert \
  --max_samples 400000 \
  --output_dir transformer_models/sentiment140 \
  --batch_size 32 \
  --epochs 3 \
  --fairness_evaluation \
  --bias_mitigation
```

### Fairness Evaluation on Existing Model

Evaluate fairness of model predictions:

```bash
python fairness_evaluation.py \
  --predictions_file predictions.csv \
  --output_dir fairness_results
```

### Applying Bias Mitigation

Apply bias mitigation to a dataset:

```bash
python bias_mitigation.py \
  --dataset_path /path/to/dataset.csv \
  --text_column text \
  --protected_columns gender age_group location \
  --output_path mitigated_dataset.csv
```

## Fairness Evaluation

The system performs comprehensive fairness evaluation across demographic intersections:

1. **Individual Protected Attributes**: Evaluates model performance across individual demographic variables (gender, age, location)

2. **Intersectional Analysis**: Identifies and visualizes bias patterns at intersections (e.g., young urban women vs. older rural men)

3. **Fairness Metrics**:
   - Disparate Impact: Ratio of positive prediction rates between demographic groups
   - Equalized Odds: Difference in true positive/false positive rates between groups
   - Demographic Parity: Equality of positive prediction rates across groups

4. **Visualization**:
   - Heatmaps showing prediction rates across demographic intersections
   - Summary plots of problematic demographic groups
   - Detailed fairness reports with recommendations

## Bias Mitigation Techniques

The system implements multiple bias mitigation strategies:

1. **Pre-processing Techniques**:
   - Data reweighting: Adjusts sample weights to balance prediction rates
   - Dataset balancing: Resampling to ensure equal representation
   - Counterfactual data augmentation: Creates balanced counterfactual examples
   - Gender-neutral preprocessing: Reduces gender-specific language

2. **In-processing Techniques**:
   - Adversarial training: Uses adversarial weights to reduce demographic correlations
   - Fairness constraints: Incorporates fairness objectives in model training

3. **Post-processing Techniques**:
   - Prediction calibration: Adjusts predictions to satisfy fairness constraints
   - Threshold optimization: Selects classification thresholds for fairness

## Performance Improvements

Based on our testing with the Sentiment140 dataset, the enhanced system demonstrates:

- **Accuracy**: Improved from 78.89% (logistic regression) to 82-85% (transformer models)
- **F1 Score**: Increased from 78.88% to 83-86%
- **Robustness**: Better handling of ambiguous or nuanced sentiment expressions
- **Fairness**: Reduced demographic disparities across protected attributes

## Future Directions

Ongoing development areas include:

1. **Multimodal Analysis**: Incorporating image and text for social media sentiment
2. **Emotion Detection**: Extending beyond binary sentiment to detect specific emotions
3. **Transfer Learning**: Adapting to domain-specific sentiment analysis tasks
4. **Production Integration**: Developing lightweight versions for production deployment
5. **Explainable AI**: Adding interpretability methods to understand model decisions

## Requirements

- Python 3.9+
- PyTorch 1.9+
- Transformers 4.15+
- scikit-learn 1.0+
- pandas, numpy, matplotlib, seaborn
- emoji, contractions (for advanced text preprocessing)

## Citation

If you use this code in your research, please cite:

```
@software{enhanced_sentiment_analysis,
  author = {WITHIN Team},
  title = {Enhanced Sentiment Analysis with Fairness Evaluation},
  year = {2025},
  url = {https://github.com/yourusername/enhanced-sentiment-analysis}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Sentiment140 dataset creators (Go, Bhayani, and Huang)
- The HuggingFace team for their Transformers library
- The fairness in machine learning research community 