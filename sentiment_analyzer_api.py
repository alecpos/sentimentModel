#!/usr/bin/env python
"""
Sentiment Analyzer API

A FastAPI-based API for sentiment analysis using our trained model on the Sentiment140 dataset.
This API provides endpoints for:
- Single text sentiment prediction
- Batch text sentiment prediction

Usage:
    uvicorn sentiment_analyzer_api:app --reload
"""

import os
import sys
import json
import logging
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import joblib
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the parent directory to the path so we can import the SentimentAnalyzer class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sentiment140_direct import SentimentAnalyzer, logger

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analyzer API",
    description="API for sentiment analysis using a model trained on the Sentiment140 dataset.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define request models
class TextRequest(BaseModel):
    text: str = Field(..., description="The text to analyze for sentiment")

class BatchTextRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze for sentiment")

# Define response models
class SentimentResponse(BaseModel):
    text: str = Field(..., description="The input text")
    sentiment_label: str = Field(..., description="The sentiment label (positive or negative)")
    sentiment_score: float = Field(..., description="The sentiment score (-1.0 to 1.0)")
    confidence: float = Field(..., description="Confidence of the prediction (0.0 to 1.0)")

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse] = Field(..., description="List of sentiment predictions")

# Global variables
MODEL_DIR = "models/sentiment140"
sentiment_analyzer = None

def load_latest_model():
    """Load the latest model from the models directory."""
    global sentiment_analyzer
    
    # Find all joblib files in the model directory
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.joblib"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODEL_DIR}")
    
    # Get the most recent model file
    latest_model = max(model_files, key=os.path.getctime)
    logger.info(f"Loading model from {latest_model}")
    
    # Initialize the sentiment analyzer and load the model
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.load_model(latest_model)
    logger.info("Model loaded successfully")
    
    # Return model metadata
    return {
        "model_path": latest_model,
        "model_type": sentiment_analyzer.model_type,
        "timestamp": datetime.fromtimestamp(os.path.getctime(latest_model)).isoformat(),
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts."""
    try:
        model_info = load_latest_model()
        logger.info(f"Initialized with model: {model_info}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Continue without a model, we'll handle this in the endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sentiment Analyzer API",
        "version": "1.0.0",
        "endpoints": [
            "/predict - Predict sentiment for a single text",
            "/predict_batch - Predict sentiment for multiple texts",
            "/model_info - Get information about the loaded model",
        ]
    }

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model."""
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Get model information
    model_path = sentiment_analyzer.model_path if hasattr(sentiment_analyzer, "model_path") else "Unknown"
    model_type = sentiment_analyzer.model_type
    
    # Try to load metrics if available
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics: {str(e)}")
    
    return {
        "model_path": model_path,
        "model_type": model_type,
        "metrics": metrics,
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: TextRequest):
    """
    Predict sentiment for a single text.
    
    Returns a SentimentResponse with sentiment label, score, and confidence.
    """
    if sentiment_analyzer is None:
        try:
            load_latest_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model: {str(e)}")
    
    # Get prediction
    result = sentiment_analyzer.predict(request.text)
    
    # Return formatted response
    return SentimentResponse(
        text=request.text,
        sentiment_label=result["sentiment_label"],
        sentiment_score=result["sentiment_score"],
        confidence=result["confidence"]
    )

@app.post("/predict_batch", response_model=BatchSentimentResponse)
async def predict_batch(request: BatchTextRequest):
    """
    Predict sentiment for multiple texts.
    
    Returns a BatchSentimentResponse with predictions for each text.
    """
    if sentiment_analyzer is None:
        try:
            load_latest_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model: {str(e)}")
    
    # Get predictions for all texts
    results = []
    for text in request.texts:
        prediction = sentiment_analyzer.predict(text)
        results.append(
            SentimentResponse(
                text=text,
                sentiment_label=prediction["sentiment_label"],
                sentiment_score=prediction["sentiment_score"],
                confidence=prediction["confidence"]
            )
        )
    
    # Return formatted response
    return BatchSentimentResponse(results=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 