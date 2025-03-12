#!/usr/bin/env python
"""
Sentiment Analysis API Client

This script demonstrates how to use the Sentiment Analysis API
programmatically with requests.

Usage:
    python sentiment_client.py
"""

import sys
import json
import argparse
import requests
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# API URL - Change this to match your deployment
API_BASE_URL = "http://localhost:8000"

def get_model_info():
    """Get information about the loaded model."""
    url = f"{API_BASE_URL}/model_info"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def predict_sentiment(text: str):
    """
    Predict sentiment for a single text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Sentiment prediction result
    """
    url = f"{API_BASE_URL}/predict"
    payload = {"text": text}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def predict_batch_sentiment(texts: List[str]):
    """
    Predict sentiment for multiple texts.
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        List of sentiment prediction results
    """
    url = f"{API_BASE_URL}/predict_batch"
    payload = {"texts": texts}
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def visualize_results(results: List[Dict[str, Any]]):
    """
    Visualize sentiment analysis results.
    
    Args:
        results: List of sentiment prediction results
    """
    # Create a dataframe from the results
    df = pd.DataFrame(results)
    
    # Sort by sentiment score
    df = df.sort_values("sentiment_score")
    
    # Create a colormap
    colors = ["red" if score < 0 else "green" for score in df["sentiment_score"]]
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df)), df["sentiment_score"], color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel("Text Index")
    plt.ylabel("Sentiment Score (-1 to 1)")
    plt.title("Sentiment Analysis Results")
    
    # Add text labels (shortened)
    shortened_texts = [t[:30] + "..." if len(t) > 30 else t for t in df["text"]]
    plt.xticks(range(len(df)), shortened_texts, rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig("sentiment_results.png")
    print(f"Results visualization saved to sentiment_results.png")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Sentiment Analysis API Client")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch", "file"],
                      default="interactive", help="Mode of operation")
    parser.add_argument("--file", type=str, help="File with texts to analyze (one per line)")
    args = parser.parse_args()
    
    # Get model info
    print("Getting model information...")
    model_info = get_model_info()
    if model_info:
        print(f"Model type: {model_info['model_type']}")
        if 'metrics' in model_info and model_info['metrics']:
            print(f"Model accuracy: {model_info['metrics'].get('accuracy', 'N/A')}")
            print(f"Model F1 score: {model_info['metrics'].get('f1_score', 'N/A')}")
    
    if args.mode == "interactive":
        # Interactive mode
        print("\nEnter texts to analyze (press Ctrl+C to exit):")
        results = []
        
        try:
            while True:
                text = input("\nText: ")
                if not text:
                    continue
                
                result = predict_sentiment(text)
                if result:
                    results.append(result)
                    print(f"Sentiment: {result['sentiment_label']} (Score: {result['sentiment_score']:.2f}, Confidence: {result['confidence']:.2f})")
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
        
        # Visualize results if we have any
        if results:
            visualize_results(results)
    
    elif args.mode == "batch":
        # Batch mode with example texts
        example_texts = [
            "I love this product! It's amazing!",
            "This is absolutely terrible, would not recommend.",
            "It's okay, nothing special but it works.",
            "Having a great day with friends!",
            "Worst experience ever. Never going back there.",
            "The service was excellent, very attentive staff.",
            "Disappointed with the quality, not worth the money.",
            "Neutral opinion, neither good nor bad.",
            "Couldn't be happier with my purchase!",
            "Frustrating experience, wasted my time and money."
        ]
        
        print("\nAnalyzing example texts...")
        results = predict_batch_sentiment(example_texts)
        
        if results:
            # Create a table of results
            table_data = []
            for result in results:
                table_data.append([
                    result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"],
                    result["sentiment_label"],
                    f"{result['sentiment_score']:.2f}",
                    f"{result['confidence']:.2f}"
                ])
            
            print("\nResults:")
            print(tabulate(table_data, headers=["Text", "Sentiment", "Score", "Confidence"], tablefmt="grid"))
            
            # Visualize results
            visualize_results(results)
    
    elif args.mode == "file" and args.file:
        # File mode
        try:
            with open(args.file, "r") as f:
                texts = [line.strip() for line in f if line.strip()]
            
            if not texts:
                print(f"No texts found in file: {args.file}")
                return
            
            print(f"\nAnalyzing {len(texts)} texts from file: {args.file}")
            results = predict_batch_sentiment(texts)
            
            if results:
                # Create a table of results
                table_data = []
                for result in results:
                    table_data.append([
                        result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"],
                        result["sentiment_label"],
                        f"{result['sentiment_score']:.2f}",
                        f"{result['confidence']:.2f}"
                    ])
                
                print("\nResults:")
                print(tabulate(table_data, headers=["Text", "Sentiment", "Score", "Confidence"], tablefmt="grid"))
                
                # Visualize results
                visualize_results(results)
        except FileNotFoundError:
            print(f"File not found: {args.file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 