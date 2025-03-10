"""
WITHIN ML Prediction System

Main application entry point that configures FastAPI, middleware, and routes.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import middleware
from app.api.v1.middleware.ml_context import ml_context_middleware

# Import routes
from app.api.v1.routes.ml_routes import router as ml_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI application
app = FastAPI(
    title="WITHIN ML Prediction System",
    description="A model-driven machine learning prediction system with data validation layers",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register middleware
app.middleware("http")(ml_context_middleware)

# Include routers
app.include_router(ml_router)


@app.get("/")
async def root():
    """Root endpoint that provides basic API information."""
    return {
        "name": "WITHIN ML Prediction System",
        "version": "1.0.0",
        "docs_url": "/api/docs",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and deployment verification."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 