# Development Setup Guide

**IMPLEMENTATION STATUS: IMPLEMENTED**


This guide provides detailed instructions for setting up a development environment for the WITHIN Ad Score & Account Health Predictor system. Follow these steps to ensure you have all the necessary dependencies and configurations for effective development.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Repository Setup](#repository-setup)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Infrastructure Setup](#infrastructure-setup)
- [Verification](#verification)
- [IDE Setup](#ide-setup)
- [Docker Environment](#docker-environment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Ensure you have the following installed on your system:

### Required Software

- **Python 3.9+**: Required for all development activities
- **Git**: For version control
- **Docker**: For containerized development and testing
- **Docker Compose**: For multi-container Docker applications
- **PostgreSQL 13+**: For local database development
- **Node.js 16+**: For frontend development tools
- **CUDA Toolkit 11.6+** (if using GPU): For GPU-accelerated model training

### Operating System Recommendations

- **Linux**: Ubuntu 20.04 LTS or later (recommended for production-like development)
- **macOS**: macOS 11 (Big Sur) or later
- **Windows**: Windows 10/11 with WSL2 (Windows Subsystem for Linux)

### Python Knowledge

Familiarity with the following Python concepts and libraries is recommended:

- Python type hints
- asyncio and asynchronous programming
- FastAPI
- SQLAlchemy
- PyTorch
- scikit-learn
- Pandas and NumPy

## Repository Setup

### Cloning the Repository

```bash
# Clone the repository
git clone https://github.com/within-company/within.git
cd within

# Initialize and update submodules
git submodule init
git submodule update
```

### Repository Structure

The repository is organized into the following main directories:

- `app/`: Main application code
  - `api/`: API endpoints and routes
  - `core/`: Core utilities and base classes
  - `models/`: Data models and ML models
  - `etl/`: Data extraction, transformation, and loading
  - `nlp/`: Natural language processing components
  - `monitoring/`: Monitoring and alerting
- `config/`: Configuration files
- `scripts/`: Utility scripts
- `tests/`: Test suite
- `docs/`: Documentation
- `infrastructure/`: Infrastructure as code
- `notebooks/`: Jupyter notebooks for exploration

## Environment Setup

### Python Environment

We recommend using a virtual environment for development:

#### Option 1: Using venv

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

#### Option 2: Using conda

```bash
# Create a conda environment
conda create -n within python=3.9
conda activate within

# Install development dependencies
pip install -r requirements-dev.txt
```

### Installing Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Set up the git hooks
pre-commit install

# Run the hooks against all files
pre-commit run --all-files
```

### GPU Setup (Optional)

For GPU-accelerated model training:

```bash
# Install PyTorch with CUDA support
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html

# Verify GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Database connection
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/within
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/within_test

# API settings
API_PORT=8000
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Authentication
SECRET_KEY=yoursecretkey
AUTH_TOKEN_EXPIRE_MINUTES=60

# External services
OPENAI_API_KEY=your_openai_api_key  # Only if using OpenAI integrations
GOOGLE_CLOUD_PROJECT=your_gcp_project  # Only if using GCP

# Model settings
MODEL_REGISTRY_PATH=./model_registry
MODEL_CACHE_SIZE=2  # GB
DEFAULT_MODEL_VERSION=latest

# Feature flags
ENABLE_EXPERIMENTAL_FEATURES=true
ENABLE_GPU_ACCELERATION=false
```

### Application Configuration

The application configuration is stored in `config/settings.py`. You can override these settings by creating a `config/local_settings.py` file:

```python
# config/local_settings.py
from config.settings import *

# Override settings here
DEBUG = True
LOG_LEVEL = "DEBUG"
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/within"

# Add custom settings
CUSTOM_SETTING = "value"
```

### Data Source Configuration

Configure access to advertising platforms for development:

1. Create a `config/datasources.json` file:

```json
{
  "facebook": {
    "app_id": "your_app_id",
    "app_secret": "your_app_secret",
    "access_token": "your_access_token"
  },
  "google": {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "refresh_token": "your_refresh_token",
    "developer_token": "your_developer_token"
  },
  "tiktok": {
    "app_id": "your_app_id",
    "app_secret": "your_app_secret",
    "access_token": "your_access_token"
  }
}
```

2. For secure credential management, consider using environment variables or a secret manager.

## Infrastructure Setup

### Database Setup

Set up a local PostgreSQL database:

```bash
# Start PostgreSQL container
docker run --name within-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=within -p 5432:5432 -d postgres:13

# Verify connection
psql -h localhost -U postgres -d within -c "SELECT 1"
```

### Database Migrations

Run migrations to create the database schema:

```bash
# Apply migrations
python -m scripts.alembic upgrade head

# Verify migrations
python -m scripts.alembic current
```

### Redis Setup (Optional)

Set up Redis for caching and job queue:

```bash
# Start Redis container
docker run --name within-redis -p 6379:6379 -d redis:6

# Verify connection
redis-cli ping
```

### MinIO Setup (Optional)

Set up MinIO for S3-compatible object storage:

```bash
# Start MinIO container
docker run --name within-minio -p 9000:9000 -p 9001:9001 -e "MINIO_ROOT_USER=minioadmin" -e "MINIO_ROOT_PASSWORD=minioadmin" -d minio/minio server /data --console-address ":9001"

# Create buckets
mc alias set within-local http://localhost:9000 minioadmin minioadmin
mc mb within-local/within-models
mc mb within-local/within-data
```

## Verification

Verify your setup by running tests and starting the development server:

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/api/
pytest tests/models/

# Run with coverage
pytest --cov=app --cov-report=html
```

### Starting the Development Server

```bash
# Start the development server
python -m app.main

# Verify API access
curl http://localhost:8000/api/v1/health
```

### Verifying ML Models

```bash
# Verify model loading
python -c "from app.models.ml.prediction import AdScorePredictor; model = AdScorePredictor(); print(f'Model loaded: {model}')"

# Run a sample prediction
python scripts/sample_prediction.py
```

## IDE Setup

### VS Code

For VS Code, create a `.vscode/settings.json` file:

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.nosetestsEnabled": false,
  "python.testing.pytestArgs": ["tests"],
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.autoImportCompletions": true,
  "python.sortImports.args": ["--profile", "black"],
  "python.envFile": "${workspaceFolder}/.env"
}
```

### PyCharm

For PyCharm:

1. Open the project in PyCharm
2. Go to File > Settings > Project > Python Interpreter
3. Configure the virtual environment
4. Go to File > Settings > Tools > Python Integrated Tools
   - Set "Default test runner" to "pytest"
5. Go to File > Settings > Editor > Code Style > Python
   - Set line length to 100
   - Enable "Optimize imports on the fly"
6. Install and configure plugins:
   - Black Formatter
   - Mypy
   - Pylint

## Docker Environment

For containerized development, use Docker Compose:

### Docker Compose Setup

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/within
      - REDIS_URL=redis://redis:6379/0
      - ENVIRONMENT=development
      - DEBUG=true
    depends_on:
      - db
      - redis
    command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=within
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6
    ports:
      - "6379:6379"

  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  minio_data:
```

### Development Dockerfile

Create a `Dockerfile.dev`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### Running with Docker Compose

```bash
# Build and start containers
docker-compose up -d

# Run migrations in Docker
docker-compose exec app python -m scripts.alembic upgrade head

# Run tests in Docker
docker-compose exec app pytest

# Stop containers
docker-compose down
```

## Troubleshooting

### Common Issues

#### Database Connection Issues

```
Error: could not connect to server: Connection refused
```

**Solution**:
1. Verify PostgreSQL is running: `docker ps | grep postgres`
2. Check PostgreSQL logs: `docker logs within-postgres`
3. Verify connection details in `.env` file

#### Package Import Errors

```
ModuleNotFoundError: No module named 'app'
```

**Solution**:
1. Ensure you're in the project root directory
2. Add the project root to PYTHONPATH:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```
3. Verify virtual environment is activated

#### Pre-commit Hook Failures

```
black....................................................................Failed
```

**Solution**:
1. Run black manually: `black .`
2. Run pre-commit manually: `pre-commit run --all-files`
3. Fix reported issues

#### GPU Issues

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**:
1. Verify CUDA installation: `nvcc --version`
2. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.version.cuda)"`
3. Reinstall PyTorch with matching CUDA version

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [FAQ](/docs/development/faq.md)
2. Search existing GitHub issues
3. Ask in the #development Slack channel
4. Contact the engineering team at [dev@within.co](mailto:dev@within.co)

## Next Steps

After completing the setup, you may want to:

1. Review the [Architecture Overview](/docs/architecture/overview.md)
2. Follow the [API Development Guide](/docs/development/api_development.md)
3. Read the [ML Development Guide](/docs/development/ml_development.md)
4. Explore the [Coding Standards](/docs/development/coding_standards.md)
5. Set up your [CI/CD Pipeline](/docs/development/ci_cd.md) 