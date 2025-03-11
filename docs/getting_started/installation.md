# Installation Guide

**IMPLEMENTATION STATUS: IMPLEMENTED**


This guide provides detailed instructions for installing and configuring the WITHIN Ad Score & Account Health Predictor system.

## Prerequisites

### System Requirements

- **Operating System**: 
  - Linux (Ubuntu 20.04+ or Amazon Linux 2 recommended)
  - macOS 11+ (Big Sur or newer)
  - Windows 10/11 with WSL2 (for development only)

- **Hardware**:
  - **Minimum**: 4 CPU cores, 16GB RAM, 100GB storage
  - **Recommended**: 8+ CPU cores, 32GB+ RAM, 250GB+ SSD storage
  - **Production**: 16+ CPU cores, 64GB+ RAM, 500GB+ SSD storage
  - **GPU** (optional): NVIDIA GPU with 8GB+ VRAM for training acceleration

- **Software**:
  - Python 3.9 or higher
  - Docker and Docker Compose
  - PostgreSQL 13+
  - Redis 6+
  - Node.js 16+ (for dashboard frontend)

### Cloud Provider Requirements

If deploying to cloud providers, you'll need:

- **AWS**: Account with permissions for EC2, S3, RDS, ElastiCache, and IAM
- **GCP**: Account with permissions for Compute Engine, Cloud Storage, Cloud SQL, and IAM
- **Azure**: Account with permissions for Virtual Machines, Storage, Database, and IAM

### Developer Tools

For development environments:

- Git
- Python virtual environment tool (virtualenv, conda, or similar)
- Code editor (VS Code recommended with Python extensions)
- Docker Desktop
- Database client for PostgreSQL

## Installation Methods

Choose one of the following installation methods:

1. **Docker Deployment** (Recommended): Using pre-built Docker images
2. **Manual Installation**: Installing components manually
3. **Cloud Deployment**: Deploying to AWS, GCP, or Azure

## Docker Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/WITHIN.git
cd WITHIN
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```
# Database Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=within_db
DB_USER=within_user
DB_PASSWORD=your_secure_password

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# API Configuration
API_PORT=8000
SECRET_KEY=your_secure_secret_key
ENVIRONMENT=production

# ML Model Configuration
MODEL_REGISTRY_PATH=/app/models/registry
ENABLE_GPU=false
```

### 3. Configure Settings

Copy and edit the example settings file:

```bash
cp config/settings.example.yaml config/settings.yaml
```

Configure the settings according to your requirements.

### 4. Start the Services

Using Docker Compose:

```bash
docker-compose up -d
```

This will start all required services:

- API Server
- Database (PostgreSQL)
- Cache (Redis)
- Worker Processes
- Dashboard (if enabled)

### 5. Run Database Migrations

```bash
docker-compose exec api alembic upgrade head
```

### 6. Create an Admin User

```bash
docker-compose exec api python -m scripts.create_admin_user \
  --username admin \
  --email admin@example.com \
  --password secure_password
```

### 7. Verify Installation

Access the API documentation at:

```
http://localhost:8000/docs
```

Access the admin dashboard at (if enabled):

```
http://localhost:3000
```

## Manual Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/WITHIN.git
cd WITHIN
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up Database

Install PostgreSQL and create database:

```bash
# Example for Ubuntu
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE within_db;
CREATE USER within_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE within_db TO within_user;
\q
```

### 4. Set Up Redis

Install and start Redis:

```bash
# Example for Ubuntu
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=within_db
DB_USER=within_user
DB_PASSWORD=your_secure_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# API Configuration
API_PORT=8000
SECRET_KEY=your_secure_secret_key
ENVIRONMENT=production

# ML Model Configuration
MODEL_REGISTRY_PATH=./models/registry
ENABLE_GPU=false
```

### 6. Configure Settings

Copy and edit the example settings file:

```bash
cp config/settings.example.yaml config/settings.yaml
```

### 7. Run Database Migrations

```bash
alembic upgrade head
```

### 8. Create an Admin User

```bash
python -m scripts.create_admin_user \
  --username admin \
  --email admin@example.com \
  --password secure_password
```

### 9. Start the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 10. Start Worker Processes (in a separate terminal)

```bash
celery -A app.worker worker --loglevel=info
```

### 11. Start the Dashboard (Optional)

```bash
cd dashboard
npm install
npm run build
npm run start
```

## Cloud Deployment

### AWS Deployment

Detailed AWS deployment instructions are available in [AWS Deployment Guide](cloud/aws_deployment.md).

### GCP Deployment

Detailed GCP deployment instructions are available in [GCP Deployment Guide](cloud/gcp_deployment.md).

### Azure Deployment

Detailed Azure deployment instructions are available in [Azure Deployment Guide](cloud/azure_deployment.md).

## Post-Installation Steps

### 1. Download ML Models

```bash
# Using Docker
docker-compose exec api python -m scripts.download_models

# Manual Installation
python -m scripts.download_models
```

### 2. Verify Model Availability

```bash
# Using Docker
docker-compose exec api python -m scripts.verify_models

# Manual Installation
python -m scripts.verify_models
```

### 3. Configure External Integrations

Follow the [Data Integration Guide](../user_guides/data_integration.md) to set up connections to advertising platforms.

### 4. Set Up Monitoring (Production)

For production environments, set up monitoring as described in the [Monitoring Guide](../maintenance/monitoring_guide.md).

## Configuration Options

### Database Configuration

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| Host | `DB_HOST` | localhost | Database server hostname |
| Port | `DB_PORT` | 5432 | Database server port |
| Name | `DB_NAME` | within_db | Database name |
| User | `DB_USER` | within_user | Database username |
| Password | `DB_PASSWORD` | | Database password |
| SSL Mode | `DB_SSL_MODE` | prefer | SSL mode for database connection |

### API Configuration

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| Host | `API_HOST` | 0.0.0.0 | API server host |
| Port | `API_PORT` | 8000 | API server port |
| Workers | `API_WORKERS` | 4 | Number of worker processes |
| Debug | `API_DEBUG` | false | Enable debug mode |
| Secret Key | `SECRET_KEY` | | Secret key for JWT encoding |
| Token Expiry | `TOKEN_EXPIRY_MINUTES` | 60 | JWT token expiry time in minutes |

### ML Model Configuration

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| Registry Path | `MODEL_REGISTRY_PATH` | ./models/registry | Path to model registry |
| Enable GPU | `ENABLE_GPU` | false | Enable GPU for model inference |
| Default Ad Score Model | `DEFAULT_AD_SCORE_MODEL` | ad_score_v2.1.0 | Default model for ad scoring |
| Default Account Health Model | `DEFAULT_ACCOUNT_HEALTH_MODEL` | account_health_v1.5.0 | Default model for account health |
| Cache Predictions | `CACHE_PREDICTIONS` | true | Enable prediction caching |
| Cache TTL | `PREDICTION_CACHE_TTL` | 3600 | Cache time-to-live in seconds |

## Troubleshooting

### Common Issues

#### Database Connection Issues

```
Error: could not connect to server: Connection refused
```

**Solution**:
- Verify PostgreSQL is running: `sudo systemctl status postgresql`
- Check database credentials in `.env` file
- Verify network connectivity to database server

#### Redis Connection Issues

```
Error: Error connecting to Redis: Connection refused
```

**Solution**:
- Verify Redis is running: `sudo systemctl status redis-server`
- Check Redis configuration in `.env` file
- Verify network connectivity to Redis server

#### Model Loading Issues

```
Error: Could not find model file: ad_score_v2.1.0
```

**Solution**:
- Run the model download script: `python -m scripts.download_models`
- Check `MODEL_REGISTRY_PATH` in configuration
- Verify disk space availability

#### API Server Won't Start

```
Error: Address already in use
```

**Solution**:
- Check if another process is using the port: `sudo lsof -i :8000`
- Change the port in configuration
- Stop the conflicting process

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting Guide](../maintenance/troubleshooting.md) for more specific solutions
2. Search or ask in the [Community Forum](https://community.within.co)
3. Open an issue on [GitHub](https://github.com/your-org/WITHIN/issues)
4. Contact support at support@within.co

## Next Steps

- [Quick Start Tutorial](quick_start.md): Learn how to use the system
- [API Reference](../api/overview.md): Explore the API endpoints
- [Dashboard Guide](../user_guides/dashboards.md): Learn how to use the dashboards
- [Data Integration Guide](../user_guides/data_integration.md): Set up data sources 