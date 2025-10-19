# Fraud Detection System

A fraud detection system using machine learning to identify suspicious transactions in real-time.

## Overview

This project implements a fraud detection system that analyzes transaction patterns to identify potentially fraudulent activities. It uses ensemble machine learning models to make predictions and provides explanations for its decisions.

## Features

- Real-time fraud prediction
- Multiple ML models (XGBoost, LightGBM, CatBoost, Neural Networks)
- Model explainability with SHAP and LIME
- REST API for integration
- Basic monitoring and logging

## Architecture

```
Data → Features → Models → API → Response
```

## Technology Stack

- Python 3.11
- FastAPI for the API
- XGBoost, LightGBM, CatBoost for ML models
- TensorFlow for neural networks
- SHAP & LIME for explanations
- Docker for deployment

## Getting Started

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)

### Installation
```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run the system
poetry run devrun
```

### What the system does
1. Generates synthetic fraud transaction data
2. Engineers features from transaction data
3. Trains ensemble ML models
4. Provides real-time fraud predictions via API

## Data Processing Pipeline

### Data Generation
The system creates synthetic fraud transactions with three fraud scenarios:
- Transactions over $220 are marked as fraud
- Random terminals become compromised for 28 days
- Random customers have 1/3 of transactions multiplied by 5

### Feature Engineering
Creates features like:
- Transaction amount and time-based features
- Customer behavior patterns
- Terminal usage statistics
- Advanced derived features

### Model Training
Trains multiple models and combines them:
- XGBoost (30%)
- LightGBM (30%) 
- CatBoost (20%)
- Neural Network (15%)
- Random Forest (5%)

## API Usage

### Single Transaction Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TX_12345678",
    "customer_id": "CUST_000001",
    "terminal_id": "TERM_000001",
    "tx_amount": 150.00,
    "tx_datetime": "2023-12-01T10:30:00Z"
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Testing

Run the test suite:
```bash
poetry run pytest tests/ -v
```

Run a simple system test:
```bash
python test_system.py
```

## Development

### Code Quality
```bash
# Format code
poetry run black src/
poetry run isort src/

# Lint code
poetry run flake8 src/

# Type checking
poetry run mypy src/

# Security scan
poetry run safety check
poetry run bandit -r src/
```

### Pre-commit Hooks
```bash
poetry run pre-commit install
```

## Docker Deployment

### Single Container
```bash
docker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector
```

### Multi-Service
```bash
docker-compose up -d
```

## Project Structure

```
fraud-detector/
├── src/                          # Source code
│   ├── data/                     # Data processing
│   ├── models/                   # ML models
│   ├── api/                      # API endpoints
│   ├── monitoring/               # Monitoring
│   ├── mlops/                    # MLOps components
│   └── utils/                    # Utilities
├── config/                       # Configuration files
├── tests/                        # Test suite
├── pyproject.toml               # Poetry configuration
├── Dockerfile                   # Container config
└── main.py                      # Application entry point
```

## Known Issues

- Models need to be trained before first use
- No authentication on API endpoints (TODO)
- Limited input validation (TODO)
- No rate limiting implemented (TODO)
- Hardcoded configuration values (TODO)

## TODO

- [ ] Add proper authentication
- [ ] Implement rate limiting
- [ ] Add more comprehensive tests
- [ ] Improve error handling
- [ ] Add model versioning
- [ ] Implement proper monitoring
- [ ] Add database integration
- [ ] Implement caching

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.