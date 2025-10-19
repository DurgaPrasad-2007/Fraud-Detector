"""
Main Application - 2025 Production Ready
Clean, efficient application orchestrator
"""

import uvicorn
import sys
import argparse
from pathlib import Path
from loguru import logger

from src.utils.config import get_config
from src.data.preprocessing import get_preprocessor
from src.models.ensemble import get_model

def setup_logging():
    """Setup structured logging"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )

def create_directories():
    """Create necessary directories"""
    directories = ['data/raw', 'data/processed', 'data/models', 'logs', 'config']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Directories created successfully")

def load_and_process_dataset():
    """Load and process the real dataset"""
    try:
        preprocessor = get_preprocessor()
        
        # Load the real dataset
        df = preprocessor.load_real_dataset()
        
        # Engineer features
        df_features = preprocessor.engineer_features(df)
        
        # Save processed dataset
        preprocessor.save_dataset(df_features)
        
        logger.info("Dataset loaded and processed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load and process dataset: {e}")
        return False

def train_models():
    """Train ML models"""
    try:
        preprocessor = get_preprocessor()
        model = get_model()
        
        # Load dataset
        df = preprocessor.load_dataset()
        X, y = preprocessor.prepare_training_data(df)
        
        # Train models
        results, X_test, y_test = model.train_models(X, y)
        
        # Save models
        model.save_models()
        
        logger.info("Models trained and saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to train models: {e}")
        return False

def start_api_server():
    """Start the API server"""
    try:
        config = get_config()
        
        logger.info("🚀 Starting Fraud Detection System...")
        logger.info(f"📊 Dashboard: http://{config.api.host}:{config.api.port}")
        logger.info(f"📚 API Docs: http://{config.api.host}:{config.api.port}/docs")
        logger.info(f"📈 Metrics: http://{config.api.host}:{config.api.port}/metrics")
        
        uvicorn.run(
            "src.api.endpoints:app",
            host=config.api.host,
            port=config.api.port,
            reload=config.api.reload,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)

def dev_run():
    """Development run - complete project startup with models"""
    logger.info("🚀 Starting Fraud Detection System - Development Mode")
    
    # Setup logging
    setup_logging()
    
    # Create directories
    create_directories()
    
    # Check if dataset exists, load and process if not
    if not Path("data/processed/fraud_dataset.csv").exists():
        logger.info("📊 Dataset not found, loading and processing real dataset...")
        load_and_process_dataset()
    else:
        logger.info("📊 Processed dataset found, using existing data")
    
    # Check if models exist, train if not
    models_dir = Path("data/models")
    if not models_dir.exists() or not any(models_dir.iterdir()):
        logger.info("🤖 Models not found, training ensemble models...")
        train_models()
    else:
        logger.info("🤖 Models found, using existing models")
    
    # Start API server
    logger.info("🌐 Starting API server...")
    start_api_server()

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Fraud Detection System 2025")
    parser.add_argument("--mode", choices=["setup", "train", "serve", "full"], 
                       default="full", help="Application mode")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("🛡️ Fraud Detection System 2025 - Starting...")
    
    if args.mode == "setup":
        logger.info("Running setup mode...")
        create_directories()
        load_and_process_dataset()
        train_models()
        logger.info("Setup completed successfully!")
        
    elif args.mode == "train":
        logger.info("Running training mode...")
        create_directories()
        if not Path("data/processed/fraud_dataset.csv").exists():
            load_and_process_dataset()
        train_models()
        logger.info("Training completed successfully!")
        
    elif args.mode == "serve":
        logger.info("Running serve mode...")
        start_api_server()
        
    elif args.mode == "full":
        logger.info("Running full mode...")
        
        # Setup
        create_directories()
        
        # Check if dataset exists
        if not Path("data/processed/fraud_dataset.csv").exists():
            logger.info("Dataset not found, loading and processing...")
            load_and_process_dataset()
        
        # Check if models exist
        models_dir = Path("data/models")
        if not models_dir.exists() or not any(models_dir.iterdir()):
            logger.info("Models not found, training...")
            train_models()
        
        # Start API server
        start_api_server()

if __name__ == "__main__":
    main()