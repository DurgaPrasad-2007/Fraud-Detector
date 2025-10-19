"""
Configuration Management - 2025 Production Ready
Centralized configuration with environment support
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass
from loguru import logger

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "fraud_detection"
    user: str = "postgres"
    password: str = "durga"

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

@dataclass
class ModelConfig:
    """ML Model configuration"""
    ensemble_weights: Dict[str, float] = None
    feature_columns: list = None
    fraud_threshold: float = 0.5
    batch_size: int = 512
    max_features: int = 15
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'xgboost': 0.3,
                'lightgbm': 0.3,
                'catboost': 0.2,
                'neural_network': 0.15,
                'random_forest': 0.05
            }
        
        if self.feature_columns is None:
            self.feature_columns = [
                'TX_AMOUNT', 'hour_of_day', 'day_of_week', 'is_weekend',
                'amount_log', 'customer_avg_amount', 'customer_transaction_count',
                'customer_fraud_rate', 'customer_amount_std', 'terminal_avg_amount',
                'terminal_transaction_count', 'terminal_fraud_rate',
                'days_since_last_transaction', 'transactions_today', 'amount_vs_avg_ratio'
            ]

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    prometheus_port: int = 9090
    log_level: str = "INFO"
    alert_thresholds: Dict[str, float] = None
    metrics_retention_days: int = 30
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'fraud_rate': 0.15,
                'processing_time_ms': 1000,
                'error_rate': 0.05,
                'cpu_usage': 80.0,
                'memory_usage': 85.0
            }

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: list = None
    rate_limit: int = 1000
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config_data = {}
        self._load_config()
        
        # Initialize sub-configurations
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.model = ModelConfig()
        self.monitoring = MonitoringConfig()
        self.api = APIConfig()
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self._config_data = yaml.safe_load(f) or {}
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
                self._config_data = {}
        else:
            logger.info("No config file found, using defaults")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            'DB_HOST': ('database', 'host'),
            'DB_PORT': ('database', 'port'),
            'DB_NAME': ('database', 'name'),
            'DB_USER': ('database', 'user'),
            'DB_PASSWORD': ('database', 'password'),
            'REDIS_HOST': ('redis', 'host'),
            'REDIS_PORT': ('redis', 'port'),
            'API_HOST': ('api', 'host'),
            'API_PORT': ('api', 'port'),
            'LOG_LEVEL': ('monitoring', 'log_level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if key in ['port', 'workers', 'rate_limit', 'metrics_retention_days']:
                    value = int(value)
                elif key in ['fraud_threshold', 'batch_size', 'max_features']:
                    value = float(value) if '.' in value else int(value)
                elif key in ['reload']:
                    value = value.lower() in ('true', '1', 'yes')
                
                setattr(getattr(self, section), key, value)
                logger.info(f"Override: {env_var} -> {section}.{key} = {value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'name': self.database.name,
                'user': self.database.user,
                'password': '***' if self.database.password else None
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'db': self.redis.db
            },
            'model': {
                'ensemble_weights': self.model.ensemble_weights,
                'feature_columns': self.model.feature_columns,
                'fraud_threshold': self.model.fraud_threshold,
                'batch_size': self.model.batch_size,
                'max_features': self.model.max_features
            },
            'monitoring': {
                'prometheus_port': self.monitoring.prometheus_port,
                'log_level': self.monitoring.log_level,
                'alert_thresholds': self.monitoring.alert_thresholds,
                'metrics_retention_days': self.monitoring.metrics_retention_days
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'workers': self.api.workers,
                'reload': self.api.reload,
                'cors_origins': self.api.cors_origins,
                'rate_limit': self.api.rate_limit
            }
        }

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance"""
    return config

if __name__ == "__main__":
    # Test configuration
    cfg = Config()
    print("Configuration loaded successfully!")
    print(f"API Host: {cfg.api.host}:{cfg.api.port}")
    print(f"Model Features: {len(cfg.model.feature_columns)}")
    print(f"Ensemble Weights: {cfg.model.ensemble_weights}")

