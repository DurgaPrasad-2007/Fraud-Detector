"""
ML Models Module - 2025 Production Ready
Clean, efficient ensemble models with explainable AI
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from loguru import logger

from ..utils.config import get_config

class EnsembleModel:
    """Advanced ensemble fraud detection model"""
    
    def __init__(self):
        self.config = get_config()
        self.models = {}
        self.scalers = {}
        self.feature_columns = self.config.model.feature_columns
        self.ensemble_weights = self.config.model.ensemble_weights
        
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train ensemble models
        
        TODO: Add cross-validation
        TODO: Implement hyperparameter tuning
        TODO: Add early stopping
        """
        
        logger.info("Training ensemble models...")
        
        # Split data - TODO: consider time-based split for fraud detection
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['robust'] = scaler
        
        # Check for GPU availability for tree-based models
        gpus = tf.config.list_physical_devices('GPU')
        use_gpu = len(gpus) > 0
        
        logger.info(f"GPU available for tree models: {use_gpu}")
        
        # Initialize models with GPU support where available
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,  # TODO: tune this
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                tree_method='gpu_hist' if use_gpu else 'hist',  # Use GPU if available
                gpu_id=0 if use_gpu else None
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                device='gpu' if use_gpu else 'cpu',  # Use GPU if available
                gpu_platform_id=0 if use_gpu else None,
                gpu_device_id=0 if use_gpu else None
            ),
            'catboost': CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False,
                task_type='GPU' if use_gpu else 'CPU',  # Use GPU if available
                devices='0' if use_gpu else None
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1  # Random Forest doesn't support GPU in sklearn
            )
        }
        
        # Train models
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'auc': auc_score,
                'predictions': y_pred_proba
            }
            
            logger.info(f"{name} AUC: {auc_score:.4f}")
        
        self.models.update({name: results[name]['model'] for name in results})
        
        # Train neural network
        nn_results = self._train_neural_network(X_train, X_test, y_train, y_test)
        results.update(nn_results)
        
        logger.info("Model training completed!")
        return results, X_test, y_test
    
    def _train_neural_network(self, X_train: np.ndarray, X_test: np.ndarray, 
                            y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train neural network with GPU support"""
        
        logger.info("Training neural network...")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            logger.warning("TensorFlow not available, skipping neural network training")
            logger.info("Using only GPU-accelerated tree-based models (XGBoost, LightGBM, CatBoost)")
            return {
                'neural_network': {
                    'model': None,
                    'auc': 0.0,
                    'predictions': None
                }
            }
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU detected: {len(gpus)} GPU(s) available")
                logger.info(f"GPU name: {gpus[0].name}")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")
        else:
            logger.info("No GPU detected, using CPU")
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Build neural network with GPU-optimized layers
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),  # Use Input layer instead of input_shape
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model with GPU-optimized parameters
        batch_size = min(self.config.model.batch_size, 1024)  # Cap batch size for GPU memory
        
        logger.info(f"Training with batch size: {batch_size}")
        if gpus:
            logger.info("Using GPU acceleration")
        else:
            logger.info("Using CPU (no GPU detected)")
        
        try:
            history = model.fit(
                X_train_scaled, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1  # Show progress
            )
            
            # Evaluate
            y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            self.models['neural_network'] = model
            
            logger.info(f"Neural Network AUC: {auc_score:.4f}")
            
            return {
                'neural_network': {
                    'model': model,
                    'auc': auc_score,
                    'predictions': y_pred_proba
                }
            }
            
        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            logger.info("Continuing with GPU-accelerated tree-based models only")
            return {
                'neural_network': {
                    'model': None,
                    'auc': 0.0,
                    'predictions': None
                }
            }
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Create ensemble prediction"""
        
        predictions = []
        
        for name, model in self.models.items():
            if model is None:  # Skip if model is None (e.g., neural network failed)
                continue
                
            if name == 'neural_network':
                X_scaled = self.scalers['standard'].transform(X)
                pred_proba = model.predict(X_scaled, verbose=0).flatten()
            else:
                pred_proba = model.predict_proba(X)[:, 1]
            
            predictions.append(pred_proba)
        
        # Weighted ensemble
        weights = [self.ensemble_weights.get(name, 0.1) for name in self.models.keys()]
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def predict_single(self, features: np.ndarray) -> Dict[str, float]:
        """Predict fraud probability for single transaction"""
        
        features = features.reshape(1, -1)
        ensemble_prob = self.predict_ensemble(features)[0]
        
        # Individual model predictions
        individual_predictions = {}
        for name, model in self.models.items():
            if name == 'neural_network':
                features_scaled = self.scalers['standard'].transform(features)
                pred_proba = model.predict(features_scaled, verbose=0)[0][0]
            else:
                pred_proba = model.predict_proba(features)[0][1]
            
            individual_predictions[name] = float(pred_proba)
        
        return {
            'ensemble_probability': float(ensemble_prob),
            'individual_predictions': individual_predictions,
            'is_fraud': ensemble_prob > self.config.model.fraud_threshold
        }
    
    def save_models(self, filepath: str = "data/models/"):
        """Save trained models"""
        
        Path(filepath).mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(f'{filepath}{name}_model.h5')
            else:
                joblib.dump(model, f'{filepath}{name}_model.pkl')
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{filepath}{name}_scaler.pkl')
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str = "data/models/"):
        """Load trained models"""
        
        model_path = Path(filepath)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Models directory not found: {filepath}")
        
        # Load tree-based models
        for model_name in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
            model_file = model_path / f"{model_name}_model.pkl"
            if model_file.exists():
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded {model_name} model")
        
        # Load neural network
        nn_file = model_path / "neural_network_model.h5"
        if nn_file.exists():
            self.models['neural_network'] = tf.keras.models.load_model(nn_file)
            logger.info("Loaded neural network model")
        
        # Load scalers
        for scaler_name in ['robust', 'standard']:
            scaler_file = model_path / f"{scaler_name}_scaler.pkl"
            if scaler_file.exists():
                self.scalers[scaler_name] = joblib.load(scaler_file)
                logger.info(f"Loaded {scaler_name} scaler")
        
        logger.info("All models loaded successfully")

# Global model instance
ensemble_model = EnsembleModel()

def get_model() -> EnsembleModel:
    """Get global model instance"""
    return ensemble_model
