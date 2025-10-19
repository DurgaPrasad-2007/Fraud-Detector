# MLflow Configuration for Model Registry and Experiment Tracking

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import os
from pathlib import Path

class MLflowManager:
    """MLflow manager for experiment tracking and model registry"""
    
    def __init__(self):
        # Set MLflow tracking URI
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set experiment name
        self.experiment_name = "fraud-detection-system"
        mlflow.set_experiment(self.experiment_name)
        
        self.client = MlflowClient()
    
    def start_run(self, run_name: str = None):
        """Start a new MLflow run"""
        return mlflow.start_run(run_name=run_name)
    
    def log_model_metrics(self, model_name: str, metrics: dict):
        """Log model metrics"""
        mlflow.log_metrics(metrics)
        mlflow.set_tag("model_name", model_name)
    
    def log_model_parameters(self, params: dict):
        """Log model parameters"""
        mlflow.log_params(params)
    
    def log_model(self, model, model_name: str, signature=None):
        """Log trained model"""
        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, "model", signature=signature)
        elif model_name == "lightgbm":
            mlflow.lightgbm.log_model(model, "model", signature=signature)
        elif model_name == "catboost":
            mlflow.catboost.log_model(model, "model", signature=signature)
        elif model_name == "neural_network":
            mlflow.tensorflow.log_model(model, "model", signature=signature)
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)
    
    def register_model(self, run_id: str, model_name: str, model_version: str = "1"):
        """Register model in MLflow Model Registry"""
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(
            model_uri, 
            f"{model_name}_fraud_detector"
        )
        return registered_model
    
    def get_best_model(self, model_name: str):
        """Get the best performing model from registry"""
        try:
            registered_models = self.client.search_registered_models(
                filter_string=f"name='{model_name}_fraud_detector'"
            )
            
            if registered_models:
                latest_version = registered_models[0].latest_versions[0]
                return latest_version
            return None
        except Exception as e:
            print(f"Error getting best model: {e}")
            return None
    
    def log_data_info(self, dataset_path: str, feature_columns: list):
        """Log dataset information"""
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("num_features", len(feature_columns))
        mlflow.log_param("feature_columns", str(feature_columns))
    
    def log_environment_info(self):
        """Log environment information"""
        import platform
        import sys
        
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("architecture", platform.architecture()[0])

# Global MLflow manager instance
mlflow_manager = MLflowManager()

def get_mlflow_manager() -> MLflowManager:
    """Get global MLflow manager instance"""
    return mlflow_manager

