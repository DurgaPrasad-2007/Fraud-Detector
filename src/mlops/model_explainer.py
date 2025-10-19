"""
Model Interpretability and Explainability
SHAP and LIME integration for 2025 fraud detection standards
"""

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from loguru import logger

from ..utils.config import get_config
from ..models.ensemble import get_model

class ModelExplainer:
    """Advanced model explainability using SHAP and LIME"""
    
    def __init__(self):
        self.config = get_config()
        self.model = get_model()
        self.feature_columns = self.config.model.feature_columns
        self.explainer = None
        self.lime_explainer = None
        
    def initialize_shap_explainer(self, X_train: np.ndarray):
        """Initialize SHAP explainer"""
        logger.info("Initializing SHAP explainer...")
        
        # Use TreeExplainer for tree-based models
        if hasattr(self.model.models, 'get') and 'xgboost' in self.model.models:
            self.explainer = shap.TreeExplainer(self.model.models['xgboost'])
        else:
            # Use KernelExplainer as fallback
            self.explainer = shap.KernelExplainer(
                self._model_predict_wrapper, 
                X_train[:100]  # Use subset for efficiency
            )
        
        logger.info("SHAP explainer initialized")
    
    def initialize_lime_explainer(self, X_train: np.ndarray):
        """Initialize LIME explainer"""
        logger.info("Initializing LIME explainer...")
        
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_columns,
            class_names=['Legitimate', 'Fraudulent'],
            mode='classification',
            discretize_continuous=True
        )
        
        logger.info("LIME explainer initialized")
    
    def _model_predict_wrapper(self, X):
        """Wrapper function for SHAP explainer"""
        predictions = []
        for x in X:
            pred = self.model.predict_single(x)
            predictions.append(pred['ensemble_probability'])
        return np.array(predictions)
    
    def explain_prediction_shap(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain single prediction using SHAP"""
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        
        # Handle different model types
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class values
        
        # Create explanation
        explanation = {
            'shap_values': shap_values[0].tolist(),
            'feature_names': self.feature_columns,
            'feature_importance': dict(zip(self.feature_columns, shap_values[0])),
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            'prediction': self.model.predict_single(instance)['ensemble_probability']
        }
        
        return explanation
    
    def explain_prediction_lime(self, instance: np.ndarray) -> Dict[str, Any]:
        """Explain single prediction using LIME"""
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized")
        
        # Get LIME explanation
        explanation = self.lime_explainer.explain_instance(
            instance,
            self._model_predict_wrapper,
            num_features=len(self.feature_columns)
        )
        
        # Extract explanation data
        lime_data = {
            'explanation': explanation.as_list(),
            'feature_importance': dict(explanation.as_list()),
            'prediction': explanation.predict_proba[1],  # Fraud probability
            'confidence': explanation.score
        }
        
        return lime_data
    
    def explain_prediction_combined(self, instance: np.ndarray) -> Dict[str, Any]:
        """Combined SHAP and LIME explanation"""
        shap_explanation = self.explain_prediction_shap(instance)
        lime_explanation = self.explain_prediction_lime(instance)
        
        # Combine explanations
        combined = {
            'shap': shap_explanation,
            'lime': lime_explanation,
            'consensus_features': self._get_consensus_features(
                shap_explanation['feature_importance'],
                lime_explanation['feature_importance']
            ),
            'prediction': shap_explanation['prediction'],
            'confidence': lime_explanation['confidence']
        }
        
        return combined
    
    def _get_consensus_features(self, shap_features: Dict, lime_features: Dict) -> List[str]:
        """Get features that both SHAP and LIME agree are important"""
        # Get top 5 features from each method
        shap_top = sorted(shap_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        lime_top = sorted(lime_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Find consensus features
        shap_top_names = set([item[0] for item in shap_top])
        lime_top_names = set([item[0] for item in lime_top])
        
        consensus = list(shap_top_names.intersection(lime_top_names))
        return consensus
    
    def generate_feature_importance_plot(self, X_sample: np.ndarray, save_path: str = None):
        """Generate feature importance plot"""
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        # Get SHAP values for sample
        shap_values = self.explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_columns, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def generate_waterfall_plot(self, instance: np.ndarray, save_path: str = None):
        """Generate SHAP waterfall plot for single prediction"""
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            self.explainer.expected_value,
            shap_values[0],
            instance,
            feature_names=self.feature_columns,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Waterfall plot saved to {save_path}")
        
        plt.show()
    
    def explain_batch_predictions(self, X_batch: np.ndarray) -> List[Dict[str, Any]]:
        """Explain multiple predictions"""
        explanations = []
        
        for i, instance in enumerate(X_batch):
            try:
                explanation = self.explain_prediction_combined(instance)
                explanations.append({
                    'instance_id': i,
                    'explanation': explanation
                })
            except Exception as e:
                logger.error(f"Error explaining instance {i}: {e}")
                explanations.append({
                    'instance_id': i,
                    'error': str(e)
                })
        
        return explanations
    
    def save_explanation_report(self, explanations: List[Dict], filepath: str):
        """Save explanation report to file"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_info': {
                'feature_columns': self.feature_columns,
                'num_features': len(self.feature_columns)
            },
            'explanations': explanations
        }
        
        with open(filepath, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        logger.info(f"Explanation report saved to {filepath}")
    
    def get_model_insights(self, X_sample: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive model insights"""
        insights = {
            'feature_importance': {},
            'risk_factors': [],
            'model_behavior': {},
            'recommendations': []
        }
        
        # Calculate feature importance across sample
        if self.explainer is not None:
            shap_values = self.explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Average importance across sample
            avg_importance = np.mean(np.abs(shap_values), axis=0)
            insights['feature_importance'] = dict(zip(self.feature_columns, avg_importance))
        
        # Identify risk factors
        for i, instance in enumerate(X_sample):
            pred = self.model.predict_single(instance)
            if pred['is_fraud']:
                insights['risk_factors'].append(f"Instance {i}: High fraud probability")
        
        # Model behavior insights
        predictions = [self.model.predict_single(instance)['ensemble_probability'] for instance in X_sample]
        insights['model_behavior'] = {
            'avg_fraud_probability': np.mean(predictions),
            'fraud_rate': np.mean([p > 0.5 for p in predictions]),
            'confidence_variance': np.var(predictions)
        }
        
        # Generate recommendations
        if insights['model_behavior']['fraud_rate'] > 0.1:
            insights['recommendations'].append("Consider reviewing high-risk transactions")
        
        if insights['model_behavior']['confidence_variance'] > 0.1:
            insights['recommendations'].append("Model confidence varies significantly - consider retraining")
        
        return insights

# Global explainer instance
model_explainer = ModelExplainer()

def get_model_explainer() -> ModelExplainer:
    """Get global model explainer instance"""
    return model_explainer

