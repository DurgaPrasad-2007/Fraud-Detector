"""
Data Preprocessing Module - 2025 Production Ready
Clean, efficient data processing with feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from loguru import logger

from ..utils.config import get_config

class DataPreprocessor:
    """Advanced data preprocessing for fraud detection"""
    
    def __init__(self):
        self.config = get_config()
        self.feature_columns = self.config.model.feature_columns
        self.feature_stats = {}
        
    def load_real_dataset(self, data_dir: str = "dataset/data") -> pd.DataFrame:
        """Load the real fraud detection dataset
        
        This loads all the pickle files from the dataset directory and combines them.
        Each file contains transactions for one day.
        """
        
        logger.info(f"Loading real dataset from {data_dir}")
        
        all_dataframes = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        # Load all pickle files
        pickle_files = sorted(data_path.glob("*.pkl"))
        logger.info(f"Found {len(pickle_files)} data files")
        
        for file_path in pickle_files:
            try:
                df_day = pd.read_pickle(file_path)
                all_dataframes.append(df_day)
                logger.debug(f"Loaded {len(df_day)} transactions from {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
        
        if not all_dataframes:
            raise ValueError("No valid data files found")
        
        # Combine all dataframes
        df = pd.concat(all_dataframes, ignore_index=True)
        
        # Convert TX_DATETIME to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['TX_DATETIME']):
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        
        logger.info(f"Dataset loaded: {len(df):,} transactions, {df['TX_FRAUD'].mean():.4%} fraud rate")
        logger.info(f"Date range: {df['TX_DATETIME'].min()} to {df['TX_DATETIME'].max()}")
        
        return df
    
    def _apply_fraud_scenarios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fraud scenarios to dataset"""
        
        # Scenario 1: Amount > 220 is fraud
        df.loc[df['TX_AMOUNT'] > 220, 'TX_FRAUD'] = 1
        
        # Scenario 2: Random terminals fraudulent for 28 days
        terminals = df['TERMINAL_ID'].unique()
        
        for day in range(365):
            current_date = datetime(2023, 1, 1) + timedelta(days=day)
            end_date = current_date + timedelta(days=28)
            
            daily_fraud_terminals = np.random.choice(terminals, 2, replace=False)
            
            mask = (df['TX_DATETIME'] >= current_date) & \
                   (df['TX_DATETIME'] < end_date) & \
                   (df['TERMINAL_ID'].isin(daily_fraud_terminals))
            df.loc[mask, 'TX_FRAUD'] = 1
        
        # Scenario 3: Random customers have 1/3 transactions × 5
        customers = df['CUSTOMER_ID'].unique()
        
        for day in range(365):
            current_date = datetime(2023, 1, 1) + timedelta(days=day)
            end_date = current_date + timedelta(days=14)
            
            daily_fraud_customers = np.random.choice(customers, 3, replace=False)
            
            for customer in daily_fraud_customers:
                customer_mask = (df['CUSTOMER_ID'] == customer) & \
                               (df['TX_DATETIME'] >= current_date) & \
                               (df['TX_DATETIME'] < end_date)
                
                customer_txs = df[customer_mask]
                if len(customer_txs) > 0:
                    n_fraud_txs = max(1, len(customer_txs) // 3)
                    fraud_indices = np.random.choice(
                        customer_txs.index, n_fraud_txs, replace=False
                    )
                    
                    df.loc[fraud_indices, 'TX_AMOUNT'] *= 5
                    df.loc[fraud_indices, 'TX_FRAUD'] = 1
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features for fraud detection
        
        This creates features that help the model understand patterns in transaction behavior.
        We create features for:
        1. Time-based patterns (hour, day, weekend)
        2. Customer behavior (spending patterns, frequency)
        3. Terminal behavior (usage patterns)
        4. Transaction context (amount ratios, timing)
        """
        
        logger.info("Engineering features...")
        df = df.copy()
        
        # Ensure datetime column is properly formatted
        if not pd.api.types.is_datetime64_any_dtype(df['TX_DATETIME']):
            df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
        
        # 1. TEMPORAL FEATURES - Time-based patterns
        logger.info("Creating temporal features...")
        df['hour_of_day'] = df['TX_DATETIME'].dt.hour
        df['day_of_week'] = df['TX_DATETIME'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['amount_log'] = np.log1p(df['TX_AMOUNT'])  # Log transform for skewed amounts
        
        # 2. CUSTOMER BEHAVIOR FEATURES - How customers typically behave
        logger.info("Creating customer behavior features...")
        customer_stats = df.groupby('CUSTOMER_ID').agg({
            'TX_AMOUNT': ['mean', 'std', 'count'],
            'TX_FRAUD': 'mean'
        }).round(4)
        
        customer_stats.columns = [
            'customer_avg_amount', 'customer_amount_std', 
            'customer_transaction_count', 'customer_fraud_rate'
        ]
        
        df = df.merge(customer_stats, left_on='CUSTOMER_ID', right_index=True, how='left')
        
        # 3. TERMINAL FEATURES - Terminal usage patterns
        logger.info("Creating terminal features...")
        terminal_stats = df.groupby('TERMINAL_ID').agg({
            'TX_AMOUNT': ['mean', 'count'],
            'TX_FRAUD': 'mean'
        }).round(4)
        
        terminal_stats.columns = [
            'terminal_avg_amount', 'terminal_transaction_count', 'terminal_fraud_rate'
        ]
        
        df = df.merge(terminal_stats, left_on='TERMINAL_ID', right_index=True, how='left')
        
        # 4. TRANSACTION CONTEXT FEATURES - Relative to customer patterns
        logger.info("Creating transaction context features...")
        
        # Time since last transaction for this customer
        df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
        df['days_since_last_transaction'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff().dt.days
        df['days_since_last_transaction'] = df['days_since_last_transaction'].fillna(0)
        
        # Number of transactions today for this customer
        df['date'] = df['TX_DATETIME'].dt.date
        daily_counts = df.groupby(['CUSTOMER_ID', 'date']).size().reset_index(name='transactions_today')
        df = df.merge(daily_counts, on=['CUSTOMER_ID', 'date'], how='left')
        df = df.drop('date', axis=1)
        
        # Amount ratio - how much this transaction is vs customer's average
        df['amount_vs_avg_ratio'] = df['TX_AMOUNT'] / df['customer_avg_amount']
        df['amount_vs_avg_ratio'] = df['amount_vs_avg_ratio'].fillna(1)
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        logger.info(f"Feature engineering completed: {len(df.columns)} total features")
        logger.info(f"Features created: {', '.join(self.feature_columns)}")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        
        # Select features and target
        X = df[self.feature_columns].values
        y = df['TX_FRAUD'].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Training data prepared: {X.shape[0]:,} samples, {X.shape[1]} features")
        return X, y
    
    def save_dataset(self, df: pd.DataFrame, filepath: str = "data/processed/fraud_dataset.csv"):
        """Save processed dataset"""
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str = "data/processed/fraud_dataset.csv") -> pd.DataFrame:
        """Load processed dataset"""
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded from {filepath}: {len(df):,} transactions")
        return df

# Global preprocessor instance
preprocessor = DataPreprocessor()

def get_preprocessor() -> DataPreprocessor:
    """Get global preprocessor instance"""
    return preprocessor
