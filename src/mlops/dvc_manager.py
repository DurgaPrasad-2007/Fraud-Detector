"""
Data Version Control (DVC) Integration
Modern data versioning and pipeline management for 2025
"""

import dvc.api
import pandas as pd
from pathlib import Path
import yaml
import hashlib
from datetime import datetime
import logging
from loguru import logger

class DVCManager:
    """DVC manager for data versioning and pipeline management"""
    
    def __init__(self):
        self.dvc_config = self._load_dvc_config()
        self.data_dir = Path("data")
        
    def _load_dvc_config(self) -> dict:
        """Load DVC configuration"""
        config_path = Path("dvc.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def init_dvc(self):
        """Initialize DVC repository"""
        logger.info("Initializing DVC repository...")
        
        # Create dvc.yaml for pipeline definition
        pipeline_config = {
            "stages": {
                "data_generation": {
                    "cmd": "python -c \"from src.data.preprocessing import get_preprocessor; get_preprocessor().create_synthetic_dataset(50000).to_csv('data/raw/synthetic_data.csv', index=False)\"",
                    "deps": ["src/data/preprocessing.py"],
                    "outs": ["data/raw/synthetic_data.csv"],
                    "metrics": ["data/raw/data_metrics.json"]
                },
                "feature_engineering": {
                    "cmd": "python -c \"from src.data.preprocessing import get_preprocessor; import pandas as pd; df = pd.read_csv('data/raw/synthetic_data.csv'); df_features = get_preprocessor().engineer_features(df); df_features.to_csv('data/processed/features.csv', index=False)\"",
                    "deps": ["data/raw/synthetic_data.csv", "src/data/preprocessing.py"],
                    "outs": ["data/processed/features.csv"],
                    "metrics": ["data/processed/feature_metrics.json"]
                },
                "model_training": {
                    "cmd": "python -c \"from src.models.ensemble import get_model; from src.data.preprocessing import get_preprocessor; import pandas as pd; df = pd.read_csv('data/processed/features.csv'); preprocessor = get_preprocessor(); X, y = preprocessor.prepare_training_data(df); model = get_model(); results, X_test, y_test = model.train_models(X, y); model.save_models()\"",
                    "deps": ["data/processed/features.csv", "src/models/ensemble.py"],
                    "outs": ["data/models/"],
                    "metrics": ["data/models/model_metrics.json"]
                }
            }
        }
        
        with open("dvc.yaml", 'w') as f:
            yaml.dump(pipeline_config, f, default_flow_style=False)
        
        logger.info("DVC pipeline configuration created")
    
    def add_data_file(self, file_path: str, description: str = ""):
        """Add data file to DVC tracking"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Calculate file hash for versioning
        file_hash = self._calculate_file_hash(file_path)
        
        # Create metadata
        metadata = {
            "file_path": str(file_path),
            "description": description,
            "hash": file_hash,
            "timestamp": datetime.now().isoformat(),
            "size": file_path.stat().st_size
        }
        
        # Save metadata
        metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Added {file_path} to DVC tracking")
        return True
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_data_version(self, file_path: str) -> dict:
        """Get version information for data file"""
        file_path = Path(file_path)
        metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                import json
                return json.load(f)
        return {}
    
    def list_data_versions(self) -> list:
        """List all tracked data versions"""
        versions = []
        
        for metadata_file in self.data_dir.rglob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                import json
                metadata = json.load(f)
                versions.append(metadata)
        
        return sorted(versions, key=lambda x: x['timestamp'], reverse=True)
    
    def create_data_snapshot(self, snapshot_name: str):
        """Create a snapshot of current data state"""
        snapshot_dir = Path(f"data/snapshots/{snapshot_name}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy current data files
        for data_file in self.data_dir.rglob("*.csv"):
            if data_file.is_file():
                snapshot_file = snapshot_dir / data_file.relative_to(self.data_dir)
                snapshot_file.parent.mkdir(parents=True, exist_ok=True)
                
                import shutil
                shutil.copy2(data_file, snapshot_file)
        
        logger.info(f"Created data snapshot: {snapshot_name}")
        return snapshot_dir
    
    def restore_data_snapshot(self, snapshot_name: str):
        """Restore data from snapshot"""
        snapshot_dir = Path(f"data/snapshots/{snapshot_name}")
        
        if not snapshot_dir.exists():
            logger.error(f"Snapshot not found: {snapshot_name}")
            return False
        
        # Restore files
        for snapshot_file in snapshot_dir.rglob("*.csv"):
            if snapshot_file.is_file():
                target_file = self.data_dir / snapshot_file.relative_to(snapshot_dir)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                import shutil
                shutil.copy2(snapshot_file, target_file)
        
        logger.info(f"Restored data from snapshot: {snapshot_name}")
        return True
    
    def run_pipeline(self, stage: str = None):
        """Run DVC pipeline"""
        if stage:
            logger.info(f"Running DVC pipeline stage: {stage}")
            # Run specific stage
            import subprocess
            result = subprocess.run(["dvc", "repro", stage], capture_output=True, text=True)
        else:
            logger.info("Running complete DVC pipeline")
            # Run all stages
            import subprocess
            result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("DVC pipeline completed successfully")
            return True
        else:
            logger.error(f"DVC pipeline failed: {result.stderr}")
            return False

# Global DVC manager instance
dvc_manager = DVCManager()

def get_dvc_manager() -> DVCManager:
    """Get global DVC manager instance"""
    return dvc_manager

