"""
Monitoring & Metrics - 2025 Production Ready
Clean, efficient monitoring with Prometheus integration
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading
import psutil
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from loguru import logger

from ..utils.config import get_config

# Prometheus Metrics
REQUEST_COUNT = Counter('fraud_detection_requests_total', 'Total fraud detection requests')
REQUEST_DURATION = Histogram('fraud_detection_request_duration_seconds', 'Request duration')
FRAUD_PREDICTIONS = Counter('fraud_predictions_total', 'Total fraud predictions', ['prediction'])
SYSTEM_METRICS = Gauge('system_metrics', 'System metrics', ['metric_type'])
MODEL_PERFORMANCE = Gauge('model_performance_auc', 'Model AUC score', ['model_name'])

@dataclass
class TransactionLog:
    """Transaction log entry"""
    transaction_id: str
    customer_id: str
    terminal_id: str
    amount: float
    timestamp: datetime
    fraud_probability: float
    is_fraud: bool
    confidence: str
    processing_time_ms: float
    risk_factors: List[str]

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_connections: int
    model_inference_time_ms: float
    queue_size: int

class MetricsCollector:
    """Advanced metrics collection and monitoring"""
    
    def __init__(self):
        self.config = get_config()
        self.transaction_logs = deque(maxlen=10000)
        self.system_metrics_history = deque(maxlen=1000)
        self.request_times = deque(maxlen=1000)
        self.fraud_rate_history = deque(maxlen=100)
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        
        def monitor_system():
            while True:
                try:
                    metrics = self._collect_system_metrics()
                    self.system_metrics_history.append(metrics)
                    self._update_prometheus_metrics(metrics)
                    time.sleep(10)  # Collect every 10 seconds
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    time.sleep(30)
        
        # Start monitoring thread
        threading.Thread(target=monitor_system, daemon=True).start()
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # Active connections
        connections = len(psutil.net_connections())
        
        # Model inference time (average from recent requests)
        avg_inference_time = np.mean(list(self.request_times)) if self.request_times else 0
        
        # Queue size
        queue_size = len(self.transaction_logs)
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_usage_percent=disk_usage_percent,
            active_connections=connections,
            model_inference_time_ms=avg_inference_time,
            queue_size=queue_size
        )
    
    def _update_prometheus_metrics(self, metrics: SystemMetrics):
        """Update Prometheus metrics"""
        
        SYSTEM_METRICS.labels(metric_type='cpu_percent').set(metrics.cpu_percent)
        SYSTEM_METRICS.labels(metric_type='memory_percent').set(metrics.memory_percent)
        SYSTEM_METRICS.labels(metric_type='memory_used_mb').set(metrics.memory_used_mb)
        SYSTEM_METRICS.labels(metric_type='disk_usage_percent').set(metrics.disk_usage_percent)
        SYSTEM_METRICS.labels(metric_type='active_connections').set(metrics.active_connections)
        SYSTEM_METRICS.labels(metric_type='model_inference_time_ms').set(metrics.model_inference_time_ms)
        SYSTEM_METRICS.labels(metric_type='queue_size').set(metrics.queue_size)
    
    def record_prediction(self, is_fraud: bool, processing_time_ms: float):
        """Record a prediction"""
        
        # Update Prometheus metrics
        FRAUD_PREDICTIONS.labels(prediction=str(is_fraud)).inc()
        
        # Record processing time
        self.request_times.append(processing_time_ms)
        
        # Update fraud rate history
        recent_fraud_rate = self._calculate_recent_fraud_rate()
        self.fraud_rate_history.append(recent_fraud_rate)
    
    def log_transaction(self, transaction_log: TransactionLog):
        """Log transaction with structured data"""
        
        # Add to local storage
        self.transaction_logs.append(transaction_log)
        
        # Log to file
        logger.info(
            f"Transaction processed: {transaction_log.transaction_id}",
            extra={
                "transaction_id": transaction_log.transaction_id,
                "customer_id": transaction_log.customer_id,
                "amount": transaction_log.amount,
                "fraud_probability": transaction_log.fraud_probability,
                "is_fraud": transaction_log.is_fraud,
                "processing_time_ms": transaction_log.processing_time_ms,
                "risk_factors": transaction_log.risk_factors
            }
        )
    
    def _calculate_recent_fraud_rate(self, window_minutes: int = 60) -> float:
        """Calculate fraud rate for recent transactions"""
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_transactions = [
            log for log in self.transaction_logs 
            if log.timestamp >= cutoff_time
        ]
        
        if not recent_transactions:
            return 0.0
        
        fraud_count = sum(1 for log in recent_transactions if log.is_fraud)
        return fraud_count / len(recent_transactions)
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        
        recent_transactions = list(self.transaction_logs)[-100:] if self.transaction_logs else []
        
        # Calculate metrics
        total_transactions = len(self.transaction_logs)
        fraud_detected = sum(1 for log in self.transaction_logs if log.is_fraud)
        fraud_rate = (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0
        
        avg_processing_time = np.mean([log.processing_time_ms for log in recent_transactions]) if recent_transactions else 0
        
        return {
            "total_transactions": total_transactions,
            "fraud_detected": fraud_detected,
            "fraud_rate": round(fraud_rate, 2),
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "recent_fraud_rate": round(self._calculate_recent_fraud_rate() * 100, 2),
            "system_health": self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health"""
        
        if not self.system_metrics_history:
            return "unknown"
        
        latest_metrics = self.system_metrics_history[-1]
        
        # Check various health indicators
        health_score = 100
        
        if latest_metrics.cpu_percent > 80:
            health_score -= 20
        elif latest_metrics.cpu_percent > 60:
            health_score -= 10
        
        if latest_metrics.memory_percent > 85:
            health_score -= 20
        elif latest_metrics.memory_percent > 70:
            health_score -= 10
        
        if latest_metrics.model_inference_time_ms > 1000:
            health_score -= 15
        elif latest_metrics.model_inference_time_ms > 500:
            health_score -= 5
        
        # Recent fraud rate
        recent_fraud_rate = self._calculate_recent_fraud_rate()
        if recent_fraud_rate > 0.2:
            health_score -= 10
        elif recent_fraud_rate > 0.1:
            health_score -= 5
        
        # Determine health status
        if health_score >= 90:
            return "excellent"
        elif health_score >= 75:
            return "good"
        elif health_score >= 60:
            return "fair"
        else:
            return "poor"
    
    def export_metrics(self) -> Response:
        """Export Prometheus metrics"""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    def get_transaction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent transaction history"""
        
        recent_transactions = list(self.transaction_logs)[-limit:]
        
        return [
            {
                "transaction_id": log.transaction_id,
                "customer_id": log.customer_id,
                "amount": log.amount,
                "fraud_probability": log.fraud_probability,
                "is_fraud": log.is_fraud,
                "confidence": log.confidence,
                "processing_time_ms": log.processing_time_ms,
                "timestamp": log.timestamp.isoformat(),
                "risk_factors": log.risk_factors
            }
            for log in recent_transactions
        ]

# Global metrics collector instance
metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    return metrics_collector

