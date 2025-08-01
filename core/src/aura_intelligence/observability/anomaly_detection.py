"""
🤖 AI-Powered Anomaly Detection - 2025 Production Grade

Intelligent anomaly detection system using:
- Facebook Prophet for time series forecasting
- Multi-signal correlation analysis
- Business-impact aware alerting
- Self-learning baseline adjustment
- Integration with OpenTelemetry metrics
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Prophet for time series forecasting
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available - using fallback anomaly detection")

# Prometheus client for metrics
from prometheus_client import Gauge, Counter, Histogram

from .telemetry import get_telemetry


class AnomalyType(Enum):
    """Types of anomalies detected by the system."""
    
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ERROR_SPIKE = "error_spike"
    PATTERN_SHIFT = "pattern_shift"
    AGENT_MALFUNCTION = "agent_malfunction"
    DATA_QUALITY = "data_quality"


class Severity(Enum):
    """Anomaly severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """Structured anomaly alert."""
    
    id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: Severity
    metric_name: str
    current_value: float
    expected_value: float
    deviation_score: float
    confidence: float
    context: Dict[str, Any]
    business_impact: str
    recommended_actions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['anomaly_type'] = self.anomaly_type.value
        result['severity'] = self.severity.value
        return result


class IntelligentAnomalyDetector:
    """
    🤖 AI-Powered Anomaly Detection System
    
    Uses Facebook Prophet and statistical methods to detect anomalies in:
    - Search latency patterns
    - Memory usage trends
    - Agent decision quality
    - Data pipeline health
    - Business metrics
    """
    
    def __init__(self, lookback_days: int = 7, confidence_interval: float = 0.95):
        self.lookback_days = lookback_days
        self.confidence_interval = confidence_interval
        self.logger = logging.getLogger(__name__)
        self.telemetry = get_telemetry()
        
        # Prophet models for different metrics
        self.models: Dict[str, Prophet] = {}
        self.model_last_trained: Dict[str, datetime] = {}
        
        # Prometheus metrics for anomaly detection system
        self.anomalies_detected = Counter(
            'aura_anomalies_detected_total',
            'Total anomalies detected by type and severity',
            ['anomaly_type', 'severity', 'metric_name']
        )
        
        self.anomaly_detection_duration = Histogram(
            'aura_anomaly_detection_duration_seconds',
            'Time spent on anomaly detection',
            ['metric_name']
        )
        
        self.model_accuracy = Gauge(
            'aura_anomaly_model_accuracy',
            'Accuracy of anomaly detection models',
            ['metric_name', 'model_type']
        )
        
        # Business impact thresholds
        self.business_impact_thresholds = {
            'search_latency': {
                'low': 100,      # 100ms
                'medium': 500,   # 500ms
                'high': 1000,    # 1s
                'critical': 5000 # 5s
            },
            'error_rate': {
                'low': 0.01,     # 1%
                'medium': 0.05,  # 5%
                'high': 0.10,    # 10%
                'critical': 0.25 # 25%
            },
            'memory_usage': {
                'low': 0.70,     # 70%
                'medium': 0.80,  # 80%
                'high': 0.90,    # 90%
                'critical': 0.95 # 95%
            }
        }
        
        self.logger.info("🤖 Intelligent Anomaly Detector initialized")
    
    async def detect_anomalies(self, metrics_data: Dict[str, pd.DataFrame]) -> List[AnomalyAlert]:
        """
        Detect anomalies across multiple metrics using AI models.
        
        Args:
            metrics_data: Dictionary of metric name to time series DataFrame
            
        Returns:
            List of anomaly alerts
        """
        
        alerts = []
        
        for metric_name, data in metrics_data.items():
            try:
                with self.anomaly_detection_duration.labels(metric_name=metric_name).time():
                    metric_alerts = await self._detect_metric_anomalies(metric_name, data)
                    alerts.extend(metric_alerts)
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to detect anomalies for {metric_name}: {e}")
                continue
        
        # Perform multi-signal correlation analysis
        correlated_alerts = await self._correlate_anomalies(alerts)
        
        # Filter and prioritize alerts
        final_alerts = await self._prioritize_alerts(correlated_alerts)
        
        # Record metrics
        for alert in final_alerts:
            self.anomalies_detected.labels(
                anomaly_type=alert.anomaly_type.value,
                severity=alert.severity.value,
                metric_name=alert.metric_name
            ).inc()
        
        self.logger.info(f"🤖 Detected {len(final_alerts)} anomalies across {len(metrics_data)} metrics")
        return final_alerts
    
    async def _detect_metric_anomalies(self, metric_name: str, data: pd.DataFrame) -> List[AnomalyAlert]:
        """Detect anomalies for a specific metric using Prophet."""
        
        if not PROPHET_AVAILABLE:
            return await self._fallback_anomaly_detection(metric_name, data)
        
        try:
            # Prepare data for Prophet
            prophet_data = self._prepare_prophet_data(data)
            
            if len(prophet_data) < 10:  # Need minimum data points
                return []
            
            # Train or update model
            model = await self._get_or_train_model(metric_name, prophet_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=1, freq='min')
            forecast = model.predict(future)
            
            # Detect anomalies
            anomalies = await self._identify_anomalies(
                metric_name, prophet_data, forecast, model
            )
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"❌ Prophet anomaly detection failed for {metric_name}: {e}")
            return await self._fallback_anomaly_detection(metric_name, data)
    
    def _prepare_prophet_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model."""
        
        # Prophet expects 'ds' (datestamp) and 'y' (value) columns
        prophet_data = data.copy()
        
        if 'timestamp' in prophet_data.columns:
            prophet_data['ds'] = pd.to_datetime(prophet_data['timestamp'])
        else:
            prophet_data['ds'] = prophet_data.index
        
        # Use the first numeric column as the value
        numeric_cols = prophet_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            prophet_data['y'] = prophet_data[numeric_cols[0]]
        else:
            raise ValueError("No numeric columns found for Prophet model")
        
        return prophet_data[['ds', 'y']].dropna()
    
    async def _get_or_train_model(self, metric_name: str, data: pd.DataFrame) -> Prophet:
        """Get existing model or train a new one."""
        
        # Check if model needs retraining (daily)
        needs_training = (
            metric_name not in self.models or
            metric_name not in self.model_last_trained or
            datetime.now() - self.model_last_trained[metric_name] > timedelta(days=1)
        )
        
        if needs_training:
            self.logger.info(f"🧠 Training Prophet model for {metric_name}")
            
            # Configure Prophet model
            model = Prophet(
                interval_width=self.confidence_interval,
                changepoint_prior_scale=0.05,  # Detect trend changes
                seasonality_prior_scale=10.0,  # Detect seasonal patterns
                holidays_prior_scale=10.0,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False  # Not relevant for short-term metrics
            )
            
            # Add custom seasonalities for AI workloads
            model.add_seasonality(
                name='hourly',
                period=1,  # 1 day
                fourier_order=8
            )
            
            # Train model
            model.fit(data)
            
            # Evaluate model performance
            accuracy = await self._evaluate_model(model, data)
            self.model_accuracy.labels(
                metric_name=metric_name,
                model_type='prophet'
            ).set(accuracy)
            
            self.models[metric_name] = model
            self.model_last_trained[metric_name] = datetime.now()
        
        return self.models[metric_name]
    
    async def _evaluate_model(self, model: Prophet, data: pd.DataFrame) -> float:
        """Evaluate model accuracy using cross-validation."""
        
        try:
            if len(data) < 20:  # Need sufficient data for CV
                return 0.5  # Default accuracy
            
            # Perform cross-validation
            cv_results = cross_validation(
                model, 
                initial='3 days', 
                period='1 day', 
                horizon='1 day'
            )
            
            # Calculate performance metrics
            performance = performance_metrics(cv_results)
            
            # Use MAPE (Mean Absolute Percentage Error) as accuracy metric
            mape = performance['mape'].mean()
            accuracy = max(0.0, 1.0 - mape)  # Convert MAPE to accuracy
            
            return accuracy
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model evaluation failed: {e}")
            return 0.5  # Default accuracy
    
    async def _identify_anomalies(
        self, 
        metric_name: str, 
        actual_data: pd.DataFrame, 
        forecast: pd.DataFrame,
        model: Prophet
    ) -> List[AnomalyAlert]:
        """Identify anomalies by comparing actual vs predicted values."""
        
        anomalies = []
        
        # Get the latest actual values
        latest_actual = actual_data.tail(10)  # Last 10 data points
        
        for _, row in latest_actual.iterrows():
            timestamp = row['ds']
            actual_value = row['y']
            
            # Find corresponding forecast
            forecast_row = forecast[forecast['ds'] == timestamp]
            
            if forecast_row.empty:
                continue
            
            forecast_row = forecast_row.iloc[0]
            predicted_value = forecast_row['yhat']
            lower_bound = forecast_row['yhat_lower']
            upper_bound = forecast_row['yhat_upper']
            
            # Check if actual value is outside confidence interval
            is_anomaly = actual_value < lower_bound or actual_value > upper_bound
            
            if is_anomaly:
                # Calculate deviation score
                if actual_value < lower_bound:
                    deviation_score = (lower_bound - actual_value) / abs(predicted_value)
                else:
                    deviation_score = (actual_value - upper_bound) / abs(predicted_value)
                
                # Determine anomaly type and severity
                anomaly_type = self._classify_anomaly_type(metric_name, actual_value, predicted_value)
                severity = self._calculate_severity(metric_name, actual_value, deviation_score)
                
                # Create alert
                alert = AnomalyAlert(
                    id=f"{metric_name}_{int(timestamp.timestamp())}",
                    timestamp=timestamp,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    metric_name=metric_name,
                    current_value=actual_value,
                    expected_value=predicted_value,
                    deviation_score=deviation_score,
                    confidence=self.confidence_interval,
                    context={
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'model_type': 'prophet'
                    },
                    business_impact=self._assess_business_impact(metric_name, actual_value, severity),
                    recommended_actions=self._generate_recommendations(metric_name, anomaly_type, severity)
                )
                
                anomalies.append(alert)
        
        return anomalies
    
    def _classify_anomaly_type(self, metric_name: str, actual: float, predicted: float) -> AnomalyType:
        """Classify the type of anomaly based on metric and values."""
        
        if 'latency' in metric_name.lower() or 'duration' in metric_name.lower():
            return AnomalyType.PERFORMANCE_DEGRADATION if actual > predicted else AnomalyType.PATTERN_SHIFT
        
        elif 'memory' in metric_name.lower() or 'cpu' in metric_name.lower():
            return AnomalyType.RESOURCE_EXHAUSTION if actual > predicted else AnomalyType.PATTERN_SHIFT
        
        elif 'error' in metric_name.lower() or 'failure' in metric_name.lower():
            return AnomalyType.ERROR_SPIKE if actual > predicted else AnomalyType.PATTERN_SHIFT
        
        elif 'agent' in metric_name.lower():
            return AnomalyType.AGENT_MALFUNCTION
        
        else:
            return AnomalyType.PATTERN_SHIFT
    
    def _calculate_severity(self, metric_name: str, value: float, deviation_score: float) -> Severity:
        """Calculate severity based on business impact thresholds."""
        
        # Extract base metric type
        base_metric = None
        for key in self.business_impact_thresholds.keys():
            if key in metric_name.lower():
                base_metric = key
                break
        
        if base_metric and base_metric in self.business_impact_thresholds:
            thresholds = self.business_impact_thresholds[base_metric]
            
            if value >= thresholds['critical']:
                return Severity.CRITICAL
            elif value >= thresholds['high']:
                return Severity.HIGH
            elif value >= thresholds['medium']:
                return Severity.MEDIUM
            else:
                return Severity.LOW
        
        # Fallback to deviation-based severity
        if deviation_score > 2.0:
            return Severity.CRITICAL
        elif deviation_score > 1.0:
            return Severity.HIGH
        elif deviation_score > 0.5:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _assess_business_impact(self, metric_name: str, value: float, severity: Severity) -> str:
        """Assess business impact of the anomaly."""
        
        impact_descriptions = {
            Severity.CRITICAL: "Severe user experience degradation, potential service outage",
            Severity.HIGH: "Significant performance impact, user experience affected",
            Severity.MEDIUM: "Moderate performance impact, monitoring required",
            Severity.LOW: "Minor deviation from normal patterns"
        }
        
        return impact_descriptions.get(severity, "Unknown impact")
    
    def _generate_recommendations(self, metric_name: str, anomaly_type: AnomalyType, severity: Severity) -> List[str]:
        """Generate actionable recommendations based on anomaly."""
        
        recommendations = []
        
        if anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION:
            recommendations.extend([
                "Check system resource utilization",
                "Review recent deployments or configuration changes",
                "Analyze slow query logs",
                "Consider scaling up resources"
            ])
        
        elif anomaly_type == AnomalyType.RESOURCE_EXHAUSTION:
            recommendations.extend([
                "Scale up memory or CPU resources",
                "Check for memory leaks",
                "Review resource limits and requests",
                "Implement resource cleanup procedures"
            ])
        
        elif anomaly_type == AnomalyType.ERROR_SPIKE:
            recommendations.extend([
                "Check application logs for error details",
                "Review recent code deployments",
                "Verify external service dependencies",
                "Implement circuit breaker if not present"
            ])
        
        elif anomaly_type == AnomalyType.AGENT_MALFUNCTION:
            recommendations.extend([
                "Review agent decision logs",
                "Check agent model performance",
                "Verify agent input data quality",
                "Consider agent model retraining"
            ])
        
        if severity in [Severity.HIGH, Severity.CRITICAL]:
            recommendations.insert(0, "IMMEDIATE ACTION REQUIRED")
        
        return recommendations
    
    async def _fallback_anomaly_detection(self, metric_name: str, data: pd.DataFrame) -> List[AnomalyAlert]:
        """Fallback anomaly detection using statistical methods."""
        
        # Simple statistical anomaly detection using z-score
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return []
        
        anomalies = []
        
        for col in numeric_cols:
            values = data[col].dropna()
            
            if len(values) < 10:
                continue
            
            mean = values.mean()
            std = values.std()
            
            # Z-score threshold for anomaly detection
            threshold = 3.0
            
            # Check recent values
            recent_values = values.tail(5)
            
            for idx, value in recent_values.items():
                z_score = abs((value - mean) / std) if std > 0 else 0
                
                if z_score > threshold:
                    alert = AnomalyAlert(
                        id=f"{metric_name}_{col}_{int(time.time())}",
                        timestamp=datetime.now(),
                        anomaly_type=self._classify_anomaly_type(metric_name, value, mean),
                        severity=Severity.MEDIUM if z_score > 4 else Severity.LOW,
                        metric_name=f"{metric_name}.{col}",
                        current_value=value,
                        expected_value=mean,
                        deviation_score=z_score,
                        confidence=0.99,  # 3-sigma confidence
                        context={'method': 'z_score', 'threshold': threshold},
                        business_impact="Statistical deviation detected",
                        recommended_actions=["Investigate recent changes", "Monitor trend"]
                    )
                    
                    anomalies.append(alert)
        
        return anomalies
    
    async def _correlate_anomalies(self, alerts: List[AnomalyAlert]) -> List[AnomalyAlert]:
        """Perform multi-signal correlation analysis."""
        
        # Group alerts by time window (5 minutes)
        time_groups = {}
        
        for alert in alerts:
            time_key = alert.timestamp.replace(second=0, microsecond=0)
            time_key = time_key.replace(minute=time_key.minute // 5 * 5)  # 5-minute buckets
            
            if time_key not in time_groups:
                time_groups[time_key] = []
            
            time_groups[time_key].append(alert)
        
        # Enhance alerts with correlation information
        for time_key, group_alerts in time_groups.items():
            if len(group_alerts) > 1:
                # Multiple anomalies in same time window - likely correlated
                for alert in group_alerts:
                    alert.context['correlated_anomalies'] = len(group_alerts)
                    alert.context['correlation_window'] = time_key.isoformat()
                    
                    # Increase severity if multiple systems affected
                    if len(group_alerts) >= 3 and alert.severity != Severity.CRITICAL:
                        alert.severity = Severity.HIGH
        
        return alerts
    
    async def _prioritize_alerts(self, alerts: List[AnomalyAlert]) -> List[AnomalyAlert]:
        """Filter and prioritize alerts to reduce noise."""
        
        # Sort by severity and deviation score
        severity_order = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1
        }
        
        sorted_alerts = sorted(
            alerts,
            key=lambda a: (severity_order[a.severity], a.deviation_score),
            reverse=True
        )
        
        # Limit number of alerts to prevent alert fatigue
        max_alerts = 20
        
        return sorted_alerts[:max_alerts]


# Global anomaly detector instance
_anomaly_detector: Optional[IntelligentAnomalyDetector] = None


def get_anomaly_detector() -> IntelligentAnomalyDetector:
    """Get global anomaly detector instance."""
    global _anomaly_detector
    
    if _anomaly_detector is None:
        _anomaly_detector = IntelligentAnomalyDetector()
    
    return _anomaly_detector
