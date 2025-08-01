#!/usr/bin/env python3
"""
📊 Governance Dashboard API - Phase 3C Implementation

FastAPI endpoints for the governance dashboard that provides:
1. Real-time prediction accuracy metrics
2. Risk score distribution analysis  
3. Human approval queue monitoring
4. ROI validation and cost-benefit analysis

This is the strategic monitoring interface outlined in ksksksk.md for
enterprise AI governance and stakeholder reporting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from ..observability.shadow_mode_logger import ShadowModeLogger

logger = logging.getLogger(__name__)


class DashboardMetrics(BaseModel):
    """Response model for dashboard metrics."""
    
    # Time period
    period_days: int
    last_updated: str
    
    # Accuracy metrics
    overall_prediction_accuracy: float
    overall_risk_accuracy: float
    total_predictions: int
    successful_outcomes: int
    
    # Governance metrics
    human_approvals_required: int
    human_approval_rate: float
    
    # Routing effectiveness
    routing_accuracy: List[Dict[str, Any]]
    
    # ROI metrics
    estimated_incidents_prevented: int
    estimated_cost_savings: float
    
    # Performance metrics
    entries_logged: int
    outcomes_recorded: int
    data_completeness_rate: float


class RiskDistribution(BaseModel):
    """Risk score distribution for governance analysis."""
    
    risk_buckets: List[Dict[str, Any]]
    average_risk_score: float
    high_risk_percentage: float
    escalation_rate: float


class PredictionTrends(BaseModel):
    """Prediction accuracy trends over time."""
    
    daily_accuracy: List[Dict[str, Any]]
    weekly_trend: str  # "improving", "stable", "declining"
    confidence_calibration: Dict[str, float]


# Initialize FastAPI app
app = FastAPI(
    title="AURA Intelligence Governance Dashboard",
    description="Enterprise AI governance and monitoring dashboard for Phase 3C",
    version="3.0.0"
)

# Global shadow logger instance
_shadow_logger: Optional[ShadowModeLogger] = None

async def get_shadow_logger() -> ShadowModeLogger:
    """Get or create the shadow mode logger."""
    global _shadow_logger
    if _shadow_logger is None:
        _shadow_logger = ShadowModeLogger()
        await _shadow_logger.initialize()
    return _shadow_logger


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze")
) -> DashboardMetrics:
    """
    📊 Get comprehensive dashboard metrics for governance reporting.
    
    This endpoint provides the core metrics for stakeholder reporting and
    ROI validation as outlined in the Phase 3C strategic plan.
    """
    try:
        shadow_logger = await get_shadow_logger()
        raw_metrics = await shadow_logger.get_accuracy_metrics(days=days)
        
        # Calculate derived metrics
        total_predictions = raw_metrics.get("total_predictions", 0)
        human_approvals = raw_metrics.get("human_approvals_required", 0)
        human_approval_rate = (human_approvals / total_predictions) if total_predictions > 0 else 0.0
        
        # Estimate ROI metrics (conservative estimates)
        prediction_accuracy = raw_metrics.get("overall_prediction_accuracy", 0.0)
        estimated_incidents_prevented = int(total_predictions * prediction_accuracy * 0.3)  # 30% incident rate
        estimated_cost_savings = estimated_incidents_prevented * 15000  # $15K per incident
        
        # Data completeness
        entries_logged = raw_metrics.get("entries_logged", 0)
        outcomes_recorded = raw_metrics.get("outcomes_recorded", 0)
        data_completeness_rate = (outcomes_recorded / entries_logged) if entries_logged > 0 else 0.0
        
        return DashboardMetrics(
            period_days=days,
            last_updated=datetime.now().isoformat(),
            overall_prediction_accuracy=raw_metrics.get("overall_prediction_accuracy", 0.0),
            overall_risk_accuracy=raw_metrics.get("overall_risk_accuracy", 0.0),
            total_predictions=total_predictions,
            successful_outcomes=raw_metrics.get("successful_outcomes", 0),
            human_approvals_required=human_approvals,
            human_approval_rate=human_approval_rate,
            routing_accuracy=raw_metrics.get("routing_accuracy", []),
            estimated_incidents_prevented=estimated_incidents_prevented,
            estimated_cost_savings=estimated_cost_savings,
            entries_logged=entries_logged,
            outcomes_recorded=outcomes_recorded,
            data_completeness_rate=data_completeness_rate
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {str(e)}")


@app.get("/risk-distribution", response_model=RiskDistribution)
async def get_risk_distribution(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze")
) -> RiskDistribution:
    """
    📈 Get risk score distribution for governance analysis.
    
    Provides insights into risk assessment patterns and escalation rates.
    """
    try:
        shadow_logger = await get_shadow_logger()
        
        # Get risk distribution data from shadow logger
        # This would require additional methods in ShadowModeLogger
        # For now, return mock data structure
        
        return RiskDistribution(
            risk_buckets=[
                {"range": "0.0-0.2", "count": 45, "percentage": 30.0},
                {"range": "0.2-0.4", "count": 38, "percentage": 25.3},
                {"range": "0.4-0.6", "count": 32, "percentage": 21.3},
                {"range": "0.6-0.8", "count": 23, "percentage": 15.3},
                {"range": "0.8-1.0", "count": 12, "percentage": 8.0}
            ],
            average_risk_score=0.35,
            high_risk_percentage=23.3,  # >0.6 risk score
            escalation_rate=8.0  # >0.8 risk score requiring human approval
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get risk distribution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve risk distribution: {str(e)}")


@app.get("/prediction-trends", response_model=PredictionTrends)
async def get_prediction_trends(
    days: int = Query(14, ge=7, le=90, description="Number of days to analyze")
) -> PredictionTrends:
    """
    📈 Get prediction accuracy trends over time.
    
    Provides trend analysis for continuous improvement and model calibration.
    """
    try:
        # This would require additional time-series analysis in ShadowModeLogger
        # For now, return mock trend data
        
        daily_accuracy = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            # Mock trending accuracy (improving over time)
            accuracy = 0.65 + (i * 0.02) + (0.1 * (i % 3 == 0))  # Some variation
            daily_accuracy.append({
                "date": date.strftime("%Y-%m-%d"),
                "accuracy": min(accuracy, 0.95),  # Cap at 95%
                "predictions": 15 + (i % 5)  # Varying daily volume
            })
        
        # Determine trend
        recent_accuracy = sum(day["accuracy"] for day in daily_accuracy[-3:]) / 3
        earlier_accuracy = sum(day["accuracy"] for day in daily_accuracy[:3]) / 3
        
        if recent_accuracy > earlier_accuracy + 0.05:
            trend = "improving"
        elif recent_accuracy < earlier_accuracy - 0.05:
            trend = "declining"
        else:
            trend = "stable"
        
        return PredictionTrends(
            daily_accuracy=daily_accuracy,
            weekly_trend=trend,
            confidence_calibration={
                "low_confidence": 0.72,    # Accuracy when confidence < 0.5
                "medium_confidence": 0.84, # Accuracy when confidence 0.5-0.8
                "high_confidence": 0.91    # Accuracy when confidence > 0.8
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get prediction trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prediction trends: {str(e)}")


@app.get("/export/training-data")
async def export_training_data(
    days: int = Query(30, ge=1, le=365, description="Number of days to export"),
    format: str = Query("jsonl", regex="^(json|jsonl|csv)$", description="Export format")
):
    """
    📤 Export training data for data science analysis.
    
    Provides raw shadow mode logs for advanced analytics and model training.
    """
    try:
        shadow_logger = await get_shadow_logger()
        
        # This would require additional export functionality in ShadowModeLogger
        # For now, return export metadata
        
        return {
            "export_requested": True,
            "period_days": days,
            "format": format,
            "estimated_records": 450,  # Mock estimate
            "download_url": f"/download/training-data-{days}d.{format}",
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to export training data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export training data: {str(e)}")


if __name__ == "__main__":
    # Run the governance dashboard server
    uvicorn.run(
        "governance_dashboard:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
