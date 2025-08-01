"""
⏰ Automated Archival Scheduler

Background task scheduler for automated data archival, retention management,
and system health monitoring. Integrates with UltimateAURASystem lifecycle.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import duckdb

from aura_intelligence.enterprise.mem0_hot.archive import ArchivalManager
from aura_intelligence.enterprise.mem0_hot.settings import DuckDBSettings
from aura_intelligence.utils.logger import get_logger


@dataclass
class SchedulerConfig:
    """Configuration for the archival scheduler."""
    
    # Archival intervals
    archival_interval_minutes: int = 60  # Run every hour
    health_check_interval_minutes: int = 15  # Health check every 15 minutes
    
    # Retention policies
    hot_retention_hours: int = 24
    emergency_cleanup_threshold_gb: float = 8.0  # Trigger emergency cleanup
    
    # Performance thresholds
    max_archival_time_seconds: float = 300.0  # 5 minutes max
    max_consecutive_failures: int = 3
    
    # Alerting
    enable_alerts: bool = True
    alert_webhook_url: Optional[str] = None


class ArchivalScheduler:
    """
    ⏰ Automated Archival Scheduler
    
    Features:
    - Hourly automated archival to S3
    - Emergency cleanup when storage is full
    - Health monitoring and alerting
    - Performance metrics tracking
    - Integration with system lifecycle
    """
    
    def __init__(self, 
                 conn: duckdb.DuckDBPyConnection,
                 settings: DuckDBSettings,
                 config: SchedulerConfig = None):
        """Initialize the archival scheduler."""
        
        self.conn = conn
        self.settings = settings
        self.config = config or SchedulerConfig()
        
        # Create archival manager
        self.archival_manager = ArchivalManager(conn, settings)
        
        # Scheduler state
        self.is_running = False
        self.archival_task = None
        self.health_task = None
        
        # Performance tracking
        self.consecutive_failures = 0
        self.last_successful_archival = None
        self.total_scheduler_uptime = 0.0
        self.start_time = None
        
        # Health callbacks
        self.health_callbacks: List[Callable] = []
        
        self.logger = get_logger(__name__)
        self.logger.info("⏰ Archival Scheduler initialized")
    
    async def start(self) -> bool:
        """Start the automated archival scheduler."""
        
        if self.is_running:
            self.logger.warning("⚠️ Scheduler already running")
            return False
        
        try:
            # Initialize archival manager
            await self.archival_manager.start_background_archival(
                interval_minutes=self.config.archival_interval_minutes
            )
            
            # Start scheduler tasks
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start health monitoring
            self.health_task = asyncio.create_task(
                self._health_monitoring_loop()
            )
            
            # Start emergency cleanup monitoring
            self.archival_task = asyncio.create_task(
                self._emergency_cleanup_loop()
            )
            
            self.logger.info("🚀 Archival Scheduler started")
            self.logger.info(f"   - Archival interval: {self.config.archival_interval_minutes} minutes")
            self.logger.info(f"   - Health check interval: {self.config.health_check_interval_minutes} minutes")
            self.logger.info(f"   - Hot retention: {self.config.hot_retention_hours} hours")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start scheduler: {e}")
            self.is_running = False
            return False
    
    async def stop(self) -> bool:
        """Stop the automated archival scheduler."""
        
        if not self.is_running:
            return True
        
        try:
            self.is_running = False
            
            # Stop background tasks
            if self.health_task:
                self.health_task.cancel()
                try:
                    await self.health_task
                except asyncio.CancelledError:
                    pass
            
            if self.archival_task:
                self.archival_task.cancel()
                try:
                    await self.archival_task
                except asyncio.CancelledError:
                    pass
            
            # Stop archival manager
            await self.archival_manager.stop_background_archival()
            
            # Update uptime
            if self.start_time:
                self.total_scheduler_uptime += (datetime.now() - self.start_time).total_seconds()
            
            self.logger.info("🛑 Archival Scheduler stopped")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop scheduler: {e}")
            return False
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        
        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval_minutes * 60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Health monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _emergency_cleanup_loop(self):
        """Background emergency cleanup monitoring."""
        
        while self.is_running:
            try:
                await self._check_emergency_cleanup()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Emergency cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        
        try:
            # Get archival manager health
            archival_health = await self.archival_manager.health_check()
            
            # Check database size
            db_size_mb = await self._get_database_size_mb()
            
            # Check recent archival performance
            archival_metrics = self.archival_manager.get_archival_metrics()
            
            # Determine overall health status
            is_healthy = (
                archival_health.get("status") == "healthy" and
                self.consecutive_failures < self.config.max_consecutive_failures and
                db_size_mb < (self.config.emergency_cleanup_threshold_gb * 1024)
            )
            
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "healthy" if is_healthy else "degraded",
                "database_size_mb": db_size_mb,
                "consecutive_failures": self.consecutive_failures,
                "last_successful_archival": self.last_successful_archival,
                "archival_health": archival_health,
                "archival_metrics": archival_metrics,
                "scheduler_uptime_hours": self._get_uptime_hours()
            }
            
            # Trigger health callbacks
            for callback in self.health_callbacks:
                try:
                    await callback(health_status)
                except Exception as e:
                    self.logger.error(f"❌ Health callback error: {e}")
            
            # Log health status
            if is_healthy:
                self.logger.debug(f"💚 System healthy - DB: {db_size_mb:.1f}MB")
            else:
                self.logger.warning(f"⚠️ System degraded - DB: {db_size_mb:.1f}MB, Failures: {self.consecutive_failures}")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _check_emergency_cleanup(self):
        """Check if emergency cleanup is needed."""
        
        try:
            db_size_mb = await self._get_database_size_mb()
            threshold_mb = self.config.emergency_cleanup_threshold_gb * 1024
            
            if db_size_mb > threshold_mb:
                self.logger.warning(f"🚨 Emergency cleanup triggered - DB size: {db_size_mb:.1f}MB > {threshold_mb:.1f}MB")
                
                # Trigger emergency archival
                result = await self.archival_manager.manual_archive(hours_back=12)
                
                if result.get("status") == "success":
                    self.logger.info(f"✅ Emergency cleanup successful - archived {result.get('archived_count', 0)} records")
                else:
                    self.logger.error(f"❌ Emergency cleanup failed: {result.get('error', 'Unknown error')}")
                    self.consecutive_failures += 1
            
        except Exception as e:
            self.logger.error(f"❌ Emergency cleanup check failed: {e}")
    
    async def _get_database_size_mb(self) -> float:
        """Get current database size in MB."""
        
        try:
            # Execute in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.conn.execute("SELECT database_size() / 1024.0 / 1024.0 as size_mb").fetchone()
            )
            
            return result[0] if result else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get database size: {e}")
            return 0.0
    
    def _get_uptime_hours(self) -> float:
        """Get scheduler uptime in hours."""
        
        if not self.start_time:
            return 0.0
        
        current_uptime = (datetime.now() - self.start_time).total_seconds()
        total_uptime = self.total_scheduler_uptime + current_uptime
        
        return total_uptime / 3600.0
    
    def add_health_callback(self, callback: Callable):
        """Add a health status callback."""
        self.health_callbacks.append(callback)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        
        return {
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_hours": self._get_uptime_hours(),
            "consecutive_failures": self.consecutive_failures,
            "last_successful_archival": self.last_successful_archival,
            "config": {
                "archival_interval_minutes": self.config.archival_interval_minutes,
                "health_check_interval_minutes": self.config.health_check_interval_minutes,
                "hot_retention_hours": self.config.hot_retention_hours,
                "emergency_cleanup_threshold_gb": self.config.emergency_cleanup_threshold_gb
            }
        }
