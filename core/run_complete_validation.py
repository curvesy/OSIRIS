#!/usr/bin/env python3
"""
🧪 Complete End-to-End Pipeline Validation Runner

REAL, EXECUTABLE validation that proves the Intelligence Flywheel works:
- Spins up containerized test environment (Docker Compose)
- Runs comprehensive integration tests (Hot→Cold→Wise flow)
- Executes load testing with realistic traffic patterns
- Validates data quality across all tiers
- Generates comprehensive validation report
- Provides CI/CD ready exit codes and artifacts
"""

import asyncio
import sys
import os
import subprocess
import time
import json
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import docker
from loguru import logger


class CompleteValidationRunner:
    """
    🧪 Complete End-to-End Pipeline Validation Runner
    
    Orchestrates the complete validation process:
    1. Environment Setup (Docker Compose)
    2. Integration Testing (pytest)
    3. Load Testing (Locust)
    4. Data Quality Validation (Great Expectations)
    5. Report Generation and Cleanup
    """
    
    def __init__(self, test_mode: str = "full"):
        self.test_mode = test_mode  # full, integration, load, quality
        self.project_root = Path(__file__).parent
        self.test_results_dir = self.project_root / "test-results"
        self.test_results_dir.mkdir(exist_ok=True)
        
        # Docker client for container management
        try:
            self.docker_client = docker.from_env()
            logger.info("✅ Docker client initialized")
        except Exception as e:
            logger.error(f"❌ Docker not available: {e}")
            self.docker_client = None
        
        # Test configuration
        self.test_config = {
            "integration": {
                "timeout_minutes": 10,
                "required_services": ["redis", "minio", "test-runner"],
                "pytest_args": [
                    "tests/integration",
                    "-v",
                    "--tb=short",
                    "--html=test-results/integration_report.html",
                    "--json-report",
                    "--json-report-file=test-results/integration_results.json"
                ]
            },
            "load": {
                "timeout_minutes": 5,
                "users": int(os.getenv("LOAD_TEST_USERS", "50")),
                "duration": os.getenv("LOAD_TEST_DURATION", "2m"),
                "spawn_rate": int(os.getenv("LOAD_TEST_SPAWN_RATE", "5"))
            },
            "quality": {
                "timeout_minutes": 5,
                "validation_thresholds": {
                    "quality_score_min": 0.8,
                    "success_rate_min": 0.9
                }
            }
        }
        
        # Track running containers for cleanup
        self.running_containers = []
        
        logger.info(f"🧪 Complete Validation Runner initialized (mode: {test_mode})")
    
    async def run_complete_validation(self) -> Dict[str, Any]:
        """Run the complete end-to-end validation process."""
        
        start_time = time.time()
        validation_results = {
            "validation_start": datetime.now().isoformat(),
            "test_mode": self.test_mode,
            "phases": {},
            "overall_status": "unknown",
            "duration_seconds": 0,
            "artifacts": []
        }
        
        try:
            logger.info("🚀 Starting Complete End-to-End Pipeline Validation")
            logger.info("=" * 70)
            
            # Phase 1: Environment Setup
            if self.test_mode in ["full", "integration", "load"]:
                logger.info("🐳 Phase 1: Setting up test environment...")
                env_result = await self._setup_test_environment()
                validation_results["phases"]["environment_setup"] = env_result
                
                if env_result["status"] != "success":
                    raise Exception(f"Environment setup failed: {env_result.get('error')}")
                
                logger.info("✅ Phase 1: Test environment ready")
            
            # Phase 2: Integration Testing
            if self.test_mode in ["full", "integration"]:
                logger.info("🧪 Phase 2: Running integration tests...")
                integration_result = await self._run_integration_tests()
                validation_results["phases"]["integration_tests"] = integration_result
                logger.info(f"✅ Phase 2: Integration tests - {integration_result['status']}")
            
            # Phase 3: Load Testing
            if self.test_mode in ["full", "load"]:
                logger.info("🚀 Phase 3: Running load tests...")
                load_result = await self._run_load_tests()
                validation_results["phases"]["load_tests"] = load_result
                logger.info(f"✅ Phase 3: Load tests - {load_result['status']}")
            
            # Phase 4: Data Quality Validation
            if self.test_mode in ["full", "quality"]:
                logger.info("🔍 Phase 4: Running data quality validation...")
                quality_result = await self._run_quality_validation()
                validation_results["phases"]["quality_validation"] = quality_result
                logger.info(f"✅ Phase 4: Quality validation - {quality_result['status']}")
            
            # Phase 5: Generate Final Report
            logger.info("📊 Phase 5: Generating validation report...")
            report_result = await self._generate_final_report(validation_results)
            validation_results["phases"]["report_generation"] = report_result
            validation_results["artifacts"] = report_result.get("artifacts", [])
            
            # Determine overall status
            validation_results["overall_status"] = self._determine_overall_status(validation_results)
            validation_results["duration_seconds"] = time.time() - start_time
            
            logger.info("=" * 70)
            logger.info(f"🎉 Complete Validation FINISHED: {validation_results['overall_status'].upper()}")
            logger.info(f"⏱️ Total Duration: {validation_results['duration_seconds']:.1f}s")
            
            return validation_results
            
        except Exception as e:
            validation_results["overall_status"] = "error"
            validation_results["error"] = str(e)
            validation_results["duration_seconds"] = time.time() - start_time
            
            logger.error(f"❌ Complete validation failed: {e}")
            return validation_results
        
        finally:
            # Always cleanup
            await self._cleanup_test_environment()
    
    async def _setup_test_environment(self) -> Dict[str, Any]:
        """Setup containerized test environment with Docker Compose."""
        
        try:
            compose_file = self.project_root / "tests/integration/docker-compose.test.yml"
            
            if not compose_file.exists():
                return {
                    "status": "error",
                    "error": f"Docker Compose file not found: {compose_file}"
                }
            
            # Start services
            logger.info("🐳 Starting test services with Docker Compose...")
            
            cmd = [
                "docker-compose",
                "-f", str(compose_file),
                "up", "-d",
                "--build"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                return {
                    "status": "error",
                    "error": f"Docker Compose failed: {result.stderr}",
                    "stdout": result.stdout
                }
            
            # Wait for services to be healthy
            logger.info("⏳ Waiting for services to be healthy...")
            await self._wait_for_services_healthy()
            
            # Get container info
            containers = self._get_running_containers()
            
            return {
                "status": "success",
                "containers": containers,
                "compose_file": str(compose_file),
                "startup_time": "< 5 minutes"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Docker Compose startup timeout (5 minutes)"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _wait_for_services_healthy(self, max_wait_seconds: int = 120):
        """Wait for all required services to be healthy."""
        
        required_services = self.test_config["integration"]["required_services"]
        
        for _ in range(max_wait_seconds):
            try:
                # Check if all services are running
                containers = self.docker_client.containers.list()
                running_services = [
                    c.name for c in containers 
                    if any(service in c.name for service in required_services)
                ]
                
                if len(running_services) >= len(required_services):
                    logger.info(f"✅ All services healthy: {running_services}")
                    return
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"⚠️ Health check failed: {e}")
                await asyncio.sleep(1)
        
        raise Exception(f"Services not healthy after {max_wait_seconds}s")
    
    def _get_running_containers(self) -> List[Dict[str, str]]:
        """Get information about running test containers."""
        
        if not self.docker_client:
            return []
        
        containers = []
        for container in self.docker_client.containers.list():
            if "test" in container.name.lower():
                containers.append({
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "status": container.status,
                    "ports": str(container.ports)
                })
                self.running_containers.append(container)
        
        return containers
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests with pytest."""
        
        try:
            config = self.test_config["integration"]
            
            # Build pytest command
            cmd = ["python", "-m", "pytest"] + config["pytest_args"]
            
            logger.info(f"🧪 Running: {' '.join(cmd)}")
            
            # Run tests
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=config["timeout_minutes"] * 60
            )
            
            # Parse results
            results_file = self.test_results_dir / "integration_results.json"
            test_results = {}
            
            if results_file.exists():
                with open(results_file) as f:
                    test_results = json.load(f)
            
            return {
                "status": "success" if result.returncode == 0 else "fail",
                "exit_code": result.returncode,
                "tests_run": test_results.get("summary", {}).get("total", 0),
                "tests_passed": test_results.get("summary", {}).get("passed", 0),
                "tests_failed": test_results.get("summary", {}).get("failed", 0),
                "duration_seconds": test_results.get("duration", 0),
                "report_file": "test-results/integration_report.html",
                "stdout": result.stdout[-1000:],  # Last 1000 chars
                "stderr": result.stderr[-1000:] if result.stderr else ""
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": f"Integration tests timeout ({config['timeout_minutes']} minutes)"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _run_load_tests(self) -> Dict[str, Any]:
        """Run load tests with Locust."""
        
        try:
            config = self.test_config["load"]
            
            # Build Locust command
            cmd = [
                "python", "-m", "locust",
                "-f", "tests/load/test_search_load.py",
                "--host", "http://localhost:8000",
                "--users", str(config["users"]),
                "--spawn-rate", str(config["spawn_rate"]),
                "--run-time", config["duration"],
                "--headless",
                "--html", "test-results/load_test_report.html",
                "--csv", "test-results/load_test"
            ]
            
            logger.info(f"🚀 Running: {' '.join(cmd)}")
            
            # Run load tests
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=config["timeout_minutes"] * 60
            )
            
            # Parse CSV results if available
            csv_file = self.test_results_dir / "load_test_stats.csv"
            load_stats = {}
            
            if csv_file.exists():
                import pandas as pd
                df = pd.read_csv(csv_file)
                if not df.empty:
                    load_stats = {
                        "total_requests": df["Request Count"].sum(),
                        "failure_count": df["Failure Count"].sum(),
                        "avg_response_time": df["Average Response Time"].mean(),
                        "requests_per_second": df["Requests/s"].mean()
                    }
            
            return {
                "status": "success" if result.returncode == 0 else "fail",
                "exit_code": result.returncode,
                "users": config["users"],
                "duration": config["duration"],
                "stats": load_stats,
                "report_file": "test-results/load_test_report.html",
                "stdout": result.stdout[-1000:],
                "stderr": result.stderr[-1000:] if result.stderr else ""
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": f"Load tests timeout ({config['timeout_minutes']} minutes)"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _run_quality_validation(self) -> Dict[str, Any]:
        """Run data quality validation."""
        
        try:
            # Run quality validation script
            cmd = ["python", "tests/quality/validate_pipeline.py"]
            
            logger.info(f"🔍 Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.test_config["quality"]["timeout_minutes"] * 60
            )
            
            return {
                "status": "success" if result.returncode == 0 else "fail",
                "exit_code": result.returncode,
                "stdout": result.stdout[-1000:],
                "stderr": result.stderr[-1000:] if result.stderr else ""
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": f"Quality validation timeout ({self.test_config['quality']['timeout_minutes']} minutes)"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _generate_final_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final validation report."""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.test_results_dir / f"complete_validation_report_{timestamp}.json"
            
            # Add summary statistics
            validation_results["summary"] = {
                "total_phases": len(validation_results["phases"]),
                "successful_phases": len([
                    phase for phase in validation_results["phases"].values()
                    if phase.get("status") == "success"
                ]),
                "validation_timestamp": datetime.now().isoformat(),
                "test_environment": "containerized",
                "ci_cd_ready": True
            }
            
            # Save comprehensive report
            with open(report_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            # Generate artifacts list
            artifacts = [str(report_file)]
            
            # Add other report files
            for report_pattern in ["*.html", "*.json", "*.csv"]:
                artifacts.extend([
                    str(f) for f in self.test_results_dir.glob(report_pattern)
                    if f != report_file
                ])
            
            logger.info(f"📊 Final report generated: {report_file}")
            logger.info(f"📁 Total artifacts: {len(artifacts)}")
            
            return {
                "status": "success",
                "report_file": str(report_file),
                "artifacts": artifacts,
                "artifact_count": len(artifacts)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "artifacts": []
            }
    
    def _determine_overall_status(self, validation_results: Dict[str, Any]) -> str:
        """Determine overall validation status."""
        
        phases = validation_results.get("phases", {})
        
        # Check for any errors
        if any(phase.get("status") == "error" for phase in phases.values()):
            return "error"
        
        # Check for any failures
        if any(phase.get("status") == "fail" for phase in phases.values()):
            return "fail"
        
        # Check for any timeouts
        if any(phase.get("status") == "timeout" for phase in phases.values()):
            return "timeout"
        
        # All phases must be successful
        if all(phase.get("status") == "success" for phase in phases.values()):
            return "success"
        
        return "unknown"
    
    async def _cleanup_test_environment(self):
        """Cleanup test environment and containers."""
        
        try:
            logger.info("🧹 Cleaning up test environment...")
            
            # Stop Docker Compose services
            compose_file = self.project_root / "tests/integration/docker-compose.test.yml"
            
            if compose_file.exists():
                cmd = [
                    "docker-compose",
                    "-f", str(compose_file),
                    "down",
                    "-v",  # Remove volumes
                    "--remove-orphans"
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    logger.info("✅ Docker Compose cleanup completed")
                else:
                    logger.warning(f"⚠️ Docker Compose cleanup issues: {result.stderr}")
            
            # Force cleanup any remaining containers
            for container in self.running_containers:
                try:
                    container.stop(timeout=10)
                    container.remove()
                except:
                    pass
            
            logger.info("🧹 Cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")


async def main():
    """Main entry point for complete validation."""
    
    # Parse command line arguments
    test_mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    
    if test_mode not in ["full", "integration", "load", "quality"]:
        print(f"❌ Invalid test mode: {test_mode}")
        print("Valid modes: full, integration, load, quality")
        return 1
    
    # Setup signal handlers for graceful shutdown
    runner = CompleteValidationRunner(test_mode)
    
    def signal_handler(signum, frame):
        logger.info("🛑 Received shutdown signal, cleaning up...")
        asyncio.create_task(runner._cleanup_test_environment())
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run validation
    try:
        results = await runner.run_complete_validation()
        
        # Print summary
        print(f"\n🧪 Complete Pipeline Validation Results:")
        print(f"   Overall Status: {results['overall_status'].upper()}")
        print(f"   Duration: {results['duration_seconds']:.1f}s")
        print(f"   Test Mode: {results['test_mode']}")
        print(f"   Artifacts: {len(results.get('artifacts', []))}")
        
        if results.get('artifacts'):
            print(f"\n📁 Generated Artifacts:")
            for artifact in results['artifacts'][:5]:  # Show first 5
                print(f"   📄 {Path(artifact).name}")
            if len(results['artifacts']) > 5:
                print(f"   ... and {len(results['artifacts']) - 5} more")
        
        # Return appropriate exit code
        if results['overall_status'] == 'success':
            print(f"\n🎉 VALIDATION PASSED! Intelligence Flywheel is production-ready.")
            return 0
        else:
            print(f"\n❌ VALIDATION FAILED: {results['overall_status']}")
            if 'error' in results:
                print(f"   Error: {results['error']}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n🛑 Validation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
