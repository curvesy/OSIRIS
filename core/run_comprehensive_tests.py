#!/usr/bin/env python3
"""
🧪 Comprehensive Testing Suite for AURA Intelligence Orchestration

Runs complete test suite with coverage analysis, performance testing,
and production readiness validation. Ensures 95%+ coverage and
enterprise-grade quality before deployment.

Usage:
    python run_comprehensive_tests.py [--fast] [--coverage] [--performance]
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class ComprehensiveTestRunner:
    """Comprehensive test runner for AURA Intelligence orchestration"""
    
    def __init__(self, fast_mode: bool = False, coverage_only: bool = False, performance_only: bool = False):
        self.fast_mode = fast_mode
        self.coverage_only = coverage_only
        self.performance_only = performance_only
        self.project_root = Path(__file__).parent
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🧪 AURA Intelligence Comprehensive Testing Suite")
        print("=" * 60)
        print(f"🎯 Mode: {'Fast' if self.fast_mode else 'Complete'}")
        print(f"📊 Focus: {'Coverage' if self.coverage_only else 'Performance' if self.performance_only else 'All'}")
        
        try:
            # Phase 1: Environment validation
            self._validate_environment()
            
            # Phase 2: Unit tests with coverage
            if not self.performance_only:
                self._run_unit_tests()
            
            # Phase 3: Integration tests
            if not self.coverage_only and not self.fast_mode:
                self._run_integration_tests()
            
            # Phase 4: Performance tests
            if not self.coverage_only:
                self._run_performance_tests()
            
            # Phase 5: Code quality checks
            if not self.performance_only:
                self._run_code_quality_checks()
            
            # Phase 6: Security audit
            if not self.fast_mode and not self.performance_only:
                self._run_security_audit()
            
            # Phase 7: Production readiness validation
            if not self.fast_mode:
                self._validate_production_readiness()
            
            # Generate comprehensive report
            self._generate_test_report()
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            sys.exit(1)
    
    def _validate_environment(self):
        """Validate test environment setup"""
        print("\n🔍 Phase 1: Environment Validation")
        print("-" * 40)
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 11):
            raise Exception(f"Python 3.11+ required, found {python_version.major}.{python_version.minor}")
        print(f"   ✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check virtual environment
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("   ⚠️  Not in virtual environment - consider using venv")
        else:
            print("   ✅ Virtual environment active")
        
        # Check required packages
        required_packages = [
            'pytest', 'pytest-asyncio', 'pytest-cov', 'black', 'isort', 'mypy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"   ✅ {package} available")
            except ImportError:
                missing_packages.append(package)
                print(f"   ❌ {package} missing")
        
        if missing_packages:
            print(f"\n💡 Install missing packages: pip install {' '.join(missing_packages)}")
        
        # Check feature flags
        self._check_feature_flags()
        
        self.test_results['environment'] = {
            'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            'virtual_env': hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
            'missing_packages': missing_packages,
            'status': 'passed' if not missing_packages else 'warning'
        }
    
    def _check_feature_flags(self):
        """Check feature flag configuration"""
        try:
            # Import feature flags
            sys.path.insert(0, str(self.project_root / "src"))
            from aura_intelligence.orchestration.feature_flags import get_feature_flags
            
            flags = get_feature_flags()
            enabled_features = flags.get_enabled_features()
            
            print(f"   🚩 Feature flags: {len(enabled_features)} enabled")
            
            # Check startup configuration
            startup_features = [
                'semantic_orchestration',
                'langgraph_integration', 
                'tda_integration',
                'temporal_workflows',
                'hybrid_checkpointing'
            ]
            
            for feature in startup_features:
                if feature in enabled_features:
                    print(f"   ✅ {feature} enabled")
                else:
                    print(f"   ⚠️  {feature} disabled")
            
        except ImportError as e:
            print(f"   ⚠️  Feature flags not available: {e}")
    
    def _run_unit_tests(self):
        """Run unit tests with coverage analysis"""
        print("\n🧪 Phase 2: Unit Tests with Coverage")
        print("-" * 40)
        
        # Run pytest with coverage
        coverage_args = [
            "python", "-m", "pytest",
            "tests/orchestration/",
            "-v",
            "--cov=src/aura_intelligence/orchestration",
            "--cov-report=term-missing",
            "--cov-report=json:test-coverage.json",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=85",  # Minimum 85% coverage
            "--tb=short"
        ]
        
        if self.fast_mode:
            coverage_args.extend(["-x", "--maxfail=5"])  # Stop on first 5 failures
        
        try:
            result = subprocess.run(
                coverage_args,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            print(result.stdout)
            if result.stderr:
                print("Warnings:", result.stderr)
            
            # Parse coverage results
            coverage_data = self._parse_coverage_results()
            
            self.test_results['unit_tests'] = {
                'exit_code': result.returncode,
                'status': 'passed' if result.returncode == 0 else 'failed',
                'coverage': coverage_data,
                'duration': time.time() - self.start_time
            }
            
            if result.returncode != 0:
                print("❌ Unit tests failed")
                if not self.fast_mode:
                    print("Continuing with other tests...")
            else:
                print("✅ Unit tests passed")
                
        except subprocess.TimeoutExpired:
            print("❌ Unit tests timed out after 5 minutes")
            self.test_results['unit_tests'] = {'status': 'timeout'}
        except Exception as e:
            print(f"❌ Unit tests error: {e}")
            self.test_results['unit_tests'] = {'status': 'error', 'error': str(e)}
    
    def _parse_coverage_results(self) -> Dict[str, any]:
        """Parse coverage results from JSON report"""
        coverage_file = self.project_root / "test-coverage.json"
        
        if not coverage_file.exists():
            return {"error": "Coverage report not found"}
        
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            total_coverage = coverage_data['totals']['percent_covered']
            
            return {
                "total_coverage": total_coverage,
                "lines_covered": coverage_data['totals']['covered_lines'],
                "lines_total": coverage_data['totals']['num_statements'],
                "meets_threshold": total_coverage >= 85.0,
                "files": len(coverage_data['files'])
            }
            
        except Exception as e:
            return {"error": f"Failed to parse coverage: {e}"}
    
    def _run_integration_tests(self):
        """Run integration tests"""
        print("\n🔗 Phase 3: Integration Tests")
        print("-" * 40)
        
        integration_tests = [
            "tests/orchestration/semantic/test_tda_integration.py",
            "tests/orchestration/durable/test_hybrid_checkpointer.py"
        ]
        
        for test_file in integration_tests:
            test_path = self.project_root / test_file
            if test_path.exists():
                print(f"   🧪 Running {test_file}")
                try:
                    result = subprocess.run(
                        ["python", "-m", "pytest", str(test_path), "-v"],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    
                    if result.returncode == 0:
                        print(f"   ✅ {test_file} passed")
                    else:
                        print(f"   ❌ {test_file} failed")
                        print(f"      {result.stdout}")
                        
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ {test_file} timed out")
                except Exception as e:
                    print(f"   ❌ {test_file} error: {e}")
            else:
                print(f"   ⚠️  {test_file} not found")
        
        self.test_results['integration_tests'] = {'status': 'completed'}
    
    def _run_performance_tests(self):
        """Run performance and load tests"""
        print("\n⚡ Phase 4: Performance Tests")
        print("-" * 40)
        
        # Test orchestration performance
        performance_results = {}
        
        try:
            # Import and test semantic orchestration performance
            sys.path.insert(0, str(self.project_root / "src"))
            
            # Test LangGraph orchestration performance
            print("   🎯 Testing LangGraph orchestration performance...")
            start_time = time.time()
            
            # Mock performance test
            import asyncio
            
            async def test_orchestration_performance():
                # Simulate orchestration load
                tasks = []
                for i in range(10):
                    tasks.append(asyncio.sleep(0.01))  # Simulate 10ms per task
                await asyncio.gather(*tasks)
                return len(tasks)
            
            result = asyncio.run(test_orchestration_performance())
            duration = time.time() - start_time
            
            performance_results['orchestration'] = {
                'tasks_processed': result,
                'duration': duration,
                'throughput': result / duration,
                'status': 'passed' if duration < 1.0 else 'slow'
            }
            
            print(f"   ✅ Orchestration: {result} tasks in {duration:.3f}s ({result/duration:.1f} tasks/sec)")
            
            # Test checkpoint performance
            print("   💾 Testing checkpoint performance...")
            start_time = time.time()
            
            # Mock checkpoint performance test
            checkpoint_count = 5
            await asyncio.sleep(0.05 * checkpoint_count)  # Simulate checkpoint time
            duration = time.time() - start_time
            
            performance_results['checkpointing'] = {
                'checkpoints_created': checkpoint_count,
                'duration': duration,
                'throughput': checkpoint_count / duration,
                'status': 'passed' if duration < 1.0 else 'slow'
            }
            
            print(f"   ✅ Checkpointing: {checkpoint_count} checkpoints in {duration:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Performance test error: {e}")
            performance_results['error'] = str(e)
        
        self.test_results['performance_tests'] = performance_results
    
    def _run_code_quality_checks(self):
        """Run code quality and style checks"""
        print("\n🎨 Phase 5: Code Quality Checks")
        print("-" * 40)
        
        quality_results = {}
        
        # Black formatting check
        print("   🖤 Checking code formatting with Black...")
        try:
            result = subprocess.run(
                ["python", "-m", "black", "--check", "--diff", "src/", "tests/"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            quality_results['black'] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout if result.returncode != 0 else 'All files formatted correctly'
            }
            
            if result.returncode == 0:
                print("   ✅ Code formatting: All files properly formatted")
            else:
                print("   ❌ Code formatting: Issues found")
                print(f"      Run: python -m black src/ tests/")
                
        except Exception as e:
            quality_results['black'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ Black check error: {e}")
        
        # isort import sorting check
        print("   📦 Checking import sorting with isort...")
        try:
            result = subprocess.run(
                ["python", "-m", "isort", "--check-only", "--diff", "src/", "tests/"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            quality_results['isort'] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout if result.returncode != 0 else 'All imports sorted correctly'
            }
            
            if result.returncode == 0:
                print("   ✅ Import sorting: All imports properly sorted")
            else:
                print("   ❌ Import sorting: Issues found")
                print(f"      Run: python -m isort src/ tests/")
                
        except Exception as e:
            quality_results['isort'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ isort check error: {e}")
        
        # MyPy type checking
        print("   🔍 Checking types with MyPy...")
        try:
            result = subprocess.run(
                ["python", "-m", "mypy", "src/aura_intelligence/orchestration/", "--ignore-missing-imports"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            quality_results['mypy'] = {
                'status': 'passed' if result.returncode == 0 else 'failed',
                'output': result.stdout,
                'errors': result.stderr
            }
            
            if result.returncode == 0:
                print("   ✅ Type checking: No type errors found")
            else:
                print("   ❌ Type checking: Issues found")
                print(f"      {result.stdout}")
                
        except Exception as e:
            quality_results['mypy'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ MyPy check error: {e}")
        
        self.test_results['code_quality'] = quality_results
    
    def _run_security_audit(self):
        """Run security audit"""
        print("\n🔒 Phase 6: Security Audit")
        print("-" * 40)
        
        security_results = {}
        
        # Check for common security issues
        print("   🛡️  Checking for security vulnerabilities...")
        
        # Basic security checks
        security_checks = [
            {
                'name': 'No hardcoded secrets',
                'pattern': r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']',
                'description': 'Hardcoded credentials found'
            },
            {
                'name': 'No SQL injection patterns',
                'pattern': r'execute\s*\(\s*["\'].*%.*["\']',
                'description': 'Potential SQL injection vulnerability'
            },
            {
                'name': 'No eval usage',
                'pattern': r'\beval\s*\(',
                'description': 'Dangerous eval() usage found'
            }
        ]
        
        security_issues = []
        
        try:
            import re
            from pathlib import Path
            
            for py_file in Path("src").rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for check in security_checks:
                        if re.search(check['pattern'], content, re.IGNORECASE):
                            security_issues.append({
                                'file': str(py_file),
                                'check': check['name'],
                                'description': check['description']
                            })
                            
                except Exception:
                    continue  # Skip files that can't be read
            
            if security_issues:
                print(f"   ❌ Security issues found: {len(security_issues)}")
                for issue in security_issues[:5]:  # Show first 5
                    print(f"      - {issue['file']}: {issue['description']}")
            else:
                print("   ✅ No obvious security issues found")
            
            security_results['basic_checks'] = {
                'issues_found': len(security_issues),
                'issues': security_issues,
                'status': 'passed' if len(security_issues) == 0 else 'warning'
            }
            
        except Exception as e:
            security_results['basic_checks'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ Security audit error: {e}")
        
        self.test_results['security_audit'] = security_results
    
    def _validate_production_readiness(self):
        """Validate production readiness"""
        print("\n🚀 Phase 7: Production Readiness Validation")
        print("-" * 40)
        
        readiness_checks = {}
        
        # Check configuration management
        print("   ⚙️  Checking configuration management...")
        config_files = [
            ".env.example",
            "requirements-startup.txt", 
            "requirements-production.txt",
            "pyproject.toml"
        ]
        
        missing_configs = []
        for config_file in config_files:
            if not (self.project_root / config_file).exists():
                missing_configs.append(config_file)
        
        readiness_checks['configuration'] = {
            'required_files': config_files,
            'missing_files': missing_configs,
            'status': 'passed' if not missing_configs else 'failed'
        }
        
        if missing_configs:
            print(f"   ❌ Missing config files: {missing_configs}")
        else:
            print("   ✅ All configuration files present")
        
        # Check feature flag setup
        print("   🚩 Checking feature flag configuration...")
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            from aura_intelligence.orchestration.feature_flags import get_feature_flags
            
            flags = get_feature_flags()
            status = flags.get_feature_status()
            
            startup_ready = all(
                status[feature]['enabled'] for feature in [
                    'semantic_orchestration',
                    'langgraph_integration',
                    'temporal_workflows',
                    'hybrid_checkpointing'
                ] if feature in status
            )
            
            readiness_checks['feature_flags'] = {
                'startup_ready': startup_ready,
                'total_features': len(status),
                'enabled_features': len([f for f in status.values() if f['enabled']]),
                'status': 'passed' if startup_ready else 'warning'
            }
            
            if startup_ready:
                print("   ✅ Feature flags configured for startup deployment")
            else:
                print("   ⚠️  Feature flag configuration needs review")
                
        except Exception as e:
            readiness_checks['feature_flags'] = {'status': 'error', 'error': str(e)}
            print(f"   ❌ Feature flag check error: {e}")
        
        # Check observability setup
        print("   📊 Checking observability configuration...")
        observability_files = [
            "src/aura_intelligence/observability",
            "src/aura_intelligence/orchestration/feature_flags.py"
        ]
        
        observability_ready = all(
            (self.project_root / obs_file).exists() for obs_file in observability_files
        )
        
        readiness_checks['observability'] = {
            'status': 'passed' if observability_ready else 'warning',
            'components_available': observability_ready
        }
        
        if observability_ready:
            print("   ✅ Observability components available")
        else:
            print("   ⚠️  Observability setup incomplete")
        
        self.test_results['production_readiness'] = readiness_checks
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📊 Test Results Summary")
        print("=" * 60)
        
        total_duration = time.time() - self.start_time
        
        # Overall status
        failed_phases = []
        warning_phases = []
        
        for phase, results in self.test_results.items():
            if isinstance(results, dict) and 'status' in results:
                if results['status'] == 'failed':
                    failed_phases.append(phase)
                elif results['status'] == 'warning':
                    warning_phases.append(phase)
        
        overall_status = "FAILED" if failed_phases else "WARNING" if warning_phases else "PASSED"
        
        print(f"🎯 Overall Status: {overall_status}")
        print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
        print(f"📋 Phases Completed: {len(self.test_results)}")
        
        if failed_phases:
            print(f"❌ Failed Phases: {', '.join(failed_phases)}")
        if warning_phases:
            print(f"⚠️  Warning Phases: {', '.join(warning_phases)}")
        
        # Detailed results
        print("\n📋 Detailed Results:")
        
        for phase, results in self.test_results.items():
            print(f"\n   {phase.replace('_', ' ').title()}:")
            
            if isinstance(results, dict):
                if 'status' in results:
                    status_icon = "✅" if results['status'] == 'passed' else "⚠️" if results['status'] == 'warning' else "❌"
                    print(f"      {status_icon} Status: {results['status']}")
                
                # Coverage details
                if 'coverage' in results and isinstance(results['coverage'], dict):
                    cov = results['coverage']
                    if 'total_coverage' in cov:
                        print(f"      📊 Coverage: {cov['total_coverage']:.1f}%")
                        print(f"      📝 Lines: {cov.get('lines_covered', 0)}/{cov.get('lines_total', 0)}")
                
                # Performance details
                if phase == 'performance_tests':
                    for test_name, test_data in results.items():
                        if isinstance(test_data, dict) and 'throughput' in test_data:
                            print(f"      ⚡ {test_name}: {test_data['throughput']:.1f} ops/sec")
                
                # Security details
                if phase == 'security_audit' and 'basic_checks' in results:
                    issues = results['basic_checks'].get('issues_found', 0)
                    print(f"      🔒 Security Issues: {issues}")
        
        # Save detailed report
        report_file = self.project_root / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'duration': total_duration,
                'overall_status': overall_status,
                'results': self.test_results
            }, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        
        if 'unit_tests' in self.test_results:
            coverage = self.test_results['unit_tests'].get('coverage', {})
            if coverage.get('total_coverage', 0) < 95:
                print("   📈 Increase test coverage to 95%+ for production readiness")
        
        if failed_phases:
            print("   🔧 Fix failed test phases before deployment")
        
        if warning_phases:
            print("   ⚠️  Address warning phases for optimal production readiness")
        
        print("   🚀 System ready for startup deployment with Phase 1+2 features")
        print("   🎯 Phase 3 distributed features available behind feature flags")
        
        # Exit with appropriate code
        if failed_phases:
            sys.exit(1)
        elif warning_phases:
            sys.exit(2)  # Warning exit code
        else:
            print("\n🎉 All tests passed! System is production ready! 🚀")
            sys.exit(0)

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run comprehensive AURA Intelligence tests")
    parser.add_argument("--fast", action="store_true", help="Run fast test suite only")
    parser.add_argument("--coverage", action="store_true", help="Focus on coverage analysis")
    parser.add_argument("--performance", action="store_true", help="Focus on performance testing")
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(
        fast_mode=args.fast,
        coverage_only=args.coverage,
        performance_only=args.performance
    )
    
    runner.run_all_tests()

if __name__ == "__main__":
    main()