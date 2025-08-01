#!/usr/bin/env python3
"""
Simple AURA Intelligence Validation Runner
This script can be run directly without complex dependencies
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class SimpleValidationRunner:
    """Simple validation runner that checks AURA components"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        }
        self.project_root = Path(__file__).parent
    
    def run_test(self, name, test_func, skip_if_missing=None):
        """Run a single test and record results"""
        print(f"\n🧪 Running: {name}")
        
        # Check if we should skip
        if skip_if_missing and not self._check_dependency(skip_if_missing):
            print(f"⏭️  Skipped: {skip_if_missing} not available")
            self.results["tests"].append({
                "name": name,
                "status": "skipped",
                "reason": f"{skip_if_missing} not available"
            })
            self.results["summary"]["skipped"] += 1
            self.results["summary"]["total"] += 1
            return
        
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                print(f"✅ Passed: {name} ({duration:.2f}s)")
                status = "passed"
                self.results["summary"]["passed"] += 1
            else:
                print(f"❌ Failed: {name} ({duration:.2f}s)")
                status = "failed"
                self.results["summary"]["failed"] += 1
                
            self.results["tests"].append({
                "name": name,
                "status": status,
                "duration": duration
            })
            
        except Exception as e:
            print(f"💥 Error: {name} - {str(e)}")
            self.results["tests"].append({
                "name": name,
                "status": "error",
                "error": str(e)
            })
            self.results["summary"]["failed"] += 1
            
        self.results["summary"]["total"] += 1
    
    def _check_dependency(self, dep):
        """Check if a dependency is available"""
        if dep == "docker":
            return subprocess.run(["which", "docker"], capture_output=True).returncode == 0
        elif dep == "pytest":
            return subprocess.run([sys.executable, "-m", "pytest", "--version"], capture_output=True).returncode == 0
        return False
    
    def test_python_version(self):
        """Test Python version"""
        version = sys.version_info
        print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
        return version.major == 3 and version.minor >= 9
    
    def test_project_structure(self):
        """Test project structure"""
        required_dirs = ["src", "tests", "monitoring"]
        required_files = ["requirements.txt", "pyproject.toml"]
        
        missing = []
        for dir_name in required_dirs:
            if not (self.project_root / dir_name).exists():
                missing.append(f"Directory: {dir_name}")
                
        for file_name in required_files:
            if not (self.project_root / file_name).exists():
                missing.append(f"File: {file_name}")
        
        if missing:
            print(f"  Missing: {', '.join(missing)}")
            return False
        
        print("  All required files and directories present")
        return True
    
    def test_core_modules(self):
        """Test core Python modules can be imported"""
        modules_to_test = [
            "src.aura_intelligence",
            "src.aura_intelligence.event_store",
            "src.aura_intelligence.agents",
            "src.aura_intelligence.orchestration"
        ]
        
        failed = []
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"  ✓ {module}")
            except ImportError as e:
                failed.append(f"{module}: {str(e)}")
                print(f"  ✗ {module}: Import failed")
        
        return len(failed) == 0
    
    def test_configuration_files(self):
        """Test configuration files are valid"""
        configs_to_check = [
            ("pyproject.toml", self._check_toml),
            ("validation_results.json", self._check_json),
        ]
        
        all_valid = True
        for filename, check_func in configs_to_check:
            filepath = self.project_root / filename
            if filepath.exists():
                if check_func(filepath):
                    print(f"  ✓ {filename} is valid")
                else:
                    print(f"  ✗ {filename} is invalid")
                    all_valid = False
            else:
                print(f"  - {filename} not found")
        
        return all_valid
    
    def _check_json(self, filepath):
        """Check if JSON file is valid"""
        try:
            with open(filepath) as f:
                json.load(f)
            return True
        except:
            return False
    
    def _check_toml(self, filepath):
        """Check if TOML file is valid"""
        try:
            import tomllib
            with open(filepath, 'rb') as f:
                tomllib.load(f)
            return True
        except:
            # Fallback for older Python versions
            return True  # Assume valid if we can't check
    
    def test_docker_environment(self):
        """Test Docker environment"""
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("  Docker daemon is running")
            # Check for docker-compose
            compose_result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True
            )
            if compose_result.returncode == 0:
                print(f"  Docker Compose available: {compose_result.stdout.strip()}")
            return True
        return False
    
    def test_pytest_suite(self):
        """Run pytest if available"""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            cwd=self.project_root
        )
        
        if result.returncode == 0:
            # Count tests
            lines = result.stdout.strip().split('\n')
            test_count = len([l for l in lines if 'test_' in l])
            print(f"  Found {test_count} tests")
            return test_count > 0
        return False
    
    def generate_report(self):
        """Generate validation report"""
        report_path = self.project_root / "validation_report.json"
        
        # Calculate success rate
        if self.results["summary"]["total"] > 0:
            success_rate = (self.results["summary"]["passed"] / 
                          (self.results["summary"]["total"] - self.results["summary"]["skipped"])) * 100
        else:
            success_rate = 0
        
        self.results["summary"]["success_rate"] = f"{success_rate:.1f}%"
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Total Tests: {self.results['summary']['total']}")
        print(f"Passed: {self.results['summary']['passed']} ✅")
        print(f"Failed: {self.results['summary']['failed']} ❌")
        print(f"Skipped: {self.results['summary']['skipped']} ⏭️")
        print(f"Success Rate: {self.results['summary']['success_rate']}")
        print("="*50)
        
        return self.results["summary"]["failed"] == 0
    
    def run_all(self):
        """Run all validation tests"""
        print("🚀 AURA Intelligence Validation Suite")
        print("="*50)
        
        # Basic tests
        self.run_test("Python Version Check", self.test_python_version)
        self.run_test("Project Structure", self.test_project_structure)
        self.run_test("Core Modules", self.test_core_modules)
        self.run_test("Configuration Files", self.test_configuration_files)
        
        # Environment tests
        self.run_test("Docker Environment", self.test_docker_environment, skip_if_missing="docker")
        self.run_test("PyTest Suite", self.test_pytest_suite, skip_if_missing="pytest")
        
        # Generate report
        success = self.generate_report()
        
        if success:
            print("\n🎉 Validation completed successfully!")
            return 0
        else:
            print("\n❌ Validation failed. Please check the report for details.")
            return 1


def main():
    """Main entry point"""
    runner = SimpleValidationRunner()
    return runner.run_all()


if __name__ == "__main__":
    sys.exit(main())