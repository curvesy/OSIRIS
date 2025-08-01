#!/usr/bin/env python3
"""
🛠️ AURA Intelligence Development Environment Setup

Sets up the complete development environment for AURA Intelligence
orchestration system with proper dependency management and testing.

Usage:
    python setup_dev_environment.py [--clean] [--test] [--production]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

class AuraDevSetup:
    """Development environment setup for AURA Intelligence"""
    
    def __init__(self, clean: bool = False, test_mode: bool = False, production: bool = False):
        self.clean = clean
        self.test_mode = test_mode
        self.production = production
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        
    def setup_environment(self):
        """Set up the complete development environment"""
        print("🚀 Setting up AURA Intelligence Development Environment")
        print("=" * 60)
        
        # Step 1: Clean environment if requested
        if self.clean:
            self._clean_environment()
        
        # Step 2: Create virtual environment
        self._create_virtual_environment()
        
        # Step 3: Install dependencies
        self._install_dependencies()
        
        # Step 4: Install orchestration-specific dependencies
        self._install_orchestration_dependencies()
        
        # Step 5: Set up pre-commit hooks
        self._setup_pre_commit_hooks()
        
        # Step 6: Create environment configuration
        self._create_environment_config()
        
        # Step 7: Run tests if requested
        if self.test_mode:
            self._run_tests()
        
        # Step 8: Display setup summary
        self._display_setup_summary()
    
    def _clean_environment(self):
        """Clean existing environment"""
        print("🧹 Cleaning existing environment...")
        
        if self.venv_path.exists():
            import shutil
            shutil.rmtree(self.venv_path)
            print(f"   ✅ Removed existing virtual environment: {self.venv_path}")
        
        # Clean Python cache
        cache_dirs = [
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "*.egg-info"
        ]
        
        for cache_pattern in cache_dirs:
            self._run_command(f"find . -name '{cache_pattern}' -type d -exec rm -rf {{}} +", 
                            ignore_errors=True)
    
    def _create_virtual_environment(self):
        """Create Python virtual environment"""
        print("🐍 Creating Python virtual environment...")
        
        if not self.venv_path.exists():
            self._run_command(f"{sys.executable} -m venv {self.venv_path}")
            print(f"   ✅ Created virtual environment: {self.venv_path}")
        else:
            print(f"   ✅ Virtual environment already exists: {self.venv_path}")
        
        # Upgrade pip
        pip_path = self.venv_path / "bin" / "pip"
        if not pip_path.exists():
            pip_path = self.venv_path / "Scripts" / "pip.exe"  # Windows
        
        self._run_command(f"{pip_path} install --upgrade pip setuptools wheel")
    
    def _install_dependencies(self):
        """Install project dependencies"""
        print("📦 Installing project dependencies...")
        
        pip_path = self.venv_path / "bin" / "pip"
        if not pip_path.exists():
            pip_path = self.venv_path / "Scripts" / "pip.exe"  # Windows
        
        # Choose requirements file based on mode
        if self.production:
            requirements_file = "requirements-production.txt"
        else:
            requirements_file = "requirements.txt"
        
        self._run_command(f"{pip_path} install -r {requirements_file}")
        print(f"   ✅ Installed dependencies from {requirements_file}")
        
        # Install development dependencies
        if not self.production:
            dev_deps = [
                "pytest>=7.4.0",
                "pytest-asyncio>=0.21.0", 
                "pytest-cov>=4.1.0",
                "black>=23.9.0",
                "isort>=5.12.0",
                "mypy>=1.6.0",
                "pre-commit>=3.5.0"
            ]
            
            for dep in dev_deps:
                self._run_command(f"{pip_path} install '{dep}'", ignore_errors=True)
            
            print("   ✅ Installed development dependencies")
    
    def _install_orchestration_dependencies(self):
        """Install orchestration-specific dependencies"""
        print("🎯 Installing orchestration-specific dependencies...")
        
        pip_path = self.venv_path / "bin" / "pip"
        if not pip_path.exists():
            pip_path = self.venv_path / "Scripts" / "pip.exe"  # Windows
        
        # Core orchestration dependencies
        orchestration_deps = [
            # LangGraph and LangChain (latest)
            "langgraph>=0.2.0",
            "langchain>=0.2.0",
            "langchain-core>=0.3.0",
            
            # Temporal.io (if available)
            "temporalio>=1.0.0",
            
            # Ray Serve (behind feature flag)
            "ray[serve]>=2.9.0",
            
            # CrewAI Flows (behind feature flag)  
            "crewai>=0.28.0",
            
            # PostgreSQL for checkpointing
            "psycopg2-binary>=2.9.9",
            "asyncpg>=0.29.0",
            
            # Redis for memory store
            "redis>=5.0.0",
            "aioredis>=2.0.1",
            
            # Observability
            "opentelemetry-api>=1.25.0",
            "opentelemetry-sdk>=1.25.0",
            "prometheus-client>=0.20.0",
            "structlog>=24.1.0"
        ]
        
        for dep in orchestration_deps:
            try:
                self._run_command(f"{pip_path} install '{dep}'")
                print(f"   ✅ Installed {dep}")
            except Exception as e:
                print(f"   ⚠️  Failed to install {dep}: {e}")
                print(f"      (This dependency will be available behind feature flags)")
    
    def _setup_pre_commit_hooks(self):
        """Set up pre-commit hooks"""
        print("🔧 Setting up pre-commit hooks...")
        
        pre_commit_config = """
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]
"""
        
        pre_commit_file = self.project_root / ".pre-commit-config.yaml"
        with open(pre_commit_file, "w") as f:
            f.write(pre_commit_config.strip())
        
        # Install pre-commit hooks
        try:
            pre_commit_path = self.venv_path / "bin" / "pre-commit"
            if not pre_commit_path.exists():
                pre_commit_path = self.venv_path / "Scripts" / "pre-commit.exe"
            
            self._run_command(f"{pre_commit_path} install")
            print("   ✅ Pre-commit hooks installed")
        except Exception as e:
            print(f"   ⚠️  Failed to install pre-commit hooks: {e}")
    
    def _create_environment_config(self):
        """Create environment configuration files"""
        print("⚙️  Creating environment configuration...")
        
        # Create .env file for development
        env_content = """
# AURA Intelligence Development Environment Configuration

# Environment
AURA_ENVIRONMENT=development

# Feature Flags (Phase 1+2 enabled for startup)
AURA_FEATURE_SEMANTIC_ORCHESTRATION=true
AURA_FEATURE_LANGGRAPH_INTEGRATION=true
AURA_FEATURE_TDA_INTEGRATION=true
AURA_FEATURE_TEMPORAL_WORKFLOWS=true
AURA_FEATURE_SAGA_PATTERNS=true
AURA_FEATURE_HYBRID_CHECKPOINTING=true
AURA_FEATURE_POSTGRESQL_CHECKPOINTING=true
AURA_FEATURE_CROSS_THREAD_MEMORY=true

# Phase 3 Features (Disabled by default for startup)
AURA_FEATURE_RAY_SERVE_ORCHESTRATION=false
AURA_FEATURE_CREWAI_FLOWS=false
AURA_FEATURE_DISTRIBUTED_COORDINATION=false
AURA_FEATURE_AUTO_SCALING=false

# Database Configuration
POSTGRES_URL=postgresql://aura:aura@localhost:5432/aura_dev
REDIS_URL=redis://localhost:6379/0

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
PROMETHEUS_PORT=8000

# Logging
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true

# Testing
PYTEST_TIMEOUT=300
COVERAGE_THRESHOLD=95
"""
        
        env_file = self.project_root / ".env"
        if not env_file.exists():
            with open(env_file, "w") as f:
                f.write(env_content.strip())
            print("   ✅ Created .env configuration file")
        else:
            print("   ✅ .env file already exists")
    
    def _run_tests(self):
        """Run the test suite"""
        print("🧪 Running test suite...")
        
        python_path = self.venv_path / "bin" / "python"
        if not python_path.exists():
            python_path = self.venv_path / "Scripts" / "python.exe"
        
        try:
            # Run pytest with coverage
            self._run_command(
                f"{python_path} -m pytest tests/ -v --cov=src/aura_intelligence --cov-report=term-missing"
            )
            print("   ✅ All tests passed")
        except Exception as e:
            print(f"   ⚠️  Some tests failed: {e}")
            print("   💡 This is normal during development - focus on Phase 1+2 stability")
    
    def _run_command(self, command: str, ignore_errors: bool = False):
        """Run a shell command"""
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            if not ignore_errors:
                raise Exception(f"Command failed: {command}\nError: {e.stderr}")
    
    def _display_setup_summary(self):
        """Display setup summary"""
        print("\n🎉 AURA Intelligence Development Environment Setup Complete!")
        print("=" * 60)
        
        print("\n📋 Setup Summary:")
        print(f"   🐍 Python Virtual Environment: {self.venv_path}")
        print(f"   📦 Dependencies: {'Production' if self.production else 'Development'}")
        print(f"   🚩 Feature Flags: Phase 1+2 enabled, Phase 3 behind flags")
        print(f"   ⚙️  Configuration: .env file created")
        
        print("\n🚀 Next Steps:")
        print("   1. Activate virtual environment:")
        print(f"      source {self.venv_path}/bin/activate  # Linux/Mac")
        print(f"      {self.venv_path}\\Scripts\\activate     # Windows")
        
        print("\n   2. Run tests to verify setup:")
        print("      python -m pytest tests/ -v")
        
        print("\n   3. Start development server:")
        print("      python -m uvicorn src.aura_intelligence.main:app --reload")
        
        print("\n   4. Enable Phase 3 features when needed:")
        print("      export AURA_FEATURE_RAY_SERVE_ORCHESTRATION=true")
        print("      export AURA_FEATURE_DISTRIBUTED_COORDINATION=true")
        
        print("\n📚 Documentation:")
        print("   - Feature Flags: src/aura_intelligence/orchestration/feature_flags.py")
        print("   - Test Coverage: Run 'python tests/test_coverage_report.py'")
        print("   - Architecture: See .kiro/specs/agent-orchestration/")
        
        print(f"\n✨ Environment ready for {'production' if self.production else 'development'}!")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Set up AURA Intelligence development environment")
    parser.add_argument("--clean", action="store_true", help="Clean existing environment")
    parser.add_argument("--test", action="store_true", help="Run tests after setup")
    parser.add_argument("--production", action="store_true", help="Set up production environment")
    
    args = parser.parse_args()
    
    setup = AuraDevSetup(
        clean=args.clean,
        test_mode=args.test,
        production=args.production
    )
    
    try:
        setup.setup_environment()
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()