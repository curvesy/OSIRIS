"""
📊 Test Coverage Analysis

Analyzes test coverage for the orchestration system and identifies gaps.
Ensures we meet the 95% coverage target for Phase 1+2 components.
"""

import os
import ast
from typing import Dict, List, Tuple
from pathlib import Path

class TestCoverageAnalyzer:
    """Analyzes test coverage for orchestration components"""
    
    def __init__(self, source_dir: str = "core/src", test_dir: str = "core/tests"):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.coverage_report = {}
    
    def analyze_coverage(self) -> Dict[str, Dict[str, any]]:
        """Analyze test coverage for all orchestration components"""
        
        # Phase 1: Semantic Foundation
        semantic_coverage = self._analyze_module_coverage(
            "aura_intelligence/orchestration/semantic",
            "orchestration/semantic"
        )
        
        # Phase 2: Durable Execution
        durable_coverage = self._analyze_module_coverage(
            "aura_intelligence/orchestration/durable",
            "orchestration/durable"
        )
        
        # Phase 3: Distributed Scaling (behind feature flags)
        distributed_coverage = self._analyze_module_coverage(
            "aura_intelligence/orchestration/distributed",
            "orchestration/distributed"
        )
        
        return {
            "semantic_foundation": semantic_coverage,
            "durable_execution": durable_coverage,
            "distributed_scaling": distributed_coverage,
            "overall_summary": self._calculate_overall_coverage([
                semantic_coverage,
                durable_coverage,
                distributed_coverage
            ])
        }
    
    def _analyze_module_coverage(
        self, 
        source_module: str, 
        test_module: str
    ) -> Dict[str, any]:
        """Analyze coverage for a specific module"""
        
        source_path = self.source_dir / source_module
        test_path = self.test_dir / test_module
        
        if not source_path.exists():
            return {"error": f"Source module {source_module} not found"}
        
        # Get all Python files in source
        source_files = list(source_path.glob("*.py"))
        source_files = [f for f in source_files if f.name != "__init__.py"]
        
        # Get all test files
        test_files = []
        if test_path.exists():
            test_files = list(test_path.glob("test_*.py"))
        
        coverage_data = {
            "source_files": len(source_files),
            "test_files": len(test_files),
            "files": {}
        }
        
        for source_file in source_files:
            file_coverage = self._analyze_file_coverage(source_file, test_path)
            coverage_data["files"][source_file.name] = file_coverage
        
        # Calculate module coverage percentage
        total_functions = sum(
            data["functions"] for data in coverage_data["files"].values()
        )
        total_tested = sum(
            data["tested_functions"] for data in coverage_data["files"].values()
        )
        
        coverage_data["coverage_percentage"] = (
            (total_tested / total_functions * 100) if total_functions > 0 else 0
        )
        
        return coverage_data
    
    def _analyze_file_coverage(self, source_file: Path, test_dir: Path) -> Dict[str, any]:
        """Analyze coverage for a specific file"""
        
        # Parse source file to count functions and classes
        try:
            with open(source_file, 'r') as f:
                source_content = f.read()
            
            tree = ast.parse(source_content)
            functions = self._count_functions(tree)
            classes = self._count_classes(tree)
            
        except Exception as e:
            return {
                "error": f"Failed to parse {source_file}: {e}",
                "functions": 0,
                "classes": 0,
                "tested_functions": 0,
                "tested_classes": 0
            }
        
        # Look for corresponding test file
        test_file_name = f"test_{source_file.stem}.py"
        test_file = test_dir / test_file_name
        
        tested_functions = 0
        tested_classes = 0
        
        if test_file.exists():
            try:
                with open(test_file, 'r') as f:
                    test_content = f.read()
                
                # Count test methods (rough estimation)
                tested_functions = test_content.count("def test_")
                tested_classes = test_content.count("class Test")
                
            except Exception as e:
                pass
        
        return {
            "functions": functions,
            "classes": classes,
            "tested_functions": min(tested_functions, functions),  # Cap at actual functions
            "tested_classes": min(tested_classes, classes),
            "test_file_exists": test_file.exists(),
            "estimated_coverage": (
                (min(tested_functions, functions) / functions * 100) 
                if functions > 0 else 0
            )
        }
    
    def _count_functions(self, tree: ast.AST) -> int:
        """Count function definitions in AST"""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private methods and special methods for coverage calculation
                if not node.name.startswith('_'):
                    count += 1
        return count
    
    def _count_classes(self, tree: ast.AST) -> int:
        """Count class definitions in AST"""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                count += 1
        return count
    
    def _calculate_overall_coverage(self, module_coverages: List[Dict]) -> Dict[str, any]:
        """Calculate overall coverage statistics"""
        
        total_files = sum(
            coverage.get("source_files", 0) for coverage in module_coverages
        )
        total_test_files = sum(
            coverage.get("test_files", 0) for coverage in module_coverages
        )
        
        # Calculate weighted average coverage
        total_functions = 0
        total_tested = 0
        
        for coverage in module_coverages:
            if "files" in coverage:
                for file_data in coverage["files"].values():
                    if "functions" in file_data:
                        total_functions += file_data["functions"]
                        total_tested += file_data["tested_functions"]
        
        overall_coverage = (
            (total_tested / total_functions * 100) if total_functions > 0 else 0
        )
        
        return {
            "total_source_files": total_files,
            "total_test_files": total_test_files,
            "overall_coverage_percentage": overall_coverage,
            "meets_95_percent_target": overall_coverage >= 95.0,
            "coverage_gap": max(0, 95.0 - overall_coverage)
        }
    
    def generate_coverage_report(self) -> str:
        """Generate a formatted coverage report"""
        
        coverage_data = self.analyze_coverage()
        
        report = []
        report.append("📊 AURA Orchestration Test Coverage Report")
        report.append("=" * 50)
        
        # Overall summary
        overall = coverage_data["overall_summary"]
        report.append(f"\n🎯 Overall Coverage: {overall['overall_coverage_percentage']:.1f}%")
        report.append(f"📁 Source Files: {overall['total_source_files']}")
        report.append(f"🧪 Test Files: {overall['total_test_files']}")
        report.append(f"✅ Meets 95% Target: {'Yes' if overall['meets_95_percent_target'] else 'No'}")
        
        if not overall['meets_95_percent_target']:
            report.append(f"📈 Coverage Gap: {overall['coverage_gap']:.1f}%")
        
        # Module-by-module breakdown
        modules = [
            ("Phase 1: Semantic Foundation", coverage_data["semantic_foundation"]),
            ("Phase 2: Durable Execution", coverage_data["durable_execution"]),
            ("Phase 3: Distributed Scaling", coverage_data["distributed_scaling"])
        ]
        
        for module_name, module_data in modules:
            if "error" in module_data:
                report.append(f"\n❌ {module_name}: {module_data['error']}")
                continue
            
            coverage_pct = module_data.get("coverage_percentage", 0)
            report.append(f"\n📦 {module_name}: {coverage_pct:.1f}%")
            report.append(f"   📁 Files: {module_data['source_files']}")
            report.append(f"   🧪 Tests: {module_data['test_files']}")
            
            # File-by-file details
            if "files" in module_data:
                for filename, file_data in module_data["files"].items():
                    if "error" in file_data:
                        report.append(f"   ❌ {filename}: {file_data['error']}")
                    else:
                        file_coverage = file_data.get("estimated_coverage", 0)
                        status = "✅" if file_coverage >= 95 else "⚠️" if file_coverage >= 80 else "❌"
                        report.append(f"   {status} {filename}: {file_coverage:.1f}% "
                                    f"({file_data['tested_functions']}/{file_data['functions']} functions)")
        
        # Recommendations
        report.append("\n🎯 Recommendations:")
        
        if overall['overall_coverage_percentage'] < 95:
            report.append("   📈 Increase test coverage to meet 95% target")
            
            # Identify files needing more tests
            low_coverage_files = []
            for module_name, module_data in modules:
                if "files" in module_data:
                    for filename, file_data in module_data["files"].items():
                        if "estimated_coverage" in file_data and file_data["estimated_coverage"] < 95:
                            low_coverage_files.append((filename, file_data["estimated_coverage"]))
            
            if low_coverage_files:
                report.append("   🎯 Priority files for testing:")
                for filename, coverage in sorted(low_coverage_files, key=lambda x: x[1]):
                    report.append(f"      - {filename}: {coverage:.1f}%")
        
        if overall['total_test_files'] < overall['total_source_files']:
            missing_tests = overall['total_source_files'] - overall['total_test_files']
            report.append(f"   🧪 Create {missing_tests} missing test files")
        
        report.append("   🔧 Focus on Phase 1+2 components for startup deployment")
        report.append("   🚩 Keep Phase 3 components behind feature flags")
        
        return "\n".join(report)

def main():
    """Generate and display coverage report"""
    analyzer = TestCoverageAnalyzer()
    report = analyzer.generate_coverage_report()
    print(report)
    
    # Save report to file
    with open("test_coverage_report.txt", "w") as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: test_coverage_report.txt")

if __name__ == "__main__":
    main()