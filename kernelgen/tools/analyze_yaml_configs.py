#!/usr/bin/env python3
"""
Script to analyze YAML configuration files for quality and optimization opportunities.
Helps identify issues that could affect LLM-based GPU kernel optimization.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Import algorithm categories from generate script
ALGORITHM_CATEGORIES = {
    'eigenvalue': ['syev', 'heev', 'syevd', 'heevd', 'syevx', 'heevx', 'sytrd', 'hetrd',
                   'sytd2', 'hetd2', 'orgtr', 'ungtr', 'ormtr', 'unmtr', 'sterf', 'stedc', 'steqr',
                   'syevdj', 'heevdj', 'syevj', 'heevj', 'stedcj', 'stedcx'],
    'svd': ['gesvd', 'gesvdx', 'gesdd', 'gebrd', 'gebd2', 'orgbr', 'ungbr', 'ormbr', 'unmbr',
            'bdsqr', 'bdsvdx', 'gesvdj'],
    'qr_decomp': ['geqrf', 'geqr2', 'orgqr', 'ungqr', 'ormqr', 'unmqr', 'gerqf'],
    'lq_decomp': ['gelqf', 'gelq2', 'orglq', 'unglq', 'ormlq', 'unmlq', 'geqlf'],
    'lu_decomp': ['getrf', 'getf2', 'getri', 'getrs'],
    'cholesky': ['potrf', 'potf2', 'potri', 'potrs', 'pstrf', 'pstf2', 'posv'],
    'generalized_eigen': ['sygv', 'hegv', 'sygvd', 'hegvd', 'sygvx', 'hegvx', 'sygs2', 'hegs2',
                         'sygst', 'hegst', 'sygvdj', 'hegvdj', 'sygvj', 'hegvj', 'sygvdx', 'hegvdx'],
    'triangular': ['trtri', 'trsm', 'trmm'],
    'auxiliary': ['lacgv', 'larf', 'larfb', 'larfg', 'larft', 'lasr', 'latrd', 'labrd'],
    'solvers': ['gesv', 'gels', 'geblttrf', 'geblttrs'],
    'factorization': ['sytrf', 'sytf2', 'sytri', 'sytrs']
}

class YAMLAnalyzer:
    def __init__(self, yaml_dir: str):
        self.yaml_dir = yaml_dir
        self.analyses = []
        self.statistics = defaultdict(list)

    def detect_algorithm_category(self, function_name: str) -> str:
        """Detect the algorithm category of a function."""
        function_name = function_name.lower()
        for category, patterns in ALGORITHM_CATEGORIES.items():
            for pattern in patterns:
                if pattern in function_name:
                    return category
        return 'unknown'

    def analyze_cross_contamination(self, function_name: str, kernels: List[str]) -> List[str]:
        """Find kernels that belong to different algorithm categories."""
        target_category = self.detect_algorithm_category(function_name)
        contaminated = []

        for kernel in kernels:
            kernel_category = self.detect_algorithm_category(kernel)
            if (kernel_category != 'unknown' and
                kernel_category != 'auxiliary' and
                target_category != 'unknown' and
                kernel_category != target_category):
                contaminated.append((kernel, kernel_category))

        return contaminated

    def analyze_kernel_patterns(self, kernels: List[str]) -> Dict[str, int]:
        """Analyze patterns in kernel names to identify groups."""
        patterns = defaultdict(int)

        # Common prefixes
        prefixes = ['rocsolver_', 'set_', 'get_', 'copy', 'init_', 'restore_', 'scalar_']
        for kernel in kernels:
            for prefix in prefixes:
                if kernel.startswith(prefix):
                    patterns[f'prefix_{prefix}'] += 1
                    break

            # Algorithm-specific patterns
            if '_kernel' in kernel:
                patterns['gpu_kernels'] += 1
            if '_template' in kernel:
                patterns['template_functions'] += 1
            if '_impl' in kernel:
                patterns['impl_functions'] += 1

        return patterns

    def analyze_file_dependencies(self, source_files: List[str]) -> Dict[str, int]:
        """Analyze source file dependencies by category."""
        categories = {
            'core_lapack': 0,
            'auxiliary': 0,
            'include': 0,
            'specialized': 0,
            'other': 0
        }

        for file in source_files:
            if 'lapack/roclapack_' in file:
                categories['core_lapack'] += 1
            elif 'auxiliary/rocauxiliary_' in file:
                categories['auxiliary'] += 1
            elif 'include/' in file:
                categories['include'] += 1
            elif 'specialized/' in file:
                categories['specialized'] += 1
            else:
                categories['other'] += 1

        return categories

    def check_command_quality(self, config: Dict) -> List[str]:
        """Check quality of test and benchmark commands."""
        issues = []

        # Check compile command
        compile_cmds = config.get('compile_command', [])
        if not compile_cmds:
            issues.append("Missing compile command")
        elif '--architecture gfx942' not in ' '.join(compile_cmds):
            issues.append("Compile command missing GPU architecture")

        # Check correctness command
        test_cmds = config.get('correctness_command', [])
        if not test_cmds:
            issues.append("Missing correctness test command")
        elif '--gtest_filter=' not in ' '.join(test_cmds):
            issues.append("Test command missing gtest filter")

        # Check performance command
        perf_cmds = config.get('performance_command', [])
        if not perf_cmds:
            issues.append("Missing performance benchmark command")
        elif 'rocsolver-bench' not in ' '.join(perf_cmds):
            issues.append("Performance command not using rocsolver-bench")

        return issues

    def analyze_single_yaml(self, yaml_file: str) -> Dict:
        """Analyze a single YAML configuration file."""
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)

        base_name = Path(yaml_file).stem.replace('roclapack_', '')

        analysis = {
            'name': base_name,
            'file': yaml_file,
            'issues': [],
            'warnings': [],
            'stats': {}
        }

        # Basic statistics
        source_files = config.get('source_file_path', [])
        gen_files = config.get('gen_file_path', [])
        kernels = config.get('target_kernel_functions', [])

        analysis['stats'] = {
            'source_files': len(source_files),
            'gen_files': len(gen_files),
            'kernels': len(kernels),
            'algorithm_category': self.detect_algorithm_category(base_name)
        }

        # Check for cross-contamination
        contaminated = self.analyze_cross_contamination(base_name, kernels)
        if contaminated:
            analysis['issues'].append(f"Cross-contamination: {len(contaminated)} kernels from wrong categories")
            analysis['contaminated_kernels'] = contaminated

        # Check kernel count thresholds
        if len(kernels) < 3:
            analysis['warnings'].append(f"Very few kernels ({len(kernels)}), might be missing dependencies")
        elif len(kernels) > 80:
            analysis['warnings'].append(f"Too many kernels ({len(kernels)}), might affect LLM context")

        # Check source file count
        if len(source_files) > 40:
            analysis['issues'].append(f"Excessive source files ({len(source_files)}), could overwhelm LLM")
        elif len(source_files) < 2:
            analysis['warnings'].append(f"Very few source files ({len(source_files)})")

        # Analyze file dependencies
        file_categories = self.analyze_file_dependencies(source_files)
        analysis['file_categories'] = file_categories

        if file_categories['include'] > 15:
            analysis['warnings'].append(f"Too many include files ({file_categories['include']})")

        # Analyze kernel patterns
        kernel_patterns = self.analyze_kernel_patterns(kernels)
        analysis['kernel_patterns'] = kernel_patterns

        # Check command quality
        command_issues = self.check_command_quality(config)
        if command_issues:
            analysis['warnings'].extend(command_issues)

        # Check for duplicate entries
        if len(set(kernels)) != len(kernels):
            analysis['issues'].append("Duplicate kernels detected")
        if len(set(source_files)) != len(source_files):
            analysis['issues'].append("Duplicate source files detected")

        return analysis

    def analyze_all(self) -> None:
        """Analyze all YAML files in the directory."""
        yaml_files = sorted(Path(self.yaml_dir).glob("roclapack_*.yaml"))

        print(f"Analyzing {len(yaml_files)} YAML configuration files...\n")

        for yaml_file in yaml_files:
            analysis = self.analyze_single_yaml(str(yaml_file))
            self.analyses.append(analysis)

            # Collect statistics
            self.statistics['kernel_counts'].append(analysis['stats']['kernels'])
            self.statistics['file_counts'].append(analysis['stats']['source_files'])
            self.statistics['categories'].append(analysis['stats']['algorithm_category'])

    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        print("=" * 80)
        print("YAML Configuration Analysis Report")
        print("=" * 80)

        # Overall statistics
        print("\n📊 Overall Statistics:")
        print(f"  Total YAML files: {len(self.analyses)}")
        print(f"  Average kernels per function: {sum(self.statistics['kernel_counts'])/len(self.statistics['kernel_counts']):.1f}")
        print(f"  Average source files per function: {sum(self.statistics['file_counts'])/len(self.statistics['file_counts']):.1f}")

        # Category distribution
        category_counts = defaultdict(int)
        for cat in self.statistics['categories']:
            category_counts[cat] += 1

        print("\n📂 Algorithm Category Distribution:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count} functions")

        # Critical issues
        critical_issues = []
        warnings_list = []

        for analysis in self.analyses:
            if analysis['issues']:
                critical_issues.append(analysis)
            if analysis['warnings']:
                warnings_list.append(analysis)

        if critical_issues:
            print(f"\n❌ Critical Issues ({len(critical_issues)} functions):")
            for analysis in critical_issues[:10]:  # Show top 10
                print(f"\n  {analysis['name']}:")
                for issue in analysis['issues']:
                    print(f"    - {issue}")
                if 'contaminated_kernels' in analysis:
                    print(f"      Contaminated kernels:")
                    for kernel, category in analysis['contaminated_kernels'][:5]:
                        print(f"        • {kernel} (from {category})")

        # Functions with extreme kernel counts
        print("\n📈 Functions with Extreme Kernel Counts:")
        sorted_by_kernels = sorted(self.analyses, key=lambda x: x['stats']['kernels'], reverse=True)

        print("  Highest kernel counts:")
        for analysis in sorted_by_kernels[:5]:
            print(f"    - {analysis['name']}: {analysis['stats']['kernels']} kernels")

        print("  Lowest kernel counts:")
        for analysis in sorted_by_kernels[-5:]:
            print(f"    - {analysis['name']}: {analysis['stats']['kernels']} kernels")

        # Functions with too many source files
        print("\n📁 Functions with Excessive Source Files:")
        sorted_by_files = sorted(self.analyses, key=lambda x: x['stats']['source_files'], reverse=True)
        for analysis in sorted_by_files[:5]:
            if analysis['stats']['source_files'] > 30:
                file_cats = analysis['file_categories']
                print(f"  - {analysis['name']}: {analysis['stats']['source_files']} files")
                print(f"    (core: {file_cats['core_lapack']}, aux: {file_cats['auxiliary']}, include: {file_cats['include']})")

        # Summary recommendations
        print("\n💡 Key Recommendations for LLM Context Optimization:")
        print("  1. Address cross-contamination issues to improve algorithm focus")
        print("  2. Reduce include file dependencies where possible")
        print("  3. Consider splitting very large functions (>60 kernels) into sub-components")
        print("  4. Verify functions with <5 kernels aren't missing critical dependencies")
        print("  5. Prioritize essential files over generic utilities for better LLM understanding")

        # Export detailed report
        self.export_detailed_report()

    def export_detailed_report(self) -> None:
        """Export detailed analysis to a file."""
        output_file = os.path.join(self.yaml_dir, "yaml_analysis_report.txt")

        with open(output_file, 'w') as f:
            f.write("Detailed YAML Configuration Analysis\n")
            f.write("=" * 80 + "\n\n")

            for analysis in sorted(self.analyses, key=lambda x: len(x['issues']), reverse=True):
                f.write(f"\n{analysis['name']}\n")
                f.write("-" * len(analysis['name']) + "\n")
                f.write(f"  Category: {analysis['stats']['algorithm_category']}\n")
                f.write(f"  Kernels: {analysis['stats']['kernels']}\n")
                f.write(f"  Source files: {analysis['stats']['source_files']}\n")

                if analysis['issues']:
                    f.write("  Issues:\n")
                    for issue in analysis['issues']:
                        f.write(f"    - {issue}\n")

                if analysis['warnings']:
                    f.write("  Warnings:\n")
                    for warning in analysis['warnings']:
                        f.write(f"    - {warning}\n")

        print(f"\n📄 Detailed report saved to: {output_file}")

def main():
    yaml_dir = "kernelgen"
    analyzer = YAMLAnalyzer(yaml_dir)
    analyzer.analyze_all()
    analyzer.generate_report()

if __name__ == "__main__":
    main()