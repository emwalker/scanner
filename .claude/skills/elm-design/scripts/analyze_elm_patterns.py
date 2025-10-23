#!/usr/bin/env python3
"""
Analyze Rust code for Elm Architecture patterns and anti-patterns.

Usage:
    python3 analyze_elm_patterns.py <file_path>

Looks for:
- Model structs and their fields
- Update functions and message handling
- View functions
- Anti-patterns (boolean flags, mixed concerns, etc.)
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class ElmPatternAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = self.file_path.read_text()
        self.lines = self.content.split('\n')

    def find_models(self) -> List[Tuple[int, str]]:
        """Find struct definitions that might be Models."""
        patterns = [
            r'pub struct\s+(\w+)\s*\{',
            r'struct\s+(\w+)\s*\{',
        ]

        results = []
        for i, line in enumerate(self.lines, 1):
            for pattern in patterns:
                if match := re.search(pattern, line):
                    results.append((i, match.group(1)))
        return results

    def find_update_functions(self) -> List[Tuple[int, str]]:
        """Find update functions."""
        patterns = [
            r'fn\s+update\s*\(',
            r'pub fn\s+update\s*\(',
        ]

        results = []
        for i, line in enumerate(self.lines, 1):
            for pattern in patterns:
                if re.search(pattern, line):
                    results.append((i, 'update'))
        return results

    def find_view_functions(self) -> List[Tuple[int, str]]:
        """Find view functions."""
        patterns = [
            r'fn\s+view\s*\(',
            r'pub fn\s+view\s*\(',
            r'fn\s+render\s*\(',
            r'pub fn\s+render\s*\(',
        ]

        results = []
        for i, line in enumerate(self.lines, 1):
            for pattern in patterns:
                if re.search(pattern, line):
                    results.append((i, re.search(pattern, line).group().strip()))
        return results

    def find_message_enums(self) -> List[Tuple[int, str]]:
        """Find Message enum definitions."""
        in_enum = False
        enum_name = None
        start_line = 0

        results = []

        for i, line in enumerate(self.lines, 1):
            if 'enum' in line and ('Message' in line or 'Msg' in line or 'Event' in line):
                results.append((i, re.search(r'enum\s+(\w+)', line).group(1)))

        return results

    def detect_boolean_flag_anti_pattern(self) -> List[Tuple[int, str]]:
        """Detect separate boolean flags instead of enum states."""
        issues = []

        # Look for multiple boolean fields that might represent state
        bool_fields = []
        for i, line in enumerate(self.lines, 1):
            if re.search(r'pub\s+\w*is_\w+:\s*bool|pub\s+\w*has_\w+:\s*bool', line):
                bool_fields.append((i, line.strip()))

        # If we find multiple boolean flags in same struct, flag it
        if len(bool_fields) >= 2:
            for line_no, text in bool_fields[:2]:  # Show first two
                issues.append((line_no, f"Possible state as booleans: {text}"))

        return issues

    def detect_side_effects_in_update(self) -> List[Tuple[int, str]]:
        """Detect potential side effects inside update functions."""
        issues = []

        update_funcs = self.find_update_functions()
        if not update_funcs:
            return issues

        # Simple heuristic: look for file I/O, HTTP calls, etc. in functions named update
        patterns = [
            (r'\.lock\(\)', 'Mutex lock in update'),
            (r'\.write\(', 'File write in update'),
            (r'\.flush\(', 'File flush in update'),
            (r'tokio::', 'Async operation in update'),
            (r'std::fs::', 'File I/O in update'),
        ]

        for func_line, _ in update_funcs:
            # Look within next 50 lines
            for i in range(func_line - 1, min(func_line + 50, len(self.lines))):
                for pattern, description in patterns:
                    if re.search(pattern, self.lines[i]):
                        issues.append((i + 1, description))

        return issues

    def detect_side_effects_in_view(self) -> List[Tuple[int, str]]:
        """Detect potential side effects inside view functions."""
        issues = []

        view_funcs = self.find_view_functions()
        if not view_funcs:
            return issues

        patterns = [
            (r'\.write\(', 'File write in view'),
            (r'\.lock\(\)', 'Mutex lock in view'),
            (r'tokio::', 'Async operation in view'),
            (r'std::fs::', 'File I/O in view'),
        ]

        for func_line, _ in view_funcs:
            # Look within next 30 lines
            for i in range(func_line - 1, min(func_line + 30, len(self.lines))):
                for pattern, description in patterns:
                    if re.search(pattern, self.lines[i]):
                        issues.append((i + 1, description))

        return issues

    def analyze(self) -> Dict:
        """Run all analyses and return results."""
        return {
            'file': str(self.file_path),
            'models': self.find_models(),
            'update_functions': self.find_update_functions(),
            'view_functions': self.find_view_functions(),
            'message_enums': self.find_message_enums(),
            'anti_patterns': {
                'boolean_flags': self.detect_boolean_flag_anti_pattern(),
                'side_effects_in_update': self.detect_side_effects_in_update(),
                'side_effects_in_view': self.detect_side_effects_in_view(),
            }
        }

    def report(self) -> str:
        """Generate a human-readable report."""
        analysis = self.analyze()
        lines = []

        lines.append(f"Elm Architecture Analysis: {analysis['file']}\n")

        # Models
        if analysis['models']:
            lines.append("Models (structs that may represent state):")
            for line_no, name in analysis['models']:
                lines.append(f"  Line {line_no}: {name}")
            lines.append("")

        # Message enums
        if analysis['message_enums']:
            lines.append("Message/Event enums:")
            for line_no, name in analysis['message_enums']:
                lines.append(f"  Line {line_no}: {name}")
            lines.append("")

        # Update functions
        if analysis['update_functions']:
            lines.append("Update functions:")
            for line_no, name in analysis['update_functions']:
                lines.append(f"  Line {line_no}: {name}")
            lines.append("")

        # View functions
        if analysis['view_functions']:
            lines.append("View/Render functions:")
            for line_no, name in analysis['view_functions']:
                lines.append(f"  Line {line_no}: {name}")
            lines.append("")

        # Anti-patterns
        all_issues = []
        for category, issues in analysis['anti_patterns'].items():
            all_issues.extend(issues)

        if all_issues:
            lines.append("⚠️  Potential Issues:")
            for line_no, description in sorted(all_issues):
                lines.append(f"  Line {line_no}: {description}")
            lines.append("")
        else:
            lines.append("✓ No obvious anti-patterns detected")

        return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_elm_patterns.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    analyzer = ElmPatternAnalyzer(file_path)
    print(analyzer.report())
