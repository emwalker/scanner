---
description: Document troubleshooting challenges and solutions
argument-hint: <description> [subdirectory]
allowed-tools: Write(docs/**), Read(docs/**)
---

Description: $1
Subdirectory: $2

Create a markdown file that documents the troubleshooting session we just completed.

Filename format:
- If subdirectory ($2) is provided: `$2/<mn>-<short-name>.md` where <mn> is month-day (e.g., "12-25")
- If subdirectory is NOT provided: `docs/issues/<year>-<month>-<short-name>.md` (e.g., "docs/issues/2025-01-<short-name>.md")

Generate <short-name> from the description: convert to lowercase, replace spaces with hyphens, remove special characters. For example, "Fix TUI Status Labels" becomes "fix-tui-status-labels.md".

Structure the document as follows:

# <Title based on description>

Brief one-sentence summary of what we fixed.

## Challenges

For each distinct challenge we encountered:

### Challenge: <Name>

**Goal**: What we were trying to accomplish.

**Failure Mode**: How it was failing or what wasn't working.

**Attempts**: Things we tried that didn't work:
- Attempt 1 and why it failed
- Attempt 2 and why it failed

**Solution**: What finally worked.

**Key Insight**: Information that would have saved time if known upfront.

---

Guidelines:
1. Focus on technical challenges, not process or meta-discussion
2. Be concise - each section should be 1-3 sentences unless more detail is needed
3. Omit challenges that were trivial (typos, missing imports, etc.)
4. Include code snippets only when they clarify the solution
5. Use matter-of-fact language without speculation or self-assessment
6. If there were multiple related challenges, group them under one Challenge heading
7. Avoid describing what the code does - focus on what the problem was and how it was solved
