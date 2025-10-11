---
description: Research design patterns for a topic and document them
argument-hint: <topic>
allowed-tools: WebSearch, WebFetch(domain:*), Write(docs/patterns/**), Read(docs/patterns/**)
---

Topic: $1

Can you do a thorough internet search of design patterns that are used in connection with $1 and summarize them in docs/patterns/<shortened-name-or-acronym>.md?

For each pattern, include a brief note about when you would and wouldn't use a pattern. Do not include a generic discussion of the benefits and drawbacks of all patterns as a collection.

Follow these guidelines:

1. **Conduct comprehensive research**: Use multiple web searches to find:
   - Industry-standard patterns specific to this domain
   - Academic research and papers
   - Best practices from popular frameworks and libraries
   - Real-world implementations and case studies

2. **Document structure**: For each pattern include:
   - **Pattern name** as a clear heading
   - **Description**: What the pattern is and how it works
   - **When to use**: Specific scenarios where this pattern excels
   - **When NOT to use**: Situations where this pattern is inappropriate or counterproductive

3. **Pattern selection**: Focus on:
   - Patterns actually used in production systems
   - Both common and specialized patterns
   - Architecture, implementation, and optimization patterns
   - Include 10-15 patterns minimum for comprehensive coverage

4. **Avoid generic content**:
   - Don't write general "benefits of design patterns" sections
   - Don't include generic pattern catalogs (GOF patterns unless domain-specific)
   - Focus on patterns specific to the topic domain

5. **Include pattern interactions**: Add a section showing how patterns work together in real systems

6. **Cite sources**: Include a references section with links to documentation, papers, and implementations

The goal is to create a practical, domain-specific pattern reference that helps developers make informed architectural decisions.
