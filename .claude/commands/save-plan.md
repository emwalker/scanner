---
description: Save a plan to docs/plans/<lmn>-<short-name>.md
argument-hint: <number> <short-name>
allowed-tools: Write(docs/plans/**), Read(docs/plans/**)
---

Plan number: $1
Short name: $2

Save the discussion of the plan that we just talked about to docs/plans/$1-$2.md following this structure:

# Plan $1: <Title>

## Guidance for Updates

When updating this plan as work progresses, avoid adding:
- Lists of accomplishments or completion summaries
- Self-aggrandizement or subjective quality assessments
- Rationales and benefits sections (unless specifically requested)
- Speculation about future improvements or possibilities
- Time estimates or risk assessments

Keep updates matter-of-fact and focused on concrete technical details. Simply check off completed tasks and add technical notes as needed.

## Proposal 1: <Name>

<Concise technical description of what will be implemented>

### Tasks
- [ ] Concrete task 1
- [ ] Concrete task 2
- [ ] Concrete task 3

## Proposal 2: <Name>

<Concise technical description of what will be implemented>

### Tasks
- [ ] Concrete task 1
- [ ] Concrete task 2

---

Guidelines:
1. Each proposal gets its own section with concrete tasks
2. Tasks should be checkboxes that can be checked off as work completes
3. Omit priorities, time estimates, risk assessments, and speculative content
4. Keep descriptions technical and matter-of-fact
5. Do not include rationales or benefits unless specifically requested
6. Focus on "what" will be done, not "why" or "how great it will be"
7. Run `make lint` and `make test` after each proposal is completed and and fix any issues
