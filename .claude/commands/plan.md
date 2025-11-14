---
description: Design specific steps to resolve the specified issue
---

# Plan Design Command

Design specific steps to resolve the specified issue.

## Instructions

$ARGUMENTS

## Execution Steps

Follow this systematic approach to design specific steps:

1. Understand the Context
    - Understand the project's directory structure
    - Read `docs/architecture.md` to understand the software architecture
    - Read `docs/project-requirements.md` to understand project requirements
    - Understand the root cause of the problem identified in the investigation
    - Understand the instructions (if any)

2. Design the Solution
    - Design a solution for the root cause
    - Be careful not to simply address symptoms
    - Ensure the solution follows project conventions and patterns

3. Formulate the Steps
    - Determine specific, step-by-step procedures to implement the solution

## Output

Include the following in the output:

- **Objective**
    - An overview of the issue to be resolved or the root cause of the problem
- **Plan**
    - Specific steps with the designed plan broken down by phase
    - Include the reason why each program modification is necessary
- **Next Steps**
    - If the designed plan is reasonable, it is recommended to record the output in a file or GitHub issue
    - `/create-branch`: Create a branch for committing changes
    - `/add-tests`: Create tests
    - `/code`: Begin implementation based on the plan

## Important Notes

- Use "think harder" mode unless otherwise instructed
- Do NOT add or modify any code during this planning phase
