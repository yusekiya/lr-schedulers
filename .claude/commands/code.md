---
description: Implement code based on the plan
---

# Code Implementation Command

Implement code based on the plan.

## Instructions

$ARGUMENTS

## Execution Steps

Follow this systematic approach to proceed with implementation:

1. Understand the Context
    - Understand the project's directory structure
    - Read `docs/architecture.md` to understand the software architecture
    - Read `docs/project-requirements.md` to understand project requirements
    - Understand the planned steps
    - Understand the instructions (if any)

2. Implementation
    - Implement code following the planned steps
    - If tests exist, iterate modifications until the code passes the tests
    - Follow the style guidelines (`docs/code-style.md`)
    - Respect the common implementation procedures (`docs/common-implementation-approach.md`)
    - Refer to the list of development commands (`docs/common-commands.md`) for test execution commands
    - Execute code formatting by referring to the list of development commands (`docs/common-commands.md`)

## Output

Include the following in the output:

- **Changes**
    - Provide a title representing the change for each modification
    - Information about the location of changes (file name, line number)
    - Summary of the changes
- **Next Steps**
    - `/commit`: Commit the changes

## Important Notes

- Do NOT modify tests at this stage
- Use subagents when possible
