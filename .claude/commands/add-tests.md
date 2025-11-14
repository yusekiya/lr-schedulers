---
description: Create tests for the features to be implemented
---

# Test Creation Command

Create tests for the features to be implemented.

## Instructions

$ARGUMENTS

## Execution Steps

Follow this systematic approach to create tests:

1. Understand the Context
    - Understand the project's directory structure
    - Read `docs/architecture.md` to understand the software architecture
    - Read `docs/project-requirements.md` to understand project requirements
    - Understand the planned steps
    - Understand the instructions (if any)

2. Create Tests
    - Create tests for the features to be implemented
    - Follow the style guidelines (`docs/code-style.md`)
    - Follow the testing guide (`docs/testing-instructions.md`)
    - Execute code formatting by referring to the list of development commands (`docs/common-commands.md`)

3. Confirm with User
    - Pause processing after creating tests and output a list of created tests
    - The list should include the following information:
        - Test name (file name)
        - Class or function name that the test targets
        - Brief description of the test purpose
    - If the user requests test modifications, return to step 2 to modify the tests

4. Run Added Tests
    - Run tests to verify the success or failure of added tests
    - Refer to the list of development commands (`docs/common-commands.md`) for test execution commands

## Output

Include the following in the output:

- **Added Test List**
    - Test name (file name)
    - Class or function name that the test targets
    - Brief description of the test purpose
    - Test success or failure
- **Next Steps**
    - `/commit`: Commit the tests
    - `/code`: Begin implementation based on the plan

## Important Notes

- Do NOT modify code other than the tests being added
- Use subagents when possible
