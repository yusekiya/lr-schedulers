---
description: Create a pull request
---

# Pull Request Creation Command

Create a pull request.

## Instructions

$ARGUMENTS

## Execution Steps

Follow this systematic approach:

1. Understand the Context
    - Read `docs/git-guidelines.md` to understand the project's Git conventions
    - Understand the changes and reasons from previous output results
    - Understand the instructions (if any)

2. Create Pull Request
    - Provide an appropriate title that represents the changes
    - Provide a clear description of the implemented changes
    - Reference the relevant issue number in the description: `Close #issue-number`
    - Command template: `gh pr create --draft --title "Your descriptive title" --body "Your PR description" --base main`

## Output

Include the following in the output:

- **Summary**
    - Pull request title and description
- **Next Steps**
    - `gh pr view --web`: View the pull request in browser

## Important Notes

- Always use English for pull request titles and descriptions
- Do NOT add or modify any code
