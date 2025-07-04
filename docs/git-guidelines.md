# Git Guidelines

**Document Contents**: Git operation rules for the lr-schedulers project. Defines commit message conventions, branch naming rules, commit granularity, etc.

**Document Purpose**: To maintain consistent Git history and facilitate change tracking and collaboration

## Basic Principles

- Divide commits into appropriate granularities
- Write commit messages in English
- Format code before pushing to remote and commit those changes

## Commit Message Conventions

Use the following verbs according to the type of change:

- **Add**: Adding functionality
- **Improve**: Improving functionality
- **Update**: Changes due to dependency updates
- **Fix**: Bug fixes
- **Refactor**: Refactoring
- **Remove**: Removing functionality
- **Document**: Creating documentation
- **Format**: Code formatting

## Branch Naming Rules

Use the following prefixes according to the type of change:

- `feature/`: Adding or improving functionality
- `fix/`: Bug fixes
- `refactor/`: Refactoring
- `docs/`: Adding or modifying documentation

## Prohibited Actions

- Do not combine multiple types of changes into one commit
