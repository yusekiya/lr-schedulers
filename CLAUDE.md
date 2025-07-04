# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

lr-schedulers is a Rust library that provides learning rate schedulers for training machine learning models. It is a pure Rust implementation with no external runtime dependencies.

## Basic Principles for Project Information Management

- Descriptions should be written in a human-readable format
- Project information should be divided into appropriate granularities and documented as development documents within the `docs/` directory
- Development documents should be created in Markdown format
- This CLAUDE.md file should contain only minimal descriptions and import development documents using the `@path/to/import` syntax
- Development documents should be updated when new implementations or important decisions are made

## List of Development Documents

- Project software architecture @docs/architecture.md
- Project requirements @docs/project-requirements.md
- Development environment setup instructions @docs/developer-environment-setup.md
- List of development commands @docs/common-commands.md
- Style guidelines @docs/code-style.md
- Common implementation procedures @docs/common-implementation-approach.md
- Comprehensive testing guide @docs/testing-instructions.md
- Build instructions @docs/building-instructions.md
- Git workflow and commit conventions @docs/git-guidelines.md
- Technical insights `docs/project-knowledge.md`
- Improvement history `docs/project-improvements.md`

## Development Rules

- **Important:** Development Rule 1. When editing code, always read the following files and follow their contents:
  - `docs/architecture.md`
  - `docs/project-requirements.md`
  - `docs/code-style.md`
  - `docs/common-implementation-approach.md`
  - `docs/testing-instructions.md`
- **Important:** Development Rule 2. When running tests, always read the following files and follow their contents:
  - `docs/common-commands.md`
- **Important:** Development Rule 3. When building the project, always read the following files and follow their contents:
  - `docs/common-commands.md`
  - `docs/building-instructions.md`
- **Important:** Development Rule 4. When executing `git` or `gh` commands, always read the following files and follow their contents:
  - `docs/common-commands.md`
  - `docs/git-guidelines.md`
