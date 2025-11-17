# GitHub Copilot Instructions

## General Guidelines

- **DO NOT** create new files unless explicitly requested by the user
- **DO NOT** modify existing files unless explicitly requested by the user
- **DO NOT** suggest changes proactively - wait for the user to ask
- When the user asks a question, provide information and explanations only
- Only take action (create/edit/delete files) when the user explicitly asks you to do so
- Always use `uv run pytest` to run tests (not `pytest` or `python -m pytest`)
- Always use `uv run python` to run Python scripts (not `python` or `python3`)
- Never created .md or README files unless explicitly requested by the user
- Do not leave comments in the code explaining what you did. That's understandable from the code changes themselves
