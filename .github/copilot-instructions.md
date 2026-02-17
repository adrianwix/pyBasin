# GitHub Copilot Instructions

## Complete Changes Policy

When making any change, consider ALL places affected by that change:

- **Delete dead code**: Remove unused imports, functions, variables, and parameters
- **Update comments and docstrings**: If behavior changes, update all related documentation
- **Update callers**: If an API changes, update all call sites
- **Update tests**: If functionality changes, update corresponding tests
- **Update configuration**: If a value changes (e.g., `n_steps`), update related metadata/config that references it
- **Regenerate artifacts**: If source data changes, regenerate any derived files (e.g., SUMMARY.md from JSONs)

Do not make partial changes that leave the codebase in an inconsistent state.

## General Guidelines

- Always use uv to run python commands
- **DO NOT** create new files unless explicitly requested by the user
- **DO NOT** modify existing files unless explicitly requested by the user
- **DO NOT** modify pyrightconfig.json under any circumstances
- **DO NOT** manually edit pyproject.toml to add dependencies - always use `uv add` instead
- **DO NOT** suggest changes proactively - wait for the user to ask
- When the user asks a question, provide information and explanations only
- Only take action (create/edit/delete files) when the user explicitly asks you to do so
- Always use `uv run pytest` to run tests (not `pytest` or `python -m pytest`)
- Always use `uv run python` to run Python scripts (not `python` or `python3`)
- Never created .md or README files unless explicitly requested by the user
- Do not leave comments in the code explaining what you did. That's understandable from the code changes themselves
- Use `uv add <package>` to install dependencies (NOT `uv pip install`)
- Use `uv add --dev <package>` to install dev dependencies

## Typing Guidelines

### Always Type Everything

- **Always type list variables**: Use `list[type]` annotations for all list declarations
  - Correct: `edges_i: list[int] = []`
  - Incorrect: `edges_i = []`
- **Always type dict variables**: Use `dict[key_type, value_type]` annotations
  - Correct: `results: dict[str, Any] = {}`
  - Incorrect: `results = {}`
- Apply proper type hints for all variables, function parameters, and return types
- Use modern Python 3.12+ type syntax (e.g., `list[int]` not `List[int]`, `dict[str, int]` not `Dict[str, int]`)

### Return Types

- **Always specify return types** for functions
  - Correct: `def compute_stability(graph: Any) -> tuple[float, float]:`
  - Incorrect: `def compute_stability(graph: Any):`
- Use `None` for functions that don't return anything
- Use `Any` from `typing` when dealing with external libraries with poor type stubs (e.g., NetworkX)

### Type Annotations

- Prefer specific types over generic ones
  - Good: `list[float]`, `dict[str, int]`
  - Bad: `list`, `dict`, `Any` (unless necessary for external libraries)
- Use `| None` for optional types instead of `Optional[T]`
  - Correct: `seed: int | None = None`
  - Incorrect: `seed: Optional[int] = None`

## Clean Code Guidelines

### Function Arguments

- **Do not add unnecessary function arguments**
- If a value is a constant that never changes in the study, hardcode it in the function
- Only make something a parameter if it will actually vary between calls
- Example of bad design:
  ```python
  def compute_stability(graph, alpha_1=0.1232, alpha_2=4.663):  # BAD - these never change
  ```
- Example of good design:
  ```python
  def compute_stability(graph):
      alpha_1 = 0.1232  # Constants defined inside
      alpha_2 = 4.663
  ```

### Keep Functions Focused

- Functions should do one thing well
- If a function is only called once in a script, it doesn't need configurable parameters for "flexibility"
- Constants used in a study should be defined where they're used, not passed through multiple function calls

### Variable Naming

- Use descriptive lowercase names with underscores: `k_min`, `graph`, `eigenvalues`
- **Constants should be UPPERCASE with underscores**: `N_NODES`, `K_DEGREE`, `ALPHA_1`
- Avoid single uppercase letters except for well-known conventions (e.g., `N` for network size in papers)
- Be consistent with naming throughout the codebase

## Docstring Guidelines

- Use **Sphinx/rST style** for all docstrings (not Google or NumPy style)
- Use `:param name:` for parameters, `:return:` for return values, `:raises:` for exceptions
- Use `:ivar name:` and `:vartype name:` for class attributes
- Example:

  ```python
  def integrate(self, ode_system: ODESystem, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      """
      Integrate the ODE system from initial conditions.

      :param ode_system: The ODE system to integrate.
      :param y0: Initial conditions tensor of shape (batch, state_dim).
      :return: Tuple of (time points, solution trajectories).
      :raises ValueError: If y0 has incorrect shape.
      """
  ```

## CI

To check for code correctness use `bash scripts/ci.sh` and fix all the reported errors
