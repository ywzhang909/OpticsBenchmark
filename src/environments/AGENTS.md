# ENVIRONMENTS MODULE

**Path:** `src/environments/` — Execution environments for optical design agents.

## OVERVIEW
Abstract sandbox interface + implementations. Currently stub-level; designed for shell execution (LocalEnvironment) and Zemax OpticStudio (ZOSAPIEnvironment via PythonNET).

## FILES
| File | Lines | Role |
|------|-------|------|
| `base_env.py` | 370 | `BaseEnvironment` (ABC) + `LocalEnvironment` (subprocess shell) + `EnvironmentConfig` / `EnvironmentResponse` dataclasses + factory |
| `zos_env.py` | 418 | `ZOSAPIEnvironment` — ZOS-API connection, lens load, MTF/spot analysis, optimization stubs |
| `__init__.py` | 29 | Exports 7 symbols |

## KEY PATTERNS
- **ABC + dataclass**: `BaseEnvironment` defines abstract `setup()`, `teardown()`, `execute()`. Config and response are dataclasses.
- **Function-call schema**: `get_available_actions()` returns OpenAI tool-call format JSON for agent function calling.
- **Command prefix routing**: `ZOSAPIEnvironment.execute()` dispatches on prefix: `python:`, `zemax:`, else shell.

## KNOWN ISSUES
- `ZOSAPIEnvironment` is **stub-only** — all high-level methods (load_lens, analyze_mtf, optimize) return placeholder data. Requires actual PythonNET + OpticStudio to function.
- `zos_env.py` imports `socket` but connection check is unused in main flow.
- `create_environment()` factory only returns `LocalEnvironment` — no ZOS-API path.
- No Docker or CODE V environments despite configs referencing them.
