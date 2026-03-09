---
description: Important user preferences and context for this project
---

# User Preferences

## Testing & Restart Behavior
- **The user ALWAYS does a full restart** (via `launch.bat`) after every code change before testing.
- **NEVER assume** the user forgot to restart or didn't restart properly.
- If a bug persists after changes, it is a **real code issue** — debug the actual problem instead of suggesting restarts.

## Development Environment
- The user develops on Windows.
- The project uses a `launch.bat` script that starts the full application stack (Python backend + Node.js frontend).
- The `ace-step-ui` submodule tracks the `qinglong` branch.

## Git Push Rules
- **NEVER push to `origin`** for the parent repo — `origin` points to `sdbds/ACE-Step-1.5-for-windows` (upstream, no write access).
- **Always push to `myfork`** — `myfork` points to `scragnog/ACE-Step-1.5-for-windows` (the user's fork).
- The `ace-step-ui` submodule's `origin` is `scragnog/ace-step-ui-localization` — that one is fine to push to directly.
