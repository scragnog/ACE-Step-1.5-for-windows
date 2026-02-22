---
description: Complete development workflow for feature delivery on ACE-Step
---

# ACE-Step Development Workflow

// turbo-all

> **This is the canonical workflow for ALL development on this project.**
> Follow every step in order. Do NOT skip steps. Do NOT improvise.

## Repository Layout

| Name      | URL / Path                                           | Default Branch |
|-----------|------------------------------------------------------|----------------|
| **Local** | `D:\Ace-Step-Latest\ACE-Step-1.5-for-windows`        | `qinglong`     |
| **Fork**  | `https://github.com/scragnog/ACE-Step-1.5-for-windows` (remote: `myfork`) | `qinglong` |
| **Upstream** | `https://github.com/sdbds/ACE-Step-1.5-for-windows` (remote: `origin`) | `qinglong` |
| **UI Sub (fork)** | `https://github.com/scragnog/ace-step-ui-localization` (remote: `myfork` inside `ace-step-ui/`) | `qinglong` |
| **UI Sub (upstream)** | `https://github.com/sdbds/ace-step-ui-localization` (remote: `origin` inside `ace-step-ui/`) | `qinglong` |

---

## Phase 1 — Sync Local with Fork

Ensure local repo and submodule match what's on GitHub.

```powershell
cd D:\Ace-Step-Latest\ACE-Step-1.5-for-windows
git checkout qinglong
git pull myfork qinglong
git submodule update --init --recursive
cd ace-step-ui
git checkout qinglong
git pull myfork qinglong
cd ..
```

Verify clean state:
```powershell
git status
cd ace-step-ui ; git status ; cd ..
```

Both should report "nothing to commit, working tree clean".

---

## Phase 2 — Check for Upstream Changes

```powershell
# Main repo
git fetch origin qinglong
git log --oneline qinglong..origin/qinglong

# UI submodule
cd ace-step-ui
git fetch origin qinglong
git log --oneline qinglong..origin/qinglong
cd ..
```

### If upstream has new commits:

1. **Assess complexity** — Check which files changed:
   ```powershell
   git diff --stat qinglong origin/qinglong
   cd ace-step-ui ; git diff --stat qinglong origin/qinglong ; cd ..
   ```
2. **Inform the user** with:
   - Number of upstream commits
   - Which files are affected
   - Whether any overlap with our custom code (handler.py, CreatePanel.tsx, models/, etc.)
   - Estimated conflict complexity (none / trivial / complex)
3. **Wait for user guidance** before proceeding with merge.

### ⚠️ CRITICAL: Submodule Merge Safety

When merging upstream changes into the UI submodule:

1. **ALWAYS note the current HEAD commit** of the submodule BEFORE merging:
   ```powershell
   cd ace-step-ui ; git log --oneline -1 ; cd ..
   ```
2. **After merging**, verify that ALL our custom commits are still present:
   ```powershell
   cd ace-step-ui ; git log --oneline -20 ; cd ..
   ```
3. **If force-pushing the submodule**, first verify your local branch includes ALL prior commits.
   Force-push can orphan commits that haven't been merged yet.
4. **Build the UI after merge** to catch syntax errors immediately:
   ```powershell
   cd ace-step-ui ; npm run build ; cd ..
   ```

### If no upstream changes:

Skip to Phase 3.

---

## Phase 3 — Plan the Feature

1. Ask the user what feature to work on.
2. Create an implementation plan (as an artifact) covering:
   - What files will be modified/created
   - The approach and reasoning
   - Any risks or breaking changes
3. Get user approval before proceeding.

---

## Phase 4 — Create Feature Branch and Develop

```powershell
cd D:\Ace-Step-Latest\ACE-Step-1.5-for-windows
git checkout qinglong
git checkout -b feature/<feature-name>
```

If the UI submodule will be modified, create a matching branch there:
```powershell
cd ace-step-ui
git checkout qinglong
git checkout -b feature/<feature-name>
cd ..
```

### During development:
- Make commits frequently with descriptive messages
- Build and verify after each significant change:
  - **Python:** `python -m py_compile <file>` for syntax checks
  - **UI:** `cd ace-step-ui ; npm run build ; cd ..`
- Test via `launch.bat` (user does full restart every time)

---

## Phase 5 — User Testing and Confirmation

1. Inform the user the feature is ready for testing.
2. **WAIT for explicit user confirmation** that everything works.
3. If bugs are found, fix them and return to testing.
4. Do NOT proceed to Phase 6 until the user confirms success.

---

## Phase 6 — Update Documentation

Only after user confirms the feature works:

1. Update `README.md` if the feature is user-facing or changes usage.
2. Update `FEATURES.md` using the template at the bottom of that file.
3. Commit documentation changes:
   ```powershell
   git add README.md FEATURES.md
   git commit -m "docs: document <feature-name>"
   ```

---

## Phase 7 — Push Feature Branch

```powershell
# Push submodule first (if changed)
cd ace-step-ui
git push myfork feature/<feature-name>
cd ..

# Push main repo feature branch (without auto-pushing submodule)
git add ace-step-ui  # update submodule pointer
git commit -m "chore: update ace-step-ui submodule for <feature-name>"
git push --no-recurse-submodules myfork feature/<feature-name>
```

---

## Phase 8 — Merge Feature into qinglong

```powershell
# Main repo
git checkout qinglong
git merge feature/<feature-name> --no-edit
git push myfork qinglong

# UI submodule (if it was changed)
cd ace-step-ui
git checkout qinglong
git merge feature/<feature-name> --no-edit
git push myfork qinglong
cd ..

# Update submodule pointer on qinglong
git add ace-step-ui
git commit -m "chore: update ace-step-ui after merging feature/<feature-name>"
git push myfork qinglong
```

---

## Phase 8b — Tag the Release

After a successful merge, tag the commit so there's always a known-good rollback point:

```powershell
git tag -a v1.5-<feature-name> -m "feat: <brief description of feature>"
git push myfork v1.5-<feature-name>
```

Use a consistent naming convention:
- `v1.5-guidance-modes` — new feature
- `v1.5-fix-vae-cpu` — bug fix
- `v1.5-upstream-sync-YYYYMMDD` — after upstream merges

To rollback to a tag if something breaks:
```powershell
git checkout v1.5-<last-known-good>
```

---

## Phase 9 — Final Sync Verification

```powershell
# Verify local matches remote
git status
git log --oneline -5
cd ace-step-ui ; git status ; git log --oneline -5 ; cd ..
```

Both repos should be:
- On `qinglong` branch
- Clean working tree
- Up to date with `myfork/qinglong`

---

## Rules

1. **NEVER force-push the UI submodule** without first verifying all custom commits are included.
2. **NEVER use `git add -A`** in the main repo — it will re-add ignored directories (`.agent/`, `.agents/`). Use explicit file paths.
3. **NEVER merge to qinglong without user testing confirmation.**
4. **ALWAYS build UI after any merge** (`npm run build`) to catch errors early.
5. **ALWAYS use `--no-recurse-submodules`** when pushing the main repo to avoid submodule push failures blocking the main push.
6. **ALWAYS tag releases** after successful feature merges for easy rollback.
7. The user ALWAYS does a full restart via `launch.bat` before testing. Never suggest "did you restart?"
8. Use PowerShell syntax (`;` not `&&`). This is a Windows environment.
9. **Use `gh` CLI for all GitHub operations** (closing issues, creating PRs, adding labels, etc.). It is already authenticated as `scragnog`. Examples:
   - `gh issue close 1 --repo scragnog/ACE-Step-1.5-for-windows --comment "Fixed in ..."`
   - `gh pr create --title "Feature X" --body "Description" --repo scragnog/ACE-Step-1.5-for-windows`
   - `gh issue list --repo scragnog/ACE-Step-1.5-for-windows --state open`
