---
description: Complete development workflow for feature delivery on ACE-Step
---

# ACE-Step Development Workflow

// turbo-all

> **This is the canonical workflow for ALL development on this project.**
> Follow every step in order. Do NOT skip steps. Do NOT improvise.

## Repository Layout

| Name      | URL / Path                                           | Branch |
|-----------|------------------------------------------------------|--------|
| **Local** | `D:\Ace-Step-Latest\ACE-Step-1.5-for-windows`        | `qinglong` |
| **Fork (origin)** | `https://github.com/scragnog/ACE-Step-1.5-for-windows` (remote: `origin`) | `qinglong` |
| **sdbds** | `https://github.com/sdbds/ACE-Step-1.5-for-windows` (remote: `sdbds`) | `qinglong` |
| **Upstream** | `https://github.com/ace-step/ACE-Step-1.5.git` (remote: `upstream`) | `qinglong` |
| **UI Sub (origin)** | `https://github.com/scragnog/ace-step-ui-localization` (remote: `origin` inside `ace-step-ui/`) | `qinglong` |

> **We do NOT use feature branches.** All work is done directly on `qinglong`.

---

## Phase 1 — Sync Local with Fork

Ensure local repo and submodule match what's on GitHub.

```powershell
cd D:\Ace-Step-Latest\ACE-Step-1.5-for-windows
git checkout qinglong
git pull origin qinglong
cd ace-step-ui
git checkout qinglong
git pull origin qinglong
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
# Main repo — fetch from sdbds (the upstream of our fork)
git fetch sdbds qinglong
git log --oneline qinglong..sdbds/qinglong

# UI submodule
cd ace-step-ui
git fetch origin qinglong
git log --oneline qinglong..origin/qinglong
cd ..
```

### If upstream has new commits:

> **⚠️ DO NOT use `git diff --stat qinglong origin/qinglong`.**
> Our fork has many custom files that upstream never had. A branch-level diff
> will show all those files as "deleted" — this is misleading noise, not real changes.

1. **Examine each upstream commit individually** — View the actual diff of each new commit:
   ```powershell
   # Main repo — show each commit's own diff
   git log --oneline qinglong..sdbds/qinglong
   # Then for each commit hash:
   git show <hash> --stat
   git show <hash>
   
   # UI submodule — same approach
   cd ace-step-ui
   git log --oneline qinglong..origin/qinglong
   git show <hash> --stat
   git show <hash>
   cd ..
   ```
2. **For each upstream commit, assess**:
   - What files does this specific commit touch?
   - Do any of those files overlap with our custom code (handler.py, CreatePanel.tsx, models/, etc.)?
   - Is the change a bugfix, new feature, refactor, or submodule pointer update?
   - Can it be cherry-picked cleanly, or does it need manual integration?
3. **Inform the user** with:
   - Number of upstream commits and a summary of each
   - Which files are affected per-commit
   - Overlap with our custom code
   - Recommended integration strategy per commit (cherry-pick / manual port / skip)
4. **Wait for user guidance** before proceeding with integration.

Since our fork diverges significantly from upstream, **do NOT use `git merge` as the primary integration strategy**.
Instead, use one of:
- **Cherry-pick** (`git cherry-pick <hash>`) — for clean, non-conflicting commits
- **Manual port** — read the upstream diff and apply the relevant changes by hand into our codebase
- **Skip** — for commits that only touch files we've replaced or that conflict with our architecture

> **After all cherry-picks/ports are done**, proceed to Phase 2b to establish the merge-base.

### ⚠️ CRITICAL: Submodule Merge Safety

When integrating upstream changes into the UI submodule:

1. **ALWAYS note the current HEAD commit** of the submodule BEFORE merging:
   ```powershell
   cd ace-step-ui ; git log --oneline -1 ; cd ..
   ```
2. **After integrating**, verify that ALL our custom commits are still present:
   ```powershell
   cd ace-step-ui ; git log --oneline -20 ; cd ..
   ```
3. **If force-pushing the submodule**, first verify your local branch includes ALL prior commits.
   Force-push can orphan commits that haven't been merged yet.
4. **Build the UI after integration** to catch syntax errors immediately:
   ```powershell
   cd ace-step-ui ; npm run build ; cd ..
   ```

### If no upstream changes:

Skip to Phase 3.

---

## Phase 2b — Establish Merge Base (after upstream sync)

> **WHY:** Cherry-picking creates new commits with different SHAs. GitHub's "X commits behind"
> counter is SHA-based, so it never decreases from cherry-picks alone. This merge records that
> we've consumed all upstream commits, so `git log qinglong..sdbds/qinglong` only shows
> genuinely new commits next time.

After all cherry-picks and manual ports are committed:

```powershell
cd D:\Ace-Step-Latest\ACE-Step-1.5-for-windows

# Merge upstream — this WILL conflict on submodule and possibly PS1 files
git merge sdbds/qinglong --no-edit

# Resolve ALL conflicts by keeping our versions:
# 1. Submodule pointer (ALWAYS keep ours)
git checkout --ours ace-step-ui
git add ace-step-ui

# 2. Any other conflicted files (we already have the changes via cherry-pick)
# For each conflicted file listed by `git diff --name-only --diff-filter=U`:
git checkout --ours -- <conflicted-file>
git add <conflicted-file>

# 3. Verify .gitmodules still points to our fork
cat .gitmodules
# Must show: url = https://github.com/scragnog/ace-step-ui-localization

# 4. Commit the merge resolution
git commit -m "Merge sdbds/qinglong to establish merge-base (resolve: keep our submodule + custom files)"
```

Verify the counter is now zero:
```powershell
git log --oneline qinglong..sdbds/qinglong
# Should output nothing
```

Then push as part of Phase 7.

---

## Phase 3 — Plan the Feature

1. Ask the user what feature to work on.
2. Create an implementation plan (as an artifact) covering:
   - What files will be modified/created
   - The approach and reasoning
   - Any risks or breaking changes
3. Get user approval before proceeding.

---

## Phase 4 — Develop on qinglong

All development is done directly on the `qinglong` branch.

### During development:
- Make commits frequently with descriptive messages
- Build and verify after each significant change:
  - **Python:** `python -m py_compile <file>` for syntax checks
  - **UI:** `cd ace-step-ui ; npm run build ; cd ..`
- Test via `launch.bat` (user does full restart every time)

---

## Phase 5 — User Testing and Confirmation

> ⛔ **STOP — DO NOT PROCEED BEYOND THIS POINT WITHOUT USER APPROVAL**
> You MUST call `notify_user` here and WAIT. Do NOT continue to Phase 6, 7, or 8 without a user message explicitly confirming success.

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

## Phase 7 — Push to Fork

> ⛔ **MANDATORY: PUSH APPROVAL REQUIRED**
> Before running ANY `git push` command, you MUST first call `notify_user` to request explicit push approval from the user.
> The `git push` command MUST NOT be in the same tool call batch as the `notify_user` call.
> Only proceed with the push after the user responds with explicit approval.

```powershell
# Push submodule first (if changed)
cd ace-step-ui
git push
cd ..

# Update submodule pointer and push main repo
git add ace-step-ui
git commit -m "chore: update ace-step-ui submodule"
git push
```

---

## Phase 8 — Tag the Release

After pushing, tag the commit so there's always a known-good rollback point:

```powershell
git tag -a v1.5-<feature-name> -m "feat: <brief description of feature>"
git push origin v1.5-<feature-name>
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
- Up to date with `origin/qinglong`

---

## Rules

1. **NEVER use feature branches.** All work is done directly on `qinglong`.
2. **NEVER force-push the UI submodule** without first verifying all custom commits are included.
3. **NEVER use `git add -A`** in the main repo — it will re-add ignored directories. Use explicit file paths.
4. **NEVER use `git add -f` on gitignored directories** (`.agents/`, `.agent/`, `.claude/`, `checkpoints/`, `stems_output/`, etc.). These are LOCAL-ONLY and must NEVER be committed or pushed. If a file is gitignored, that's intentional. Do NOT bypass it.
5. **ALWAYS build UI after any merge** (`npm run build`) to catch errors early.
6. **ALWAYS use `--no-recurse-submodules`** when pushing the main repo to avoid submodule push failures blocking the main push.
7. **ALWAYS tag releases** after successful feature merges for easy rollback.
7. The user ALWAYS does a full restart via `launch.bat` before testing. Never suggest "did you restart?"
8. Use PowerShell syntax (`;` not `&&`). This is a Windows environment.
9. **Use `gh` CLI for all GitHub operations** (closing issues, creating PRs, adding labels, etc.). It is already authenticated as `scragnog`. Examples:
   - `gh issue close 1 --repo scragnog/ACE-Step-1.5-for-windows --comment "Fixed in ..."`
   - `gh pr create --title "Feature X" --body "Description" --repo scragnog/ACE-Step-1.5-for-windows`
   - `gh issue list --repo scragnog/ACE-Step-1.5-for-windows --state open`
10. **NEVER run `git push` without prior user approval.** Any `git push` command MUST be preceded by a separate `notify_user` call requesting explicit push permission. The push MUST NOT be in the same tool call batch as the notification. This applies to both the main repo and the UI submodule.
