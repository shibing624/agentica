---
name: github
description: "Interact with GitHub using the `gh` CLI. Use `gh issue`, `gh pr`, `gh run`, and `gh api` for issues, PRs, CI runs, and advanced queries."
trigger: /github
requires:
  - gh
allowed-tools:
  - shell
metadata:
  emoji: "🐙"
---

# GitHub Skill

Use the `gh` CLI to interact with GitHub. Always specify `--repo owner/repo` when not in a git directory, or use URLs directly.

## Pull Requests

Check CI status on a PR:
```bash
gh pr checks 55 --repo owner/repo
```

List recent workflow runs:
```bash
gh run list --repo owner/repo --limit 10
```

View a run and see which steps failed:
```bash
gh run view <run-id> --repo owner/repo
```

View logs for failed steps only:
```bash
gh run view <run-id> --repo owner/repo --log-failed
```

## Issues

Create a new issue:
```bash
gh issue create --repo owner/repo --title "Bug: ..." --body "Description..."
```

List open issues:
```bash
gh issue list --repo owner/repo --state open
```

Close an issue:
```bash
gh issue close <issue-number> --repo owner/repo
```

## API for Advanced Queries

The `gh api` command is useful for accessing data not available through other subcommands.

Get PR with specific fields:
```bash
gh api repos/owner/repo/pulls/55 --jq '.title, .state, .user.login'
```

## JSON Output

Most commands support `--json` for structured output. You can use `--jq` to filter:

```bash
gh issue list --repo owner/repo --json number,title --jq '.[] | "\(.number): \(.title)"'
```

## Common Patterns

### Review a PR
```bash
# View PR details
gh pr view <pr-number> --repo owner/repo

# Check out PR locally
gh pr checkout <pr-number>

# Approve PR
gh pr review <pr-number> --approve --repo owner/repo

# Request changes
gh pr review <pr-number> --request-changes --body "Please fix..." --repo owner/repo
```

### Create a PR
```bash
# Create PR from current branch
gh pr create --title "Feature: ..." --body "Description..."

# Create draft PR
gh pr create --draft --title "WIP: ..."
```
