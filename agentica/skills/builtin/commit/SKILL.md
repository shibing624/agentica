---
name: commit
description: "Create well-formatted git commits with conventional commit messages. Handles staging, commit message generation, and push operations."
trigger: /commit
requires:
  - git
allowed-tools:
  - shell
metadata:
  emoji: "📝"
---

# Git Commit Skill

Create well-formatted git commits following conventional commit style.

## Commit Message Format

Use the conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files

### Examples

```bash
# Feature commit
git commit -m "feat(auth): add OAuth2 support"

# Bug fix with scope
git commit -m "fix(api): handle null response in user endpoint"

# Documentation
git commit -m "docs: update API documentation"

# Breaking change (use ! after type)
git commit -m "feat(api)!: change response format"
```

## Workflow

### 1. Check Status
```bash
git status
```

### 2. Stage Changes
```bash
# Stage specific files
git add path/to/file.py

# Stage all changes
git add -A

# Stage interactively (see changes)
git add -p
```

### 3. Review Staged Changes
```bash
# See what's staged
git diff --staged

# See recent commits for style reference
git log --oneline -5
```

### 4. Commit
```bash
# Simple commit
git commit -m "type(scope): description"

# Commit with body (use heredoc for multi-line)
git commit -m "$(cat <<'EOF'
feat(auth): add password reset functionality

- Add password reset endpoint
- Send reset email via SendGrid
- Implement token expiration (24h)

Closes #123
EOF
)"
```

### 5. Push
```bash
# Push to remote
git push

# Push and set upstream
git push -u origin feature-branch
```

## Best Practices

1. **Keep commits atomic**: Each commit should represent one logical change
2. **Write descriptive messages**: Explain what and why, not how
3. **Reference issues**: Use `Closes #123` or `Fixes #456` in the footer
4. **Don't commit secrets**: Never commit API keys, passwords, or credentials
5. **Review before committing**: Always run `git diff --staged` before committing

## Common Issues

### Amend last commit
```bash
# Change commit message
git commit --amend -m "new message"

# Add forgotten files to last commit
git add forgotten-file.py
git commit --amend --no-edit
```

### Undo last commit (keep changes)
```bash
git reset --soft HEAD~1
```

### Undo last commit (discard changes)
```bash
git reset --hard HEAD~1
```
