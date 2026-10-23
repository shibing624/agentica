# 多会话共用一个仓库 (`Worktree`)

当你同时开了几个 Agentica 会话（比如两个终端窗口，或者一个在终端、一个在手机微信里遥控）修改**同一个 Git 仓库**时，如果直接站在同一个目录里改，很容易互相覆盖未提交的文件，或者争抢 `.git/index.lock`。

Agentica 提供两样东西来解决多会话并行开发：
1. **`/list-agents`（或 `/peers`）**：不用发消息打听，直接看到其他会话正在哪个分支、改了哪些文件。
2. **`/worktree`**：一键为当前任务拉出独立的目录和 `wt/<任务名>` 分支，改完一键合回主分支并清理。

---

## 一、查看其他会话在改什么：`/list-agents`

在任意 CLI 会话里敲 `/list-agents`（或 `/peers`），就能看到本机所有正在运行的会话及其 Git 状态：

```text
> /list-agents
agentica-52 [peer=52d79d86]
  status: running a turn
  cwd: /Users/me/Codes/agentica
  git: main @ 8ca321e · 3 dirty
  dirty: CHANGELOG.md, agentica/gateway/main.py, docs/advanced/gateway.md
  working on: 修复微信重登逻辑
```

- `git:` 显示当前分支、最新 commit、相对本地主分支（`main` / `master`）的领先/落后提交数，以及未提交文件数。
- `dirty:` 列出对方正在修改的具体文件路径。Agent 在调用 `list_agents` 工具时也能看到这份列表，从而主动避开对方正在改的文件。

---

## 二、用 `/worktree` 隔离任务分支

### 方式 1：在已打开的会话里管理 worktree

不需要重启终端，直接在会话里用斜杠命令（或直接用自然语言让 Agent 调 `worktree` 工具）：

```text
> /worktree                        # 查看当前仓库的所有 worktree
> /worktree new auth-refactor      # 新建 .agentica/worktrees/auth-refactor 目录与 wt/auth-refactor 分支
> /worktree merge auth-refactor    # 将 wt/auth-refactor 合回本地主分支，并自动删除该 worktree 目录
> /worktree remove auth-refactor   # 直接删除该 worktree 目录（未合入的提交仍保留在分支上）
```

### 方式 2：启动时直接进入独立 worktree

如果你准备开一个新终端专门做某项任务，可以在启动时直接加上 `--worktree`：

```bash
agentica --worktree auth-refactor
```

- 启动后自动创建并进入 `<仓库>/.agentica/worktrees/auth-refactor`（分支名为 `wt/auth-refactor`）。
- 退出该终端时：如果这个 worktree 里没有任何未提交改动或独有提交，会自动清理目录；如果有你的改动或新提交，会自动保留在磁盘上。

---

## 三、目录位置与 `.env` 自动共享

默认情况下，所有任务树都放在仓库内的 `.agentica/worktrees/<任务名>` 下，并自动写入 `.agentica/worktrees/.gitignore`（`*`），因此：
- 不会弄脏主仓库的 `git status`，也不会修改你仓库里原有的 `.gitignore`。
- 主仓库里的 `.env` 文件会自动以软链接（symlink）方式链进新 worktree，不需要再手动拷一遍 API Key。

如果你想把 worktree 放到仓库外面，或者需要额外软链其他未入库文件（如 `.env.local`），可以在 `~/.agentica/config.yaml` 中配置：

```yaml
settings:
  worktree:
    root: sibling                    # 放到上级目录：../<仓库名>-<任务名>
    # root: ~/worktrees              # 或指定集中存放的绝对路径：~/worktrees/<仓库名>/<任务名>
    link: [".env", ".env.local"]     # 新建 worktree 时自动软链的文件列表
```

---

## 四、常见问题

- **`/worktree merge` 提示拒绝合并怎么办？**
  为了防止冲掉未保存的工作，合并前要求**任务 worktree 内所有改动已 commit**，且**主目录当前在主分支上、工作区干净**。先把未提交的修改提交或 stash 后再执行 `/worktree merge` 即可。
- **在主目录用 `glob` / `grep` 会搜到 `.agentica/worktrees/` 里的重复文件吗？**
  不会。从主目录搜索时会自动跳过所有子 worktree 目录；只有把 `path` 明确指向某个 worktree 时才会搜那棵树。
