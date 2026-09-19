# 多个会话共用一个仓库：worktree + 现场可见

几个 agentica 会话（终端里的 CLI、手机 IM 遥控的 CLI、Gateway 自己的 agent）同时改**同一个
仓库**时，真正花钱的不是合并冲突，而是两件事：

1. **互相覆盖**：两个会话在同一个工作目录里编辑同一批文件，还会抢 git 的 `index.lock`。
2. **互相打听**：为了搞清"你是不是也在改这个文件 / 你落后 main 几个 commit"，得发消息、
   等对方回、而答案在第三个会话提交的那一刻就过期了。

agentica 对这两件事的回答是分开的：**worktree** 解决第 1 个，**presence 带 git 状态**解决第 2 个。

## 一、不用问就知道对方在干什么

每个 CLI 会话在自己的心跳里发布 git 位置（`agentica/git_state.py` → `agentica/peers.py`），
所以 `list_agents`（以及 `/list-agents`）直接就能看到：

```text
agentica-52 [peer=52d79d86]
  status: running a turn
  cwd: /Users/xuming/Documents/Codes/agentica
  git: main @ 8ca321e · 3 dirty
  dirty: CHANGELOG.md, agentica/gateway/main.py, docs/advanced/gateway.md
  working on: 把 wechat 的 -14 重登逻辑补上
```

- `git:` 一行 = 分支 · head · 相对基准分支的 `+ahead/-behind` · 脏文件数
- `dirty:` 一行 = 具体路径（最多 12 个，超出显示 `+N more`）
- 采集有 10s 缓存，心跳只在内容变化或 30s 到点时才落盘——不会因为这个功能变成每秒三次 `git` 调用

**基准分支是本地 `main`（没有则 `master`），不是 upstream**：另一个会话提交到本地 main 还没
push 时，`origin/main` 看不见它——而那个会话恰恰是你马上要撞上的人。

> 没发布过这些字段的老会话记录只会显示 `git: <branch>`，**不会**显示 "clean"。"clean" 是别人
> 据以决定能不能 rebase 的信息，缺字段不许冒充它。

## 二、对方改了哪些文件：看 presence，不写进工具结果

真正要动手之前，用 `list_agents`（或 `/list-agents`）看对方心跳里的 `dirty:`。那是现场可见的 git 状态，写文件工具**不会**再往结果里追加「另一个会话也改了这个文件」——提醒曾经挂在 `write_file` / `apply_patch` 上，模型会把它当成写入合同的一部分。

需要自己在写入时查一次的调用方，仍可用 `agentica.peer_conflicts.PeerConflictChecker`；产品默认路径是看 presence，不是改工具结果。

三个刻意的取舍（对 presence / checker 都成立）：

- **只提醒、不拦截**。两个会话同时改一个文件有时正是对的（一个写实现、一个写测试）。
- **只比同一个仓库**。peer 会发布 `repo_root`，所以是精确比较；同一个仓库的两个 **worktree** 仍然会命中。
- **同一个文件同一个 peer，checker 只说一次**。每次编辑都提醒等于训练 agent 忽略它。

## 三、worktree：一个任务一个目录、一个分支

会话身份留在主 checkout。worktree 是一棵给工具用的树，不是把这场对话搬进去。

```bash
# 可选：新进程从一开始就站在树上（这场对话的身份在树上）
agentica --worktree gateway-peers
```

已经在跑的会话不必重启。模型读内置 `worktree` skill（人也可用 `/worktree`）：

```text
agent 调用：worktree(action="new", name="gateway-peers")
  → /repo/.agentica/worktrees/gateway-peers
然后：read_file / write_file / apply_patch / execute(..., work_dir=<该路径>)
      glob / grep(..., path=<该路径>)
```

漏传 `work_dir` / `path`，调用落在当前会话目录（通常是 main）。这是可观察的错，比把会话 cwd 绑到树上、树一删全挂要好修。

`worktree` 工具的动作：

| 动作 | 作用 |
|------|------|
| `worktree(action="status")` | 列出本仓库所有 worktree，附带当前会话所在目录 |
| `worktree(action="new", name="<任务>")` | 创建或复用健康 checkout，**返回路径**；会话不搬家 |
| `worktree(action="merge", name="<任务>")` | 把该分支并回本地基准分支，然后删除 checkout |
| `worktree(action="remove", name="<任务>")` | 丢掉一个 agentica 的 `wt/*` checkout。脏树由 git 拒绝；未合入的提交留在分支上 |

`new(name="main")` 会被拒绝——`main` 不是任务名。没有 `main` / `use` 搬家动作。登记项不再是 checkout（目录没了、或占坑的裸目录）时，`new` 拒绝而不是销毁占坑内容：先 `remove` 那个名字，或换一个名字。

退出时：只动以 `--worktree` 启动的进程脚下那棵 `wt/*` 树。没有独有工作 → 自动删；有未提交或未合并的提交 → 留在盘上。手动 `git worktree add`、detached、Claude Code 的树一律不碰。

### 目录位置与命名

默认：仓库内 `<repo>/.agentica/worktrees/<任务>`，分支 `wt/<任务>`（Claude Code 的 `.claude/worktrees/` 形态）。

```text
~/Documents/Codes/agentica                         主 checkout（main）
~/Documents/Codes/agentica/.agentica/worktrees/gateway   wt/gateway
~/Documents/Codes/agentica/.agentica/worktrees/paper     wt/paper
```

首次创建时会在 `.agentica/worktrees/.gitignore` 里写一个 `*`，**自我忽略**：`git status` 干净，且**不动仓库里那个被跟踪的 `.gitignore`**。从主 checkout `glob` / `grep` 会跳过仓库内的任何 worktree（问 `git worktree list`，不是按名字猜）。把 `path` 指到某棵树上，才能搜到那棵树自己的文件。

两种情况需要换地方，都用 `~/.agentica/config.yaml` 里的设置（`settings:` 块；嵌套 `worktree.root` 和扁平行 `worktree.root` 都行）：

```yaml
settings:
  worktree:
    root: sibling              # 旧默认：../<repo>-<任务>
    # root: ~/worktrees        # 绝对路径 → <root>/<repo>/<任务>
    link: [".env", ".envrc"]   # 新 worktree 里 symlink 过来的 gitignored 文件
```

- **不想 worktree 落在仓库里**（比如经常 `git clean -xdff`）→ `root: sibling`
- **父目录塞了二十个仓库，想集中放** → 绝对路径
- **共享挂载的父目录不可写** → 默认的仓库内布局正好不需要写父目录

`git clean -xdf`（单 `-f`）对嵌套 checkout 是 `Skipping repository`，安全；`git clean -xdff`（双 `-f`）会 `Removing .agentica/`。不想被主仓库的 clean 扫到 → `root: sibling`。

`.env` 是 **symlink 而不是拷贝**：轮换一次密钥所有 worktree 同时生效，而且机器上只存在一份。

### 合并回去，然后删除 worktree

`worktree(action="merge", name="<任务>")` 的顺序是刻意的：

1. **先在那棵 worktree 里把基准分支合进当前分支**。冲突留在写代码的目录里，而不是扔在人人共用的主 checkout。
2. **再在主 checkout 里把分支并进基准分支**，此时必然是 fast-forward。两个会话同时 merge 时，git 自己的 `index.lock` 就是互斥锁，这里只是**等它**（重试 5 次）。
3. **删除 worktree。** 会话一直在它原来的目录，不搬家。合完后与本地 base 齐平，删除不会丢掉独有提交。

会被明确拒绝（而不是替你猜）的情况：worktree 里有未提交改动、主 checkout 不干净、主 checkout 不在基准分支上。分支已经全部落在 base 上不是错误——那是任务做完的样子，随后照常清理 checkout。

`remove` 只拦所有权（agentica 的 `wt/*`，不是主 checkout，不是 detached / Claude Code）。脏树交给 git 拒绝，git 的话原样转达。未合入 base 的提交不是拒绝条件：checkout 删掉，`git branch -d` 会留下分支。以 `--worktree` 启动的进程退出时的自动清理才看「有没有独有工作」——那是猜测，不是一条显式 remove。

## 四、几个实测出来的坑

1. **`agentica` 命令永远跑主目录的代码**。console script 是 editable 安装，`sys.path[0]` 是
   bin 目录，所以在 worktree 里敲 `agentica` 加载的仍是主 checkout 的 agentica。而
   `python -m pytest` / `python x.py` 在 worktree 里 import 的是**该 worktree 的**代码。
   自举（用 agentica 改 agentica）时：测试用 `python -m pytest`，要跑 worktree 版 CLI 用
   `python -m agentica.cli.main`。
2. **gitignored 文件不会进 worktree**：`.env` 缺失的症状是"会话起来了但连不上任何模型"，
   和真正的原因八竿子打不着。默认 symlink `.env`，其它用 `worktree.link` 加。
3. **`index.lock` 冲突基本上是"多进程同一个工作树"的症状**，切了 worktree 就没了；
   但仓库级操作（`worktree add`、`fetch`、`gc`）仍会短暂锁 `.git`。
4. **按任务命名，不要按会话命名。** 同名只在任务未完成时复用，合完即拆；两个会话仍能在同一个未完成的任务上接力，目录不会因此堆下去。
5. `~/.agentica` 是共享的：session 列表和 profile 覆盖按 work_dir 分桶（正是我们要的），
   但 cron、skills、MEMORY.md 是全局共享——别指望它们被隔离。
