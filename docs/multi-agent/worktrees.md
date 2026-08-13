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

## 二、写文件时的一次性提醒

真正要动手的那一刻，如果**另一个 live 会话**（同一个仓库，可以在别的 worktree）也把这个文件
改脏了，写入结果里会追加一行：

```text
Another live session has agentica/peers.py uncommitted: agentica-52 (main,
/Users/xuming/Documents/Codes/agentica). Your write went through — decide whether
to coordinate (send_message) or to work in your own checkout (worktree(...)).
```

三个刻意的取舍：

- **只提醒、不拦截**。两个会话同时改一个文件有时正是对的（一个写实现、一个写测试），没有启发式
  能替你判断；拦住写入只会把 agent 逼进死角。
- **只比同一个仓库**。peer 会发布 `repo_root`（共享 `.git` 的主 checkout），所以是精确比较，
  不会满世界的 `README.md` 互相报警；而同一个仓库的两个 **worktree** 仍然会命中——这才是值得报警的情况。
- **同一个文件同一个 peer 只说一次**。每次编辑都提醒等于训练 agent 忽略它。

## 三、worktree：一个任务一个目录、一个分支

```bash
# 启动时就进入自己的 worktree（不存在则创建，存在则复用）
agentica --worktree gateway-peers
```

已经跑了几周的会话不必重启——让它自己切（这也是被 `send_message` 遥控时唯一可行的方式）：

```text
你（或另一个会话）："先切到 gateway-peers 那个 worktree，再改代码"
agent 调用：worktree(action="use", name="gateway-peers")
```

`worktree` 工具的三个动作：

| 动作 | 作用 |
|------|------|
| `worktree(action="status")` | 列出本仓库所有 worktree，标出"你在这"，附带当前 git 位置 |
| `worktree(action="use", name="<任务>")` | 进入该任务的 worktree，首次自动创建，之后永远复用 |
| `worktree(action="merge")` | 把本 worktree 的分支并回基准分支，**worktree 保留** |

### 就地切换到底动了什么

"会话的目录"是四件事，必须一起动，否则隔离就是假的（比如只改了 prompt 里那句话，工具却还在
往原来的 checkout 写）：

| 动的东西 | 为什么 |
|---|---|
| 进程 cwd | git、`@file` 补全、shell 命令都读它 |
| agent 的执行环境 | prompt 里的工作目录、sandbox 可写目录、每个文件/shell 工具各自捕获的 work_dir（`Agent.rebind_work_dir`） |
| live peer 记录 | 别人靠它判断谁在哪；**但可寻址的名字不变**——别的会话和手机上 pin 的就是那个名字 |
| 状态栏 | 人得看得见自己在往哪个 worktree 里打字 |

**不动的**：transcript。会话日志继续写在它原本的位置（这正是 `session_base_dir` 的用途），
一段对话不会因为工作目录搬家而被切成两个文件。代价是之后 `/resume` 会问你回哪个目录——那个
提示本来就有，而且会记住你的选择。

### 目录位置与命名

默认：主 checkout 的**兄弟目录** `../<repo>-<任务>`，分支 `wt/<任务>`。

```text
~/Documents/Codes/agentica              主 checkout（main）
~/Documents/Codes/agentica-gateway      wt/gateway
~/Documents/Codes/agentica-paper        wt/paper
```

两种情况需要换地方，都用 `~/.agentica/config.yaml` 里的设置（`settings:` 块）：

```yaml
settings:
  worktree:
    root: ~/worktrees          # 绝对路径 → <root>/<repo>/<任务>
    # root: .agentica/worktrees  # 相对路径 → 仓库内 <repo>/.agentica/worktrees/<任务>
    link: [".env", ".envrc"]   # 新 worktree 里 symlink 过来的 gitignored 文件
```

- **父目录塞了二十个仓库**，不想再散一堆 `xxx-yyy` → 指到一个集中目录
- **共享挂载的父目录不可写** → 同上（不可写时报错会直接点名这个设置）

### 仓库内布局（`root: .agentica/worktrees`）

这是 Claude Code 的 `.claude/worktrees/` 形态，**一等支持**，选它不需要任何额外准备：

- 目录形如 `<repo>/.agentica/worktrees/<任务>`（不再插一层仓库名——仓库已经由位置隐含了）
- 首次创建时会在 `.agentica/worktrees/.gitignore` 里写一个 `*`，**自我忽略**：`git status`
  干净，且**不动仓库里那个被跟踪的 `.gitignore`**（那是共享文件，工具不该替你改）
- agentica 自己的 `glob` / `grep` 会跳过**仓库内的任何 worktree**（不只是 `.agentica`）：
  否则 `glob("**/*.py")` 把每个文件返回 N+1 份，而真正危险的不是噪音，是**改到副本那一份**上去。
  这个排除是**问 git 要的**（`git worktree list`，按仓库缓存 10s），不是按名字猜的——
  嵌套 worktree 不一定是 agentica 建的：人或另一个 agent 手打
  `git worktree add .worktrees/x` 会造成一模一样的重复。实测本仓库当时就有
  `.worktrees/wechat-media`（另一个会话的临时 worktree），`glob("**/peers.py")` 确实返回了
  它那一份。绑定在该 worktree 里工作的会话仍然看得见自己的全部文件（排除永不包含搜索根自身）

它的优点也是实打实的：不污染父目录、只要仓库可写就能用（共享挂载友好）、删掉仓库时
worktree 跟着一起走、和 `.cursor/` `.claude/` 同一套心智。

**唯一不可逆的代价**（也是它没被设为默认的原因）：嵌套 checkout 在主 checkout 里
`git clean` 的射程内。实测——

| 命令 | 结果 |
|------|------|
| `git clean -xdf`（单 `-f`，最常打的那条） | `Skipping repository .agentica/worktrees/docs` → **安全** |
| `git clean -xdff`（双 `-f`） | `Removing .agentica/` → 树没了，**连别的会话未提交的改动一起**，注册项变 `prunable` |

sibling 布局下这条命令完全无害。所以两种布局的错误代价不对称：sibling 选错是"报错 + 加一行
配置"（可恢复、可见），仓库内选错是"某次双 `-f` 清理顺手删掉三个会话的活"（不可恢复）。
默认因此留在 sibling；哪条更贴你的机器，改一行配置就切。

`.env` 是 **symlink 而不是拷贝**：轮换一次密钥所有 worktree 同时生效，而且机器上只存在一份。

### 合并回去，但不删 worktree

`worktree(action="merge")` 的顺序是刻意的，不是随手写的：

1. **先在 worktree 里把基准分支合进当前分支**。冲突就留在写这段代码的会话手上、在它自己的目录里、
   测试一条命令就能跑——而不是把一个半合并的 index 扔在所有会话共用的主 checkout 里。
2. **再在主 checkout 里把分支并进基准分支**，此时必然是 fast-forward，共享目录被碰的时间最短。
   两个会话同时 merge 时，git 自己的 `index.lock` 就是互斥锁，这里只是**等它**（重试 5 次），
   不另造一把锁。

副产品正是长期可用的关键：合完之后这个 worktree **与基准分支齐平**，下次接着用不会背着旧历史。

会被明确拒绝（而不是替你猜）的情况：worktree 里有未提交改动、主 checkout 不干净、主 checkout
不在基准分支上、当前分支没有新提交、在主 checkout 里执行 merge。

**任何情况下都不会删除 worktree**，也没有自动清理——它值钱的地方就是那个已经暖好的 IDE 索引、
装好的虚拟环境和一屏 shell 历史。要删就自己 `git worktree remove`。

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
4. **按任务命名，不要按会话命名**。一会话一 worktree 且不删除 = 目录爆炸；
   让会话复用同名 worktree 才是可持续的，也才能两个会话在同一个任务上接力。
5. `~/.agentica` 是共享的：session 列表和 profile 覆盖按 work_dir 分桶（正是我们要的），
   但 cron、skills、MEMORY.md 是全局共享——别指望它们被隔离。
