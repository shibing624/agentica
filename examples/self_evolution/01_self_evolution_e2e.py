# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: End-to-end demo of the self-evolution pipeline (arch_v5 §8).

Closed-loop self-evolution demo using DeepSeek-V3.2 via Volc Ark
(stronger reasoning, slower than GLM — use for "do the model fairy-dust
matter?" comparisons after you've validated the loop on GLM).

The case: **search before read**. When the user asks to read a file by an
imprecise name, naive agents call read_file with a guessed path and fail
with FileNotFoundError. The user corrects them with a procedural rule
("list the directory first, find the actual filename, then read"). After
3-4 corrections we expect a SKILL.md to be spawned, and a brand-new agent
in Session 2 must use that skill to **actually solve** a similar task.

Why this case (vs the old "ls before read" placeholder):

1. **Real tool failure**: read_file actually returns FileNotFoundError —
   we get tool_error events, not just corrections. This means we can also
   demonstrate ``tool_recovery`` events firing in Session 2 (read_file
   succeeds for a tool that previously failed).
2. **Real procedural rule**: agents that don't search first cannot solve
   Session 2's task. Skill use is observable through tool_calls
   (list_directory before read_file) AND through outcome (the file's
   real content is returned).
3. **Verifiable**: Session 2 asserts both (a) ``get_skill_info`` was
   called for the spawned skill and (b) the requested file's content
   was actually retrieved.

Session 1 — self-evolution (skill spawn):
  Hooks chain: ConversationArchive + MemoryExtract (evidence gate) +
  ExperienceCapture (cards + skill_upgrade=shadow). Across 4 rounds we
  ask the agent to read 4 files using guessed paths that don't exist.
  After each failure the user issues the same procedural correction.
  Correction repeat_count climbs to 4 → maybe_spawn_skill is invoked
  → a SKILL.md and meta.json land under generated_skills/<slug>/.

Session 2 — cross-session application:
  Brand-new Agent + brand-new SkillTool, same workspace. SkillTool is
  given the generated_skills/ directory via custom_skill_dirs. The agent
  is asked to read one of the pre-created files using only a keyword,
  and we assert it (a) loads the spawned skill via get_skill_info and
  (b) returns the file's actual content.

Required env: ``ARK_API_KEY`` and ``ARK_MODEL_NAME`` (DeepSeek-V3.2
endpoint id) in ``.env``.

Run:
    python examples/self_evolution/01_self_evolution_e2e.py

Debug spawn decisions:
    AGENTICA_LOG_LEVEL=debug python examples/self_evolution/01_self_evolution_e2e.py 2>&1 \\
        | grep -iE "skill spawn|action=|ignore|recovery"
"""
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, Workspace, ArkChat
from agentica.agent.config import ExperienceConfig, SkillUpgradeConfig
from agentica.experience.compiler import _RULE_STOPWORDS, _rule_to_title, _stem
from agentica.experience.skill_upgrade import SkillEvolutionManager
from agentica.hooks import (
    ConversationArchiveHooks,
    ExperienceCaptureHooks,
    MemoryExtractHooks,
    _CompositeRunHooks,
)
from agentica.tools.buildin_tools import get_builtin_tools
from agentica.tools.skill_tool import SkillTool


WS_PATH = Path("./tmp/self_evo_ds").resolve()
USER_ID = "demo_user"

# Pre-created files with unguessable filenames. Session 2 must search this
# directory to find them — no LLM should be able to guess the exact stem.
DATA_DIR = WS_PATH / "test_data"
DATA_FILES = {
    "agent_design_notes_v3.md": (
        "# Agent Design Notes (v3)\n\n"
        "Core idea: tool-loop with structured event capture.\n"
        "Hot-key invariant: every correction lifts repeat_count.\n"
    ),
    "architecture_overview_2026q2.md": (
        "# Architecture Overview Q2 2026\n\n"
        "Layered: model → runner → hooks → workspace.\n"
        "Hooks fire deterministic + LLM gates.\n"
    ),
    "experimental_results_run17.md": (
        "# Experiment Results — run 17\n\n"
        "Self-evo baseline: 41% recovery; +skill: 78%.\n"
    ),
    "tooling_changelog_apr2026.md": (
        "# Tooling Changelog (April 2026)\n\n"
        "Added: tool_recovery event, INDEX.md L1 router.\n"
        "Removed: textbook-format SKILL.md prompt.\n"
    ),
}

# Same procedural rule, embedded verbatim so _rule_to_title converges to
# the same correction title across all 4 rounds (drives repeat_count → 4).
PROCEDURAL_RULE = "list directory before read file"
CORRECTION_TEMPLATE = (
    "{prefix} The rule (apply verbatim every time): "
    f"'{PROCEDURAL_RULE}'. "
    "Step 1: call list_directory on the directory you think the file is in. "
    "Step 2: pick the actual filename from the listing (don't guess). "
    "Step 3: only then call read_file with that exact path. "
    "Do not skip these steps."
)


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def dump_tree(path: Path, prefix: str = "", max_depth: int = 4) -> None:
    if max_depth < 0 or not path.exists():
        return
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    for i, entry in enumerate(entries):
        last = i == len(entries) - 1
        connector = "└── " if last else "├── "
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            ext = "    " if last else "│   "
            dump_tree(entry, prefix + ext, max_depth - 1)


# ─── Test data setup ─────────────────────────────────────────────────────
def seed_test_data() -> None:
    """Create the unguessable-named files Session 2 must locate via search."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, body in DATA_FILES.items():
        (DATA_DIR / name).write_text(body, encoding="utf-8")


# ─── Session 1: self-evolution agent ─────────────────────────────────────
def _make_model():
    """DeepSeek-V3.2 via Volc Ark endpoint."""
    from dotenv import load_dotenv
    load_dotenv()
    model_name = os.environ.get("ARK_MODEL_NAME")
    return ArkChat(id=model_name)


def build_evo_agent(workspace: Workspace) -> Agent:
    """Agent with full hooks chain to drive the self-evolution pipeline."""
    model = _make_model()

    hooks = _CompositeRunHooks([
        ConversationArchiveHooks(),
        MemoryExtractHooks(),
        ExperienceCaptureHooks(
            ExperienceConfig(
                capture_tool_errors=True,
                capture_user_corrections=True,
                capture_success_patterns=True,
                feedback_confidence_threshold=0.55,
                promotion_count=2,
                promotion_window_days=1,
                skill_upgrade=SkillUpgradeConfig(
                    mode="shadow",
                    min_repeat_count=3,
                    min_tier="warm",
                    checkpoint_interval=2,
                    # Bootstrap demo: cold workspace, no prior recoveries.
                    # The new tool_recovery semantic does not deadlock here
                    # (recoveries are emitted by ExperienceCaptureHooks for
                    # any tool success after a prior tool_error), but on a
                    # truly empty workspace the FIRST few runs have nothing
                    # to recover from. Setting 0 lets the first skill spawn
                    # purely from correction repeat_count.
                    min_success_applications=0,
                ),
            )
        ),
    ])

    agent = Agent(
        model=model,
        auxiliary_model=model,
        name="FileExplorer",
        instructions=[
            "You are a filesystem helper.",
            "When the user asks to read a file, call the read_file tool.",
            "Report concisely whether the file exists and what it contains.",
            "When reporting file contents, reproduce them verbatim — do not summarize or paraphrase.",
        ],
        tools=get_builtin_tools(
            include_file_tools=True,
            include_execute=False,
            include_web_search=False,
            include_fetch_url=False,
            include_todos=False,
            include_task=False,
        ),
        workspace=workspace,
    )
    agent._default_run_hooks = hooks
    return agent


async def run_turn(agent: Agent, turn_idx: int, label: str, user_msg: str) -> None:
    print_section(f"Turn {turn_idx} [{label}]: {user_msg[:80]!r}")
    response = await agent.run(user_msg)
    anchor = agent.run_context.task_anchor
    print(f"[run_id]       {response.run_id}")
    print(f"[anchor.goal]  {anchor.goal[:80]!r}")
    print(f"[status]       {agent.run_context.status.value}")
    print(f"[reply]        {(response.content or '').strip()[:200]}")


# ─── Session 1 fallback: print spawn-candidate diagnosis ─────────────────
async def diagnose_spawn_failure(workspace: Workspace) -> Path:
    """Print why the hook chain didn't spawn — without burning more tokens.

    If the hook chain decided ``action=ignore`` and we re-invoke the LLM
    with the same gate (``min_success_applications=0``) and the same
    candidates, the LLM almost always returns the same verdict. Diagnostic
    value comes from showing the candidates and existing skills, not from
    hammering the LLM a second time. To override the LLM verdict, lower
    ``min_repeat_count`` and re-run with ``AGENTICA_LOG_LEVEL=debug``.
    """
    print_section("Session 1 diagnostic: candidates + existing skills (no LLM retry)")
    manager = SkillEvolutionManager()
    exp_dir = workspace._get_user_experience_dir()
    gen_dir = workspace._get_user_generated_skills_dir()

    candidates = manager.get_candidate_cards(
        exp_dir=exp_dir, min_repeat_count=2, min_tier="warm",
    )
    print(f"candidates collected: {len(candidates)}")
    for c in candidates:
        print(f"  - {c['title']:50s}  type={c['type']:14s}"
              f"  repeat={c['repeat_count']}  tier={c['tier']}")

    existing = sorted(d.name for d in gen_dir.iterdir() if d.is_dir()) if gen_dir.exists() else []
    print(f"existing generated skills: {existing}")

    if not candidates:
        print("[!] no candidates → corrections were not consolidated."
              " Check whether _rule_to_title is producing stable titles.")
    else:
        print("[!] hook chain returned action=ignore for these candidates."
              " Re-run with AGENTICA_LOG_LEVEL=debug to capture the LLM"
              " reason; do NOT just retry — same input → same verdict.")
    return gen_dir


# ─── Session 2: cross-session consumer agent ─────────────────────────────
def build_consumer_agent(workspace: Workspace, generated_skills_dir: Path):
    """Fresh agent + fresh SkillTool that loads Session 1's generated skill.

    Also attaches ExperienceCaptureHooks so tool_recovery events emitted
    by the consumer's successful read_file (which previously failed in
    Session 1) are persisted into events.jsonl — the closed-loop demo
    asserts on those events.
    """
    model = _make_model()

    skill_tool = SkillTool(
        custom_skill_dirs=[str(generated_skills_dir)],
        auto_load=False,
    )
    skill_tool.initialize()

    tools = get_builtin_tools(
        include_file_tools=True,
        include_execute=False,
        include_web_search=False,
        include_fetch_url=False,
        include_todos=False,
        include_task=False,
    ) + [skill_tool]

    agent = Agent(
        model=model,
        name="FileExplorerV2",
        instructions=[
            "You are a filesystem helper.",
            "If a skill matches the task, call get_skill_info(skill_name) "
            "to load its full instructions BEFORE acting.",
            "When reporting file contents, reproduce them verbatim — do not summarize or paraphrase.",
        ],
        tools=tools,
        workspace=workspace,
    )
    # Attach an ExperienceCaptureHooks with skill_upgrade off so we observe
    # ``tool_recovery`` events fire (read_file succeeded — and read_file
    # failed in Session 1 — therefore one tool_recovery is expected per
    # run). No auxiliary_model needed: skill_upgrade=off and
    # capture_user_corrections=False mean no LLM gates run here.
    agent._default_run_hooks = _CompositeRunHooks([
        ExperienceCaptureHooks(
            ExperienceConfig(
                capture_tool_errors=True,
                capture_user_corrections=False,
                capture_success_patterns=True,
                skill_upgrade=SkillUpgradeConfig(mode="off"),
            )
        ),
    ])
    return agent, skill_tool


# ─── Artifact inspection ─────────────────────────────────────────────────
def inspect_artifacts(workspace: Workspace) -> Path:
    user_path = workspace._get_user_path()

    print_section("Workspace tree")
    print(str(user_path))
    dump_tree(user_path, max_depth=4)

    print_section("1) Evidence gate — memory_candidates/")
    cand_dir = user_path / workspace.config.memory_candidates_dir
    if cand_dir.exists():
        candidates = list(cand_dir.glob("*.md"))
        print(f"found {len(candidates)} quarantined memory file(s)")
        for f in candidates[:3]:
            print(f"\n── {f.name} ──")
            print(f.read_text(encoding="utf-8")[:300])
    else:
        print("no candidates created")

    canon = user_path / workspace.config.memory_dir
    canon_count = len(list(canon.glob("*.md"))) if canon.exists() else 0
    print(f"\ntrusted memory entries in canonical memory/: {canon_count} (expected 0)")

    print_section("2) Experience cards — experiences/")
    exp_dir = user_path / "experiences"
    if exp_dir.exists():
        cards = sorted(p for p in exp_dir.glob("*.md") if p.name != "EXPERIENCE.md")
        print(f"found {len(cards)} experience card(s)")
        for card in cards:
            print(f"\n── {card.name} ──")
            print(card.read_text(encoding="utf-8")[:600])

        events_path = exp_dir / "events.jsonl"
        if events_path.exists():
            lines = events_path.read_text(encoding="utf-8").strip().splitlines()
            event_types = [json.loads(line).get("event_type", "?") for line in lines if line]
            type_counts: dict = {}
            for t in event_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"\n── events.jsonl ({len(lines)} events) ──")
            for t, n in sorted(type_counts.items()):
                print(f"  {t:20s} {n}")
    else:
        print("no experience cards written")

    print_section("3) Learning reports — reports/learning/")
    reports_dir = user_path / "reports" / "learning"
    if reports_dir.exists():
        md_files = sorted(reports_dir.glob("*.md"))
        jsonl = reports_dir / "learning.jsonl"
        print(f"found {len(md_files)} per-run markdown report(s)")
        if jsonl.exists():
            lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
            print(f"\n── learning.jsonl summary ({len(lines)} lines) ──")
            for line in lines:
                obj = json.loads(line)
                print(
                    f"  run={obj['run_id'][:8]}  status={obj['status']:<8}"
                    f"  cards={obj['cards_written']}  errors={obj['tool_errors_captured']}"
                    f"  corrections={obj['corrections_persisted']}"
                    f"  skill={obj.get('skill_state_change') or '-'}"
                )
    else:
        print("no learning reports written")

    print_section("4) Generated skills — generated_skills/")
    gen_dir = user_path / "generated_skills"
    if gen_dir.exists():
        skills = [p for p in gen_dir.iterdir() if p.is_dir()]
        print(f"found {len(skills)} generated skill(s)")
        for skill_dir in skills:
            meta_path = skill_dir / "meta.json"
            skill_md = skill_dir / "SKILL.md"
            if meta_path.exists():
                print(f"\n── {skill_dir.name}/meta.json ──")
                print(meta_path.read_text(encoding="utf-8"))
            if skill_md.exists():
                print(f"\n── {skill_dir.name}/SKILL.md ──")
                print(skill_md.read_text(encoding="utf-8"))

        index_md = gen_dir / "INDEX.md"
        if index_md.exists():
            print(f"\n── INDEX.md (L1 keyword router) ──")
            print(index_md.read_text(encoding="utf-8"))
    else:
        print("no skills generated")

    return gen_dir


# ─── Main ─────────────────────────────────────────────────────────────────
async def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("ARK_API_KEY"):
        print("Set ARK_API_KEY in .env first.")
        sys.exit(1)
    model_name = os.environ.get("ARK_MODEL_NAME") or "deepseek-v3-2"
    print(f"[model] provider=ark(Volcengine)  id={model_name}")

    if WS_PATH.exists():
        shutil.rmtree(WS_PATH)
    seed_test_data()

    workspace = Workspace(path=WS_PATH, user_id=USER_ID)
    workspace.initialize()
    print(f"workspace: {WS_PATH}")
    print(f"test data: {DATA_DIR}  ({len(DATA_FILES)} files)")

    # ═══ SESSION 1: self-evolution ═══════════════════════════════════════
    print_section("SESSION 1 — self-evolution: collect experience + spawn skill")
    evo_agent = build_evo_agent(workspace)

    # 4 rounds. Each round: ask agent to read a file using a guessed (wrong)
    # path → tool_error → user correction with the procedural rule.
    fake_paths = [
        str(DATA_DIR / "agent.md"),
        str(DATA_DIR / "architecture.md"),
        str(DATA_DIR / "experiments.md"),
        str(DATA_DIR / "changelog.md"),
    ]
    correction_prefixes = [
        "Wrong, that file does not exist.",
        "Same mistake again.",
        "Third time — please listen.",
        "Final reminder before I escalate.",
    ]

    turn = 0
    for path, prefix in zip(fake_paths, correction_prefixes):
        turn += 1
        await run_turn(
            evo_agent, turn, "read",
            f"Please read {path} and tell me the content.",
        )
        turn += 1
        await run_turn(
            evo_agent, turn, "correction",
            CORRECTION_TEMPLATE.format(prefix=prefix),
        )

    gen_dir = inspect_artifacts(workspace)

    # ── _rule_to_title diagnosis (helps debug if corrections don't merge) ──
    print_section("_rule_to_title diagnosis: raw rule → derived title")
    exp_dir = workspace._get_user_experience_dir()
    if exp_dir.exists():
        for card_path in sorted(exp_dir.glob("*.md")):
            if card_path.name == "EXPERIENCE.md":
                continue
            text = card_path.read_text(encoding="utf-8")
            rule_match = re.search(r"^Rule:\s*(.+)$", text, re.MULTILINE)
            type_match = re.search(r"^type:\s*(\S+)", text, re.MULTILINE)
            if rule_match:
                raw_rule = rule_match.group(1).strip()
                derived = _rule_to_title(raw_rule)
                card_type = type_match.group(1) if type_match else "?"
                print(f"  [{card_type:10s}] rule={raw_rule!r}")
                print(f"               title={derived!r}")
                tokens = re.findall(r"[a-z0-9]+", raw_rule.lower())
                seen = set()
                stems = []
                for t in tokens:
                    if t in _RULE_STOPWORDS:
                        continue
                    s = _stem(t)
                    if s and s not in seen:
                        seen.add(s)
                        stems.append(s)
                print(f"               all stems({len(stems)}) → {'_'.join(stems)!r}")

    # If hook chain didn't spawn anything, print diagnostics (no LLM retry).
    spawned_now = [p for p in gen_dir.iterdir() if p.is_dir()] if gen_dir.exists() else []
    if not spawned_now:
        gen_dir = await diagnose_spawn_failure(workspace)

    # ═══ Session 1 verdict ════════════════════════════════════════════════
    print_section("SESSION 1 verdict")
    spawned_skills = [p for p in gen_dir.iterdir() if p.is_dir()] if gen_dir.exists() else []
    has_shadow_skill = False
    shadow_skill_name = None
    for skill_dir in spawned_skills:
        meta_path = skill_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("status") in ("shadow", "draft"):
                has_shadow_skill = True
                shadow_skill_name = meta.get("skill_name") or skill_dir.name
                print(f"[\u2713] skill spawned: {skill_dir.name}")
                print(f"    status={meta.get('status')}, source={meta.get('source_experience')}")
    if not has_shadow_skill:
        print("[\u2717] Session 1 did not spawn a shadow skill — Session 2 cannot validate.")
        print("    Diagnostics:")
        print("    - check experiences/ for correction card with repeat_count >= min_repeat_count")
        print("    - re-run with AGENTICA_LOG_LEVEL=debug to inspect LLM ignore reasons")
        sys.exit(1)

    # ═══ SESSION 2: cross-session application ═══════════════════════════
    print_section("SESSION 2 — fresh agent + same workspace, must use new skill")
    consumer_agent, skill_tool = build_consumer_agent(workspace, gen_dir)

    # ── 2a. Static discovery ──
    print(f"\n[SkillTool registry — list_skills]")
    print(skill_tool.list_skills())

    generated_loaded = [
        s for s in skill_tool.registry.list_all() if s.location == "generated"
    ]
    assert generated_loaded, "Session 2 loaded zero generated skills"
    print(f"\n[\u2713] Session 2 loaded {len(generated_loaded)} generated skill(s):")
    for s in generated_loaded:
        print(f"    - {s.name}: {s.description[:80]}")
    assert any(s.name == shadow_skill_name for s in generated_loaded), (
        f"Session 1's spawned skill {shadow_skill_name!r} "
        f"is missing from Session 2's registry"
    )

    # ── 2b. System prompt injection ──
    skill_prompt = skill_tool.get_system_prompt() or ""
    assert shadow_skill_name in skill_prompt, (
        f"skill {shadow_skill_name!r} not present in SkillTool.get_system_prompt() output"
    )
    print(f"\n[\u2713] skill {shadow_skill_name!r} injected into SkillTool system prompt")

    # ── 2c. Real task: ask for a file by keyword only ──
    # The agent cannot guess the exact filename ("agent_design_notes_v3.md").
    # Without skill use → fails. With skill → list_directory first → finds it.
    target_file = "agent_design_notes_v3.md"
    target_content_marker = "Hot-key invariant"
    print_section("Session 2: real task — agent must apply the new skill")
    response = await consumer_agent.run(
        f"I need the content of the agent design notes file under {DATA_DIR}. "
        "I don't remember the exact filename. Use any relevant skill if it helps."
    )
    print(f"[reply] {(response.content or '').strip()[:600]}")

    tool_calls = response.tool_calls or []
    tool_call_names = [tc.tool_name for tc in tool_calls]
    get_info_args = [
        tc.tool_args.get("skill_name") or tc.tool_args.get("name")
        for tc in tool_calls
        if tc.tool_name == "get_skill_info"
    ]
    # The agent may use any equivalent dir-listing tool: list_directory,
    # ls (shell), or grep with a glob. The "search before read" verdict
    # only counts a search if BOTH:
    #   (i) it inspected DATA_DIR (not some unrelated path), AND
    #   (ii) it ran AFTER get_skill_info (skill-driven, not coincidental
    #        pre-read scan).
    # Without these constraints any well-behaved file-search agent passes
    # the verdict by accident, regardless of whether the spawned skill
    # actually influenced its plan.
    SEARCH_TOOLS = {"list_directory", "ls", "grep", "glob"}
    data_dir_str = str(DATA_DIR)

    def _touches_data_dir(tc) -> bool:
        for value in (tc.tool_args or {}).values():
            if isinstance(value, str) and data_dir_str in value:
                return True
        return False

    search_calls = [tc for tc in tool_calls if tc.tool_name in SEARCH_TOOLS]
    read_file_calls = [tc for tc in tool_calls if tc.tool_name == "read_file"]
    skill_info_idx = next(
        (
            i for i, tc in enumerate(tool_calls)
            if tc.tool_name == "get_skill_info"
            and (tc.tool_args.get("skill_name") or tc.tool_args.get("name")) == shadow_skill_name
        ),
        None,
    )
    first_read_idx = next(
        (i for i, tc in enumerate(tool_calls) if tc.tool_name == "read_file"),
        None,
    )
    first_qualified_search_idx = next(
        (
            i for i, tc in enumerate(tool_calls)
            if tc.tool_name in SEARCH_TOOLS and _touches_data_dir(tc)
            and (skill_info_idx is None or i > skill_info_idx)
        ),
        None,
    )
    searched_first = (
        first_qualified_search_idx is not None
        and first_read_idx is not None
        and first_qualified_search_idx < first_read_idx
    )

    print(f"\n[tool calls observed] {tool_call_names}")
    print(f"[get_skill_info args]  {get_info_args}")
    print(f"[search-tool calls]    {len(search_calls)} ({[tc.tool_name for tc in search_calls]})")
    print(f"[read_file calls]      {len(read_file_calls)}")
    print(f"[skill-driven search]  idx={first_qualified_search_idx} "
          f"(skill_info_idx={skill_info_idx}, first_read_idx={first_read_idx})")

    # ── 2d. Closed-loop assertions ──
    skill_loaded = skill_info_idx is not None
    found_real_file = any(
        target_file in str(tc.tool_args.get("file_path", ""))
        for tc in read_file_calls
    )
    content_returned = target_content_marker in (response.content or "")

    print_section("CLOSED-LOOP VERDICT")
    print(f"  [ {'\u2713' if skill_loaded     else '\u2717'} ] (a) get_skill_info({shadow_skill_name!r}) called")
    print(f"  [ {'\u2713' if searched_first   else '\u2717'} ] (b) a search tool called BEFORE read_file")
    print(f"  [ {'\u2713' if found_real_file  else '\u2717'} ] (c) read_file called with the actual filename")
    print(f"  [ {'\u2713' if content_returned else '\u2717'} ] (d) reply contains the file's real content marker")

    # ── 2e. tool_recovery emission verification ──
    events_path = workspace._get_user_experience_dir() / "events.jsonl"
    recoveries: list = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event_type") == "tool_recovery":
                recoveries.append(ev)

    print(f"\n[tool_recovery events in events.jsonl] {len(recoveries)}")
    for r in recoveries[-3:]:
        print(f"    - tool={r.get('tool')}  elapsed={r.get('elapsed', 0):.3f}s")

    closed_loop_ok = skill_loaded and searched_first and found_real_file and content_returned
    print_section("DONE" if closed_loop_ok else "DONE (with gaps)")
    print(f"workspace: {WS_PATH}")
    if not closed_loop_ok:
        print(
            "Hint: stronger models can be chatty / over-explore even with a "
            "skill loaded. Inspect [tool calls observed] above to see whether "
            "the agent ignored the skill or just took a longer path."
        )


if __name__ == "__main__":
    asyncio.run(main())
