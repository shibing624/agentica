# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Cross-Provider Fallback Models Demo.

Real-world problem:
    Some user prompts hit a provider's content moderation layer
    ("我无法给到相关内容" / "I cannot help with that"). Or the provider
    has a transient outage (5xx / timeout / 429). Either way, the user's
    request fails.

Solution in agentica:
    Configure a cross-provider fallback chain on the Agent. When the
    primary model fails (content_filter / exhausted-retry timeout / 5xx),
    the next model in the chain is tried automatically. The user gets a
    real answer instead of a refusal.

Three guarantees, all demoed below:
  1. PER-CALL switch: each LLM call starts from the primary; fallback
     only rescues this one call. The next call retries the primary.
     `agent.model` is never mutated.
  2. AUDIT trail: every successful fallback rescue emits a structured
     `fallback.recovered` event via `agent._event_callback` AND a
     WARNING log line `[fallback.recovered] primary=X -> used=Y`.
  3. TRUTHFUL `RunResponse.model`: the field reflects the model that
     actually answered, not the primary.

Run:
    export OPENAI_API_KEY=...
    export DEEPSEEK_API_KEY=...
    python examples/agent_patterns/14_fallback_models.py
"""

import os

from agentica import Agent, OpenAIChat, RunConfig, DeepSeekChat


def _build_agent(primary: OpenAIChat, name: str) -> Agent:
    """Same agent, fallback configured ONCE at construction time."""
    return Agent(
        name=name,
        model=primary,
        fallback_models=[DeepSeekChat(id="deepseek-v4-flash")],
        instructions="You are a concise assistant. Answer in ONE sentence.",
    )


def _attach_audit(agent: Agent) -> list:
    """Capture fallback.recovered events for assertion + display."""
    events: list = []

    def _cb(evt: dict) -> None:
        if evt.get("type") == "fallback.recovered":
            events.append(evt)
            print(f"  >>> [audit] {evt}")

    agent._event_callback = _cb
    return events


def demo_happy_path() -> None:
    """Primary answers normally, fallback chain present but unused."""
    print("=" * 60)
    print("Demo 1: happy path -- primary answers, fallback unused")
    print("=" * 60)

    agent = _build_agent(OpenAIChat(id="gpt-4o"), name="happy")
    audit = _attach_audit(agent)

    response = agent.run_sync("用一句话解释什么是 RAG。")

    print(f"  answer model : {response.model}")
    print(f"  answer       : {response.content}")
    print(f"  audit events : {len(audit)} (expected 0)\n")


def demo_provider_outage() -> None:
    """Primary unreachable -> retry exhausted -> fallback rescues."""
    print("=" * 60)
    print("Demo 2: primary outage -> auto-fallback to deepseek")
    print("=" * 60)

    bad_primary = OpenAIChat(
        id="gpt-4o",
        base_url="https://unreachable-host-for-fallback-demo.invalid",
    )
    agent = _build_agent(bad_primary, name="fallback")
    audit = _attach_audit(agent)

    response = agent.run_sync("用一句话解释什么是 ReAct 智能体。")

    print(f"  answer model : {response.model} (truthfully reports fallback)")
    print(f"  answer       : {response.content}")
    print(f"  audit events : {len(audit)} (expected 1)")
    print(f"  agent.model  : {agent.model.id} (primary intact, NOT mutated)\n")


def demo_per_call_recovery() -> None:
    """Same agent, two sequential calls.
    Call 1: primary down -> fallback rescues.
    Call 2: primary back up -> primary answers (NOT stuck on fallback).
    """
    print("=" * 60)
    print("Demo 3: per-call discipline -- primary recovers next call")
    print("=" * 60)

    bad_primary = OpenAIChat(
        id="gpt-4o",
        base_url="https://unreachable-host-for-fallback-demo.invalid",
    )
    agent = _build_agent(bad_primary, name="per-call")
    audit = _attach_audit(agent)

    r1 = agent.run_sync("用一句话解释什么是 LLM。")
    print(f"  call 1 model : {r1.model} (fallback rescued)")

    # Provider recovered. Swap primary back. Same agent instance.
    agent.model = OpenAIChat(id="gpt-4o")
    r2 = agent.run_sync("用一句话解释什么是工具调用。")
    print(f"  call 2 model : {r2.model} (back to primary)")
    print(f"  total audit  : {len(audit)} (expected 1, only call 1 fired)\n")


def demo_run_config_override() -> None:
    """RunConfig.fallback_models takes precedence over Agent default
    for one specific run -- useful for query-tier-aware fallbacks
    (e.g. paid users get a premium fallback, free users get a cheap one).
    """
    print("=" * 60)
    print("Demo 4: RunConfig overrides Agent default for one run")
    print("=" * 60)

    agent = _build_agent(OpenAIChat(id="gpt-4o"), name="override")
    print(f"  agent default fallback : {[m.id for m in agent.fallback_models]}")

    # For this one call, force a different (also benign) chain.
    custom_chain = [DeepSeekChat(id="deepseek-v4-pro")]
    response = agent.run_sync(
        "用一句话解释什么是 LangChain。",
        config=RunConfig(fallback_models=custom_chain),
    )

    # agent.fallback_models is unchanged; the override only affected this run.
    print(f"  agent default after run: {[m.id for m in agent.fallback_models]} (unchanged)")
    print(f"  answer model           : {response.model}\n")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("DEEPSEEK_API_KEY"):
        print("Set OPENAI_API_KEY and DEEPSEEK_API_KEY first.")
        return

    demo_happy_path()
    demo_provider_outage()
    demo_per_call_recovery()
    demo_run_config_override()

    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
