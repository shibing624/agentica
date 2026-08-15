# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Deep Research Agent Evaluation Script

This script evaluates the performance of agentica Agent with multi-round strategy
on various deep research benchmarks (BrowseComp, GAIA, SimpleQA, etc.)

Usage:
    python run.py --model gpt-4o --dataset browsecomp_zh_small --eval_n_limit 10
"""
import argparse
import os
import json
import asyncio
import re
from typing import List, Dict, Any, Optional
from agentica.utils.log import logger
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentica import Agent, OpenAIChat, ZhipuAIChat, Message, PromptConfig
from agentica import ArkChat
from agentica.tools.builtin import get_builtin_tools
from prompt import JUDGE_PROMPT_GAIA, JUDGE_PROMPT_BC, JUDGE_PROMPT_QA, SYSTEM_PROMPT_MULTI

pwd_path = os.path.abspath(os.path.dirname(__file__))


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def extract_answer(text: str) -> str:
    """
    Extract the answer from model response.
    
    The new multi-round implementation doesn't use <answer> tags,
    so we return the full response as the answer.
    """
    if not text:
        return ""
    
    # Try to extract from <answer> tags if present (for backward compatibility)
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    
    # Otherwise return the full response (new behavior)
    return text.strip()


def extract_correct_judgement(response: str) -> Optional[str]:
    """Extract yes/no judgement from LLM response."""
    match = re.search(r'correct\s*:\s*(yes|no)', response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


async def call_llm_judge(item: Dict, judge_prompt: str, dataset: str) -> Dict[str, Any]:
    """Judge if predicted answer matches ground-truth using LLM."""
    question = item.get("question", "")
    correct_answer = item.get("answer", "")
    prediction = item.get("prediction", "").strip()

    try:
        prompt = judge_prompt.format(
            question=question,
            correct_answer=correct_answer,
            prediction=prediction,
            response=prediction  # Some prompts use {response} instead of {prediction}
        )
        
        judge_model = OpenAIChat(id='gpt-4o-mini')
        response = await judge_model.response([Message(role="user", content=prompt)])
        judgement = response.content

        # Process judgement based on dataset type
        is_correct = False
        if "browsecomp" in dataset or "qa" in dataset:
            judge_result = extract_correct_judgement(judgement)
            if judge_result and judge_result.lower() == 'yes':
                is_correct = True
            # Also check for "CORRECT" or "A" responses
            if judgement.strip().upper() in ["CORRECT", "A"]:
                is_correct = True
        else:
            is_correct = judgement.strip() == "Correct"

        return {
            "question": question,
            "answer": correct_answer,
            "prediction": prediction,
            "judgement": judgement,
            "is_correct": is_correct
        }

    except Exception as e:
        logger.error(f"Error in judgement for question: {question[:100]}... Error: {e}")
        return {
            "question": question,
            "answer": correct_answer,
            "prediction": prediction,
            "judgement": "Error",
            "is_correct": False,
            "error": str(e)
        }


def calculate_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate statistics from evaluation results."""
    tool_use_counts = []
    visit_tool_counts = []
    search_tool_counts = []
    other_tool_counts = []
    answer_lengths = []
    reasoning_lengths = []
    num_rounds = []

    for item in results:
        tool_calls = item.get("tool_calls", [])
        response = item.get("full_response", "")
        
        # Count tool calls by type
        num_tool_use = 0
        num_visit = 0
        num_search = 0
        num_other = 0
        
        if isinstance(tool_calls, list):
            for tool in tool_calls:
                tool_name = ""
                if isinstance(tool, dict):
                    tool_name = tool.get("function", {}).get("name", "").lower()
                
                num_tool_use += 1
                if "url" in tool_name or "visit" in tool_name or "crawler" in tool_name:
                    num_visit += 1
                elif "search" in tool_name:
                    num_search += 1
                else:
                    num_other += 1
        
        tool_use_counts.append(num_tool_use)
        visit_tool_counts.append(num_visit)
        search_tool_counts.append(num_search)
        other_tool_counts.append(num_other)
        
        # Answer length
        answer_lengths.append(len(item.get("prediction", "")))
        
        # Reasoning length (from messages)
        reasoning_len = 0
        messages = item.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict) and msg.get("reasoning_content"):
                reasoning_len += len(msg.get("reasoning_content", ""))
        reasoning_lengths.append(reasoning_len)
        
        # Count rounds (assistant messages)
        rounds = sum(1 for msg in messages if isinstance(msg, dict) and msg.get("role") == "assistant")
        num_rounds.append(rounds)

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0

    return {
        "total_instances": len(results),
        "avg_tool_calls": round(safe_avg(tool_use_counts), 2),
        "avg_visit_calls": round(safe_avg(visit_tool_counts), 2),
        "avg_search_calls": round(safe_avg(search_tool_counts), 2),
        "avg_other_calls": round(safe_avg(other_tool_counts), 2),
        "avg_answer_length": round(safe_avg(answer_lengths), 2),
        "avg_reasoning_length": round(safe_avg(reasoning_lengths), 2),
        "avg_rounds": round(safe_avg(num_rounds), 2),
        "max_rounds": max(num_rounds) if num_rounds else 0,
    }


def aggregate_trajectory_stats(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate the per-instance session-log trajectories.

    Read straight from ``SessionLog.trajectory_stats()``, i.e. from what the
    runner actually wrote to the transcript: token usage, cache reads, tool
    error rate, per-tool distribution. Purely additive — accuracy and the
    existing ``statistics`` block are untouched, so old summaries stay
    comparable.

    Instances whose run produced no trajectory (crash before the first write)
    contribute nothing instead of a zero, so the averages describe the runs
    that happened.
    """
    trajectories = [
        item["trajectory"] for item in results
        if isinstance(item.get("trajectory"), dict) and item["trajectory"].get("entries")
    ]
    if not trajectories:
        return {"instances_with_trajectory": 0}

    summed_keys = (
        "tool_calls", "tool_errors", "tool_call_rounds", "assistant_messages",
        "input_tokens", "output_tokens", "total_tokens",
        "cached_tokens", "cache_read_tokens", "reasoning_tokens", "compactions",
    )
    totals = {key: sum(int(t.get(key, 0)) for t in trajectories) for key in summed_keys}
    tools_by_name: Dict[str, int] = {}
    for trajectory in trajectories:
        for name, count in (trajectory.get("tools_by_name") or {}).items():
            tools_by_name[name] = tools_by_name.get(name, 0) + int(count)

    n = len(trajectories)
    executed = totals["tool_calls"]
    return {
        "instances_with_trajectory": n,
        "avg_tool_calls_logged": round(totals["tool_calls"] / n, 2),
        "avg_tool_call_rounds": round(totals["tool_call_rounds"] / n, 2),
        "avg_assistant_steps": round(totals["assistant_messages"] / n, 2),
        "tool_error_rate": round(totals["tool_errors"] / executed, 4) if executed else 0.0,
        "avg_input_tokens": round(totals["input_tokens"] / n, 2),
        "avg_output_tokens": round(totals["output_tokens"] / n, 2),
        "avg_total_tokens": round(totals["total_tokens"] / n, 2),
        "avg_cached_tokens": round(totals["cached_tokens"] / n, 2),
        "avg_cache_read_tokens": round(totals["cache_read_tokens"] / n, 2),
        "avg_reasoning_tokens": round(totals["reasoning_tokens"] / n, 2),
        "compactions": totals["compactions"],
        "tools_by_name": dict(sorted(tools_by_name.items(), key=lambda kv: -kv[1])),
        "totals": totals,
    }


async def evaluate_instance(
    model_name: str,
    instance: Dict,
    debug: bool = False,
    session_log_dir: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single instance using multi-round Agent.
    
    Args:
        model_name: Model ID to use
        instance: Test instance with 'question' and 'answer'
        debug: Enable debug mode
        session_log_dir: Where to write this instance's session transcript. The
            transcript is what trajectory metrics are read back from, so without
            it the result carries no ``trajectory`` block.
        session_id: Session id for that transcript (one per instance).
    
    Returns:
        Evaluation result dict
    """
    question = instance.get('Question', instance.get('question', ''))
    ground_truth = instance.get('Answer', instance.get('answer', ''))

    try:
        # Create Agent with built-in tools
        agent = Agent(
            model=ArkChat(id=model_name),
            debug=debug,
            tools=get_builtin_tools(
                include_web_search=True,
                include_fetch_url=True,
                include_execute=True,
                include_todos=True,
                include_file_tools=True,
            ),
            prompt_config=PromptConfig(markdown=True, enable_agentic_prompt=True),
            work_dir="./tmp/",
            instructions=SYSTEM_PROMPT_MULTI,
            # A transcript per instance: the source of the trajectory metrics.
            session_id=session_id,
            session_base_dir=session_log_dir,
            enable_session_log=session_log_dir is not None,
        )
        
        # Run the agent
        response = await agent.run(question)
        response_content = response.content if response else ""

        logger.info(f"question: {question}\nresponse: {response_content}")
        
        # Collect messages and tool calls
        messages = []
        if agent.working_memory and agent.working_memory.messages:
            messages = [msg.to_dict() for msg in agent.working_memory.messages]
        
        # Get tool call history
        tool_calls = agent.working_memory.get_tool_calls(num_calls=100)
        logger.info(f"tool_calls: {tool_calls}")
        trajectory = agent._session_log.trajectory_stats() if agent._session_log is not None else {}
        return {
            'question': question,
            'answer': ground_truth,
            'prediction': extract_answer(response_content),
            'messages': messages,
            'tool_calls': tool_calls,
            'full_response': response_content,
            'trajectory': trajectory,
        }
        
    except Exception as e:
        logger.error(f"Error evaluating: {question[:100]}... Error: {str(e)}")
        return {
            'question': question,
            'answer': ground_truth,
            'prediction': f"Error: {str(e)}",
            'messages': [],
            'tool_calls': [],
            'full_response': '',
            'trajectory': {},
            'error': str(e)
        }


async def main():
    parser = argparse.ArgumentParser(description="Deep Research Agent Evaluation")
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model ID (e.g., gpt-4o, gpt-4o-mini, o3-mini, deepseek-reasoner)')
    parser.add_argument('--dataset', type=str, default='browsecomp_zh_small',
                        choices=[
                            'browsecomp_zh_small', 'browsecomp_zh', 
                            'browsecomp_en', 'browsecomp_en_small',
                            'simple_qa', 'simple_qa_small', 'time_qa',
                            'gaia_2023_all_validation', 'xbench_deepsearch',
                            'sailorfog-QA'
                        ],
                        help='Evaluation dataset')
    parser.add_argument('--eval_n_limit', type=int, default=3,
                        help='Number of instances to evaluate (0 for all)')
    parser.add_argument('--debug', type=int, default=1,
                        help='Debug mode (0=off, 1=on)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--skip_judge', action='store_true',
                        help='Skip LLM judgement (only run predictions)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Select judge prompt based on dataset
    if args.dataset.startswith("gaia") or args.dataset == "xbench_deepsearch":
        judge_prompt = JUDGE_PROMPT_GAIA
    elif args.dataset.startswith("browsecomp"):
        judge_prompt = JUDGE_PROMPT_BC
    elif "qa" in args.dataset.lower():
        judge_prompt = JUDGE_PROMPT_QA
    else:
        judge_prompt = JUDGE_PROMPT_GAIA

    # Load dataset
    data_file = os.path.join(pwd_path, 'data', f'{args.dataset}.jsonl')
    logger.info(f"Loading dataset: {data_file}")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    test_data = load_jsonl(data_file)
    logger.info(f"Total instances: {len(test_data)}")
    
    if args.eval_n_limit > 0:
        test_data = test_data[:args.eval_n_limit]
    logger.info(f"Evaluating {len(test_data)} instances")

    # Run evaluation
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info("=" * 60)

    results = []
    debug = args.debug == 1
    
    # One transcript per instance: trajectory metrics are read back from these.
    session_log_dir = os.path.join(args.output_dir, f"session_logs-{args.dataset}")
    os.makedirs(session_log_dir, exist_ok=True)

    for index, instance in enumerate(tqdm(test_data, desc="Running Agent")):
        result = await evaluate_instance(
            model_name=args.model,
            instance=instance,
            debug=debug,
            session_log_dir=session_log_dir,
            session_id=f"{args.dataset}-{index:04d}",
        )
        results.append(result)

    # Save predictions
    predictions_file = os.path.join(args.output_dir, f'predictions-{args.dataset}.jsonl')
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"Predictions saved: {predictions_file}")

    # Calculate statistics
    statistics = calculate_statistics(results)
    # Trajectory metrics from the session transcripts (additive; accuracy and
    # the statistics block above are unchanged).
    trajectory_statistics = aggregate_trajectory_stats(results)
    
    # Judge results (unless skipped)
    if args.skip_judge:
        logger.info("Skipping LLM judgement")
        judged_results = []
        accuracy = 0
        correct_count = 0
    else:
        logger.info("Running LLM judgement...")
        # Use asyncio.gather for concurrent judge calls
        judge_tasks = [
            call_llm_judge(item, judge_prompt, args.dataset)
            for item in results
        ]
        judged_results = await asyncio.gather(*judge_tasks)

        # Calculate accuracy
        correct_count = sum(1 for r in judged_results if r.get("is_correct", False))
        accuracy = round(correct_count / len(judged_results) * 100, 2) if judged_results else 0

    # Print results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Instances: {len(results)}")
    print("-" * 40)
    
    if not args.skip_judge:
        print(f"✅ Accuracy: {accuracy}% ({correct_count}/{len(judged_results)})")
    
    print(f"📈 Avg Tool Calls: {statistics['avg_tool_calls']}")
    print(f"   - Search: {statistics['avg_search_calls']}")
    print(f"   - Visit: {statistics['avg_visit_calls']}")
    print(f"   - Other: {statistics['avg_other_calls']}")
    print(f"📝 Avg Rounds: {statistics['avg_rounds']} (max: {statistics['max_rounds']})")
    print(f"📄 Avg Answer Length: {statistics['avg_answer_length']}")
    print(f"🧠 Avg Reasoning Length: {statistics['avg_reasoning_length']}")
    if trajectory_statistics.get("instances_with_trajectory"):
        print("-" * 40)
        print(f"🧾 Trajectory (from session logs, n={trajectory_statistics['instances_with_trajectory']})")
        print(f"   - Avg assistant steps: {trajectory_statistics['avg_assistant_steps']}")
        print(f"   - Avg tool calls: {trajectory_statistics['avg_tool_calls_logged']}")
        print(f"   - Tool error rate: {trajectory_statistics['tool_error_rate']}")
        print(f"   - Avg tokens in/out: {trajectory_statistics['avg_input_tokens']}/{trajectory_statistics['avg_output_tokens']}")
        print(f"   - Avg cached tokens: {trajectory_statistics['avg_cached_tokens']}")
    print("=" * 60)

    # Save final results
    final_result = {
        "dataset": args.dataset,
        "model": args.model,
        "config": {
            "eval_n_limit": args.eval_n_limit
        },
        "accuracy": accuracy if not args.skip_judge else None,
        "correct": correct_count if not args.skip_judge else None,
        "total": len(results),
        "statistics": statistics,
        "trajectory_statistics": trajectory_statistics,
        "judged_results": judged_results if not args.skip_judge else []
    }

    summary_file = os.path.join(args.output_dir, f"summary-{args.dataset}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
