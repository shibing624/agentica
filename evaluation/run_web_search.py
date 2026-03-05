# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: WebSearchAgent Evaluation Script

Evaluates WebSearchAgent on BrowseComp/GAIA/SimpleQA benchmarks
using the structured search pipeline (deep_search method).

Usage:
    python run_web_search.py --model gpt-4o --dataset browsecomp_zh_small --eval_n_limit 10
"""
import argparse
import asyncio
import json
import os
import sys
from typing import List, Dict, Any

from loguru import logger
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentica import OpenAIChat, Message
from agentica.web_search_agent import WebSearchAgent
from agentica.agent.config import ToolConfig
from prompt import JUDGE_PROMPT_GAIA, JUDGE_PROMPT_BC, JUDGE_PROMPT_QA

pwd_path = os.path.abspath(os.path.dirname(__file__))


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def get_judge_prompt(dataset_name: str) -> str:
    """Get judge prompt based on dataset name."""
    if "browsecomp" in dataset_name.lower():
        return JUDGE_PROMPT_BC
    elif "gaia" in dataset_name.lower():
        return JUDGE_PROMPT_GAIA
    else:
        return JUDGE_PROMPT_QA


async def judge_answer(judge_model, question: str, pred: str, gold: str, judge_prompt: str) -> Dict[str, Any]:
    """Use LLM to judge if the prediction matches the gold answer."""
    prompt = judge_prompt.format(
        question=question,
        ground_truth=gold,
        model_output=pred,
    )
    try:
        response = await judge_model.response([
            Message(role="user", content=prompt),
        ])
        text = response.content.strip().lower() if response.content else ""
        is_correct = "correct" in text and "incorrect" not in text
        return {"is_correct": is_correct, "judge_output": response.content}
    except Exception as e:
        logger.error(f"Judge error: {e}")
        return {"is_correct": False, "judge_output": str(e)}


async def evaluate(args):
    """Run evaluation on the specified dataset."""
    # Load dataset
    dataset_path = os.path.join(pwd_path, f"{args.dataset}.jsonl")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    instances = load_jsonl(dataset_path)
    if args.eval_n_limit and args.eval_n_limit > 0:
        instances = instances[:args.eval_n_limit]

    logger.info(f"Evaluating {len(instances)} instances from {args.dataset}")

    # Create WebSearchAgent
    model = OpenAIChat(id=args.model)
    agent = WebSearchAgent(
        model=model,
        name="EvalWebSearchAgent",
        description="WebSearchAgent for evaluation",
        max_search_rounds=args.max_search_rounds,
        max_queries_per_round=5,
        min_evidence_count=2,
        confidence_threshold=args.confidence_threshold,
        enable_query_decomposition=True,
        enable_answer_verification=True,
        enable_evidence_tracking=True,
        enable_step_reflection=True,
        enable_context_overflow_handling=True,
        tool_config=ToolConfig(compress_tool_results=True),
        enable_repetition_detection=True,
    )

    # Create judge model
    judge_model = OpenAIChat(id=args.judge_model)
    judge_prompt = get_judge_prompt(args.dataset)

    # Run evaluation
    results = []
    correct = 0
    total = 0

    output_path = os.path.join(pwd_path, f"results_{args.dataset}_{args.model.replace('/', '_')}.jsonl")

    for instance in tqdm(instances, desc="Evaluating"):
        question = instance.get("question", instance.get("input", ""))
        gold = instance.get("answer", instance.get("expected_answer", ""))

        if not question:
            continue

        total += 1
        try:
            # Use deep_search for structured search
            agent.reset_search_state()
            pred = await agent.deep_search(question)

            # Judge the answer
            judge_result = await judge_answer(judge_model, question, pred, gold, judge_prompt)
            is_correct = judge_result["is_correct"]
            if is_correct:
                correct += 1

            status = agent.get_search_status()
            result = {
                "question": question,
                "gold": gold,
                "pred": pred,
                "is_correct": is_correct,
                "judge_output": judge_result["judge_output"],
                "search_status": status,
            }
            results.append(result)

            # Write result immediately
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            logger.info(
                f"[{total}] Correct: {is_correct}, "
                f"Rounds: {status['current_round']}, Evidence: {status['evidence_count']}, "
                f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%"
            )
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            results.append({
                "question": question,
                "gold": gold,
                "pred": "",
                "is_correct": False,
                "error": str(e),
            })

    # Summary
    accuracy = correct / total * 100 if total > 0 else 0
    logger.info(f"\nFinal Results: {correct}/{total} = {accuracy:.1f}%")
    logger.info(f"Results saved to: {output_path}")

    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="WebSearchAgent Evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model ID")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Judge model ID")
    parser.add_argument("--dataset", type=str, default="browsecomp_zh_small", help="Dataset name")
    parser.add_argument("--eval_n_limit", type=int, default=0, help="Limit number of instances (0=all)")
    parser.add_argument("--max_search_rounds", type=int, default=10, help="Max search rounds")
    parser.add_argument("--confidence_threshold", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()

    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
