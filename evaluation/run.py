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
from loguru import logger
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentica import DeepAgent, OpenAIChat, ZhipuAIChat, Message, JinaTool, SearchSerperTool
from agentica.tools.baidu_search_tool import BaiduSearchTool
from agentica.tools.url_crawler_tool import UrlCrawlerTool
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
        
        judge_model = ZhipuAIChat(model='glm-4-flash')
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


async def evaluate_instance(
    model_name: str,
    instance: Dict,
    max_rounds: int = 100,
    max_tokens: int = 128000,
    debug: bool = False,
    tools_config: str = "baidu"
) -> Dict[str, Any]:
    """
    Evaluate a single instance using multi-round Agent.
    
    Args:
        model_name: Model ID to use
        instance: Test instance with 'question' and 'answer'
        max_rounds: Maximum rounds for multi-round strategy
        max_tokens: Maximum tokens limit
        debug: Enable debug mode
        tools_config: Tool configuration ("baidu", "serper", "jina", "all")
    
    Returns:
        Evaluation result dict
    """
    question = instance.get('Question', instance.get('question', ''))
    ground_truth = instance.get('Answer', instance.get('answer', ''))

    try:
        # Configure tools based on setting
        tools = []
        if tools_config in ["baidu", "all"]:
            tools.append(BaiduSearchTool())
        if tools_config in ["serper", "all"]:
            tools.append(SearchSerperTool())
        if tools_config in ["jina", "all"]:
            tools.append(JinaTool(
                jina_search=False,
                jina_reader_by_goal=True,
                jina_reader=True,
                work_dir='saved_html'
            ))
        # Always add URL crawler for visiting pages
        tools.append(UrlCrawlerTool(work_dir='saved_html'))
        
        # Create DeepAgent with deep research capabilities
        agent = DeepAgent(
            model=OpenAIChat(id=model_name),
            tools=tools,
            add_history_to_messages=False,
            enable_deep_research=True,
            include_web_search=False,  # Use external search tools configured above
            include_fetch_url=False,   # Use UrlCrawlerTool configured above
            include_file_tools=False,
            include_execute=False,
            include_todos=False,
            include_task=False,
            include_skills=False,
            debug=debug,
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
        tool_calls = []
        tool_calls_str = agent.get_tool_call_history(100)
        if tool_calls_str:
            try:
                tool_calls = json.loads(tool_calls_str)
            except json.JSONDecodeError:
                tool_calls = [{"raw": tool_calls_str}]
        logger.info(f"tool_calls: {tool_calls}")
        return {
            'question': question,
            'answer': ground_truth,
            'prediction': extract_answer(response_content),
            'messages': messages,
            'tool_calls': tool_calls,
            'full_response': response_content
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
    parser.add_argument('--max_rounds', type=int, default=100,
                        help='Maximum rounds for multi-round strategy')
    parser.add_argument('--max_tokens', type=int, default=128000,
                        help='Maximum tokens limit')
    parser.add_argument('--tools', type=str, default='baidu',
                        choices=['baidu', 'serper', 'jina', 'all'],
                        help='Search tools to use')
    parser.add_argument('--debug', type=int, default=0,
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
    logger.info(f"Multi-round: max_rounds={args.max_rounds}, max_tokens={args.max_tokens}")
    logger.info(f"Tools: {args.tools}")
    logger.info("=" * 60)

    results = []
    debug = args.debug == 1
    
    for instance in tqdm(test_data, desc="Running Agent"):
        result = await evaluate_instance(
            model_name=args.model,
            instance=instance,
            max_rounds=args.max_rounds,
            max_tokens=args.max_tokens,
            debug=debug,
            tools_config=args.tools
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
    print("=" * 60)

    # Save final results
    final_result = {
        "dataset": args.dataset,
        "model": args.model,
        "config": {
            "max_rounds": args.max_rounds,
            "max_tokens": args.max_tokens,
            "tools": args.tools,
            "eval_n_limit": args.eval_n_limit
        },
        "accuracy": accuracy if not args.skip_judge else None,
        "correct": correct_count if not args.skip_judge else None,
        "total": len(results),
        "statistics": statistics,
        "judged_results": judged_results if not args.skip_judge else []
    }

    summary_file = os.path.join(args.output_dir, f"summary-{args.dataset}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    asyncio.run(main())
