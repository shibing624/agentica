# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import os
import json
import time
import asyncio
import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
import sys

sys.path.append('../../..')
from agentica import Agent, PythonAgent, ShellTool, JinaTool, SearchSerperTool, OpenAIChat
from agentica.tools.baidu_search_tool import BaiduSearchTool


def extract_solution(text: str) -> str:
    """Extract the solution from between <solution> tags"""
    import re
    matches = re.findall(r'<solution>(.*?)</solution>', text, re.DOTALL)
    return matches[0].strip() if matches else text.strip()


def calculate_score(prediction: str, ground_truth: str) -> float:
    """
    Calculate the score for a prediction against the ground truth.
    This is a simple exact match scorer - you might want to implement
    a more sophisticated scoring method based on your needs.
    """
    return float(prediction.strip().lower() == ground_truth.strip().lower())


async def evaluate_instance(
        agent,
        instance,
) -> Dict[str, Any]:
    """Evaluate a single instance from the GAIA benchmark"""
    try:
        question = f"{instance['Question']}\n\nPlease provide your answer within <solution> tags."
        r = await agent.arun(question)
        response = r.content

        # Extract and score the answer
        model_answer = extract_solution(response)
        score = calculate_score(model_answer, instance['Final answer'])

        return {
            'instance_id': instance['instance_id'],
            'question': instance['Question'],
            'model_answer': model_answer,
            'ground_truth': instance['Final answer'],
            'score': score,
            'full_response': response
        }
    except Exception as e:
        logger.error(f"Error evaluating instance {instance.get('instance_id', 'unknown')}: {str(e)}")
        return {
            'instance_id': instance.get('instance_id', 'unknown'),
            'error': str(e),
            'score': 0
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--data_split', type=str, default='validation')
    parser.add_argument('--level', type=str, default='2023_all')
    parser.add_argument('--eval_n_limit', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='gaia_results')
    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Setup Agent
    instructions = """You are a highly capable AI assistant evaluating questions from the GAIA benchmark. 
Your task is to provide accurate and concise answers to questions.
You have access to web search and can process various types of data.
Always provide your final answer within <solution> tags.
Be direct and precise in your responses."""
    agent = Agent(
        model=OpenAIChat(id=args.model),
        # tools=[
        #     ShellTool(),
        #     JinaTool(),
        #     BaiduSearchTool()
        # ],
        instructions=instructions,
        # debug=True
    )

    # Load GAIA dataset
    logger.info(f"Loading GAIA dataset (level: {args.level}, split: {args.data_split})")
    dataset = load_dataset('gaia-benchmark/GAIA', args.level)
    gaia_tests = dataset[args.data_split].to_pandas()
    gaia_tests.rename(columns={'task_id': 'instance_id'}, inplace=True)

    if args.eval_n_limit:
        gaia_tests = gaia_tests.head(args.eval_n_limit)

    # Run evaluation
    results = []
    for _, instance in tqdm(gaia_tests.iterrows(), total=len(gaia_tests)):
        result = await evaluate_instance(agent, instance)
        results.append(result)
        # Save intermediate results
        output_file = os.path.join(args.output_dir, f'gaia_eval_results_{args.level}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Calculate and save metrics
    scores = [r['score'] for r in results if 'score' in r]
    metrics = {
        'average_score': sum(scores) / len(scores) if scores else 0,
        'total_evaluated': len(scores),
        'total_successful': sum(scores),
        'model': args.model,
        'level': args.level,
        'data_split': args.data_split
    }

    # Save metrics
    metrics_file = os.path.join(args.output_dir, f'gaia_eval_metrics_{args.level}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation complete. Average score: {metrics['average_score']:.3f}")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
