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
from agentica.tools.wikipedia_tool import WikipediaTool
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
    """Evaluate a single instance from the GAIA benchmark

    "instance_id": "c61d22de-5f6c-4958-a7f6-5e9707bd3466",
            "question": "A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?",
            "model_answer": "Utopia",
            "ground_truth": "egalitarian",
    """
    try:
        question = f"{instance['question']}\n\nPlease provide your answer within <solution> tags."
        r = await agent.arun(question)
        response = r.content

        # Extract and score the answer
        model_answer = extract_solution(response)
        score = calculate_score(model_answer, instance['ground_truth'])

        return {
            'instance_id': instance['instance_id'],
            'question': instance['question'],
            'model_answer': model_answer,
            'ground_truth': instance['ground_truth'],
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
    agent = PythonAgent(
        model=OpenAIChat(id=args.model),
        tools=[
            ShellTool(),
            JinaTool(),
            BaiduSearchTool(),
            WikipediaTool(),

        ],
        instructions=instructions,
        debug=True
    )

    # Load GAIA dataset

    ds = [
        {
            "instance_id": "14569e28-c88c-43e4-8c32-097d35b9a67d",
            "question": "In Unlambda, what exact charcter or text needs to be added to correct the following code to output \"For penguins\"? If what is needed is a character, answer with the name of the character. If there are different names for the character, use the shortest. The text location is not needed. Code:\n\n`r```````````.F.o.r. .p.e.n.g.u.i.n.si",
            "model_answer": "backtick",
            "ground_truth": "backtick",
            "score": 1.0,
            "full_response": "In Unlambda programming, an additional backtick (also known as the grave accent) is often added to ensure proper function application and execution. In the given code, the character that is needed is the backtick, and its name as used in Unlambda is \"backtick\".\n\n<solution>backtick</solution>"
        },
        {
            "instance_id": "e1fc63a2-da7a-432f-be78-7c4a95598703",
            "question": "If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary.",
            "model_answer": "17000",
            "ground_truth": "17",
            "score": 0.0,
            "full_response": "To solve this problem, we need to calculate the number of hours it would take Eliud Kipchoge to run the distance between the Earth and the Moon at its minimum perigee, using his marathon record-setting pace indefinitely.\n\n1. **Determine the minimum perigee distance to the Moon:**\n   On the Wikipedia page for the Moon, the minimum perigee (closest approach) distance is given as approximately 356,500 kilometers.\n\n2. **Determine Kipchoge's marathon record pace:**\n   Eliud Kipchoge set a marathon record with a time of 2 hours, 1 minute, and 9 seconds. The marathon distance is 42.195 kilometers.\n\n3. **Calculate Kipchoge's pace in kilometers per hour:**\n   First, convert his marathon time into hours:\n   \\[\n   \\text{Total time in hours} = 2 + \\frac{1}{60} + \\frac{9}{3600} \\approx 2.0192 \\text{ hours}\n   \\]\n   Then, calculate his pace:\n   \\[\n   \\text{Pace} = \\frac{42.195 \\text{ km}}{2.0192 \\text{ hours}} \\approx 20.9 \\text{ km/h}\n   \\]\n\n4. **Calculate the time to run the distance to the Moon at this pace:**\n   Divide the Moon's closest distance by Kipchoge's pace:\n   \\[\n   \\text{Time} = \\frac{356,500 \\text{ km}}{20.9 \\text{ km/h}} \\approx 17058 \\text{ hours}\n   \\]\n\n5. **Convert the time to thousands of hours and round to the nearest 1000 hours:**\n   \\[\n   \\text{Thousands of hours} = \\frac{17058 \\text{ hours}}{1000} \\approx 17.058\n   \\]\n   Rounded to the nearest thousand hours gives 17000 hours.\n\nTherefore, it would take Kipchoge approximately 17000 thousand hours to run the distance between the Earth and the Moon at its closest approach.\n\n<solution>17000</solution>"
        },
        {
            "instance_id": "8e867cd7-cff9-4e6c-867a-ff5ddc2550be",
            "question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.",
            "model_answer": "6",
            "ground_truth": "3",
            "score": 0.0,
            "full_response": "First, I will access the latest version of English Wikipedia to determine the number of studio albums published by Mercedes Sosa between 2000 and 2009.\n\nBased on the information available:\n\nMercedes Sosa released the following studio albums between 2000 and 2009:\n1. \"Misa Criolla\" (2000)\n2. \"Ac\u00fastico\" (2000)\n3. \"Gestos de amor\" (2000)\n4. \"Coraz\u00f3n Libre\" (2005)\n5. \"Cantora 1\" (2009)\n6. \"Cantora 2\" (2009)\n\nTherefore, she released 6 studio albums during that period.\n\n<solution>6</solution>"
        },
        {
            "instance_id": "3627a8be-a77f-41bb-b807-7e1bd4c0ebdf",
            "question": "The object in the British Museum's collection with a museum number of 2012,5015.17 is the shell of a particular mollusk species. According to the abstract of a research article published in Science Advances in 2021, beads made from the shells of this species were found that are at least how many thousands of years old?",
            "model_answer": "To find the relevant information about the age of the beads made from the shells of the mollusk species in the British Museum's collection, I will perform a web search on the research article published in Science Advances in 2021.",
            "ground_truth": "142",
            "score": 0.0,
            "full_response": "To find the relevant information about the age of the beads made from the shells of the mollusk species in the British Museum's collection, I will perform a web search on the research article published in Science Advances in 2021."
        },

    ]
    # Run evaluation
    results = []
    for instance in tqdm(ds):
        result = await evaluate_instance(agent, instance)
        results.append(result)
        output_file = os.path.join(args.output_dir, f'gaia_eval_results_testcase_{args.level}.json')
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
    metrics_file = os.path.join(args.output_dir, f'gaia_eval_metrics_testcase_{args.level}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Evaluation complete. Average score: {metrics['average_score']:.3f}")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
