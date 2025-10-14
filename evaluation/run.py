# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import os
import json
import asyncio
import concurrent.futures
import re
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
import sys
import tiktoken

sys.path.append('../..')
from agentica import Agent, PythonAgent, ShellTool, JinaTool, SearchSerperTool, OpenAIChat, ZhipuAIChat, Message
from agentica.tools.baidu_search_tool import BaiduSearchTool
from prompt import JUDGE_PROMPT_GAIA, JUDGE_PROMPT_BC, JUDGE_PROMPT_QA

pwd_path = os.path.abspath(os.path.dirname(__file__))


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_answer(text: str) -> str:
    """Extract the answer from between <answer> tags"""
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return matches[0].strip() if matches else text.strip()


def extract_correct_judgement(response: str) -> str:
    match = re.search(r'correct\s*:\s*(yes|no)', response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    else:
        return None


def call_llm_judge(item, judge_prompt, dataset):
    """Judge if predicted answer matches ground-truth"""
    question = ''
    correct_answer = ''
    try:
        question = item["question"]
        correct_answer = item["answer"]
        response = item["prediction"].strip()

        prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)
        m = ZhipuAIChat(model='glm-4-flash')
        r = m.response([Message(role="user", content=prompt)])
        response = r.content

        # For BrowseComp and XX_QA datasets, process judgement
        if ("browsecomp" in dataset) or ("qa" in dataset):
            judge = extract_correct_judgement(response)
            if judge and judge.lower() == 'yes':
                response = "Correct"

        return {
            "question": question,
            "answer": correct_answer,
            "judgement": response,
            "is_correct": response == "Correct"
        }

    except Exception as e:
        logger.error(f"Error judgement for question: {question}: {e}")
        return {
            "question": question,
            "answer": correct_answer,
            "judgement": "Error",
            "is_correct": False,
            "error": str(e)
        }


def single_round_statistics(results):
    """Calculate statistics for a single round"""
    num_invalid, num_extra = 0, 0
    tool_use_cnt, visit_tool_cnt, search_tool_cnt, other_tool_cnt = [], [], [], []
    all_ans_lengths, all_think_lengths = [], []

    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    for item in results:
        texts = item.get("messages", [])
        final_msg = texts[-1]["content"] if len(texts) else ""

        # Analyze answer
        if "<answer>" not in final_msg or "</answer>" not in final_msg:
            num_invalid += 1
            answer_length = 0
        else:
            answer_length = len(final_msg.split("<answer>")[1].split("</answer>")[0].strip())

        # Analyze tool use & thinking
        num_tool_use, num_visit_tool, num_search_tool, num_other_tool = 0, 0, 0, 0
        think_lengths = []
        for idx in range(2, len(texts), 2):
            response = texts[idx]["content"]
            tool = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)

            if tool:
                try:
                    tool_name = tool[0].split("name\": \"")[1].split("\"")[0].strip()
                except Exception:
                    tool_name = ""

                num_tool_use += 1
                if "visit" in tool_name:
                    num_visit_tool += 1
                elif "search" in tool_name:
                    num_search_tool += 1
                else:
                    num_other_tool += 1

            think_lengths.append(len(response))

        tool_use_cnt.append(num_tool_use)
        visit_tool_cnt.append(num_visit_tool)
        search_tool_cnt.append(num_search_tool)
        other_tool_cnt.append(num_other_tool)

        all_ans_lengths.append(answer_length)
        think_length = sum(think_lengths) / len(think_lengths) if think_lengths else 0
        all_think_lengths.append(think_length)

        # Overlength
        if len(tokenizer.encode("".join([text["content"] for text in texts]))) > 30000:
            num_extra += 1

    return {
        "extra_length": num_extra,
        "num_invalid": num_invalid,
        "avg_action": sum(tool_use_cnt) / len(tool_use_cnt) if tool_use_cnt else 0,
        "avg_visit_action": sum(visit_tool_cnt) / len(visit_tool_cnt) if visit_tool_cnt else 0,
        "avg_search_action": sum(search_tool_cnt) / len(search_tool_cnt) if search_tool_cnt else 0,
        "avg_other_action": sum(other_tool_cnt) / len(other_tool_cnt) if other_tool_cnt else 0,
        "avg_ans_length": sum(all_ans_lengths) / len(all_ans_lengths) if all_ans_lengths else 0,
        "avg_think_length": sum(all_think_lengths) / len(all_think_lengths) if all_think_lengths else 0
    }


async def evaluate_instance(agent, instance, dataset_name: str) -> Dict[str, Any]:
    """Evaluate a single instance"""
    try:
        question = instance.get('Question', instance.get('question', ''))
        ground_truth = instance.get('Answer', instance.get('answer', ''))

        # Create messages format
        messages = [
            {"role": "user", "content": question}
        ]

        # Run agent
        r = await agent.arun(question)
        response = r.content

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response})

        return {
            'question': question,
            'answer': ground_truth,
            'prediction': extract_answer(response),
            'messages': messages,
            'full_response': response
        }
    except Exception as e:
        logger.error(f"Error evaluating instance: {str(e)}")
        question = instance.get('Question', instance.get('question', ''))
        ground_truth = instance.get('Answer', instance.get('answer', ''))
        return {
            'question': question,
            'answer': ground_truth,
            'prediction': f"Error: {str(e)}",
            'messages': [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"Error: {str(e)}"}
            ],
            'error': str(e)
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--dataset', type=str, default='browsecomp_zh_small',
                        choices=['browsecomp_zh_small', 'browsecomp_zh', 'browsecomp_en',
                                 'browsecomp_en_small', 'simple_qa', 'simple_qa_small', 'time_qa',
                                 'gaia_2023_all_validation', ])
    parser.add_argument('--eval_n_limit', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--restore_result_path', default='summary.json', help="record result")
    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Setup judge prompt based on dataset
    if args.dataset in ["gaia", "webwalker", "xbench-deepsearch", "hle"]:
        judge_prompt = JUDGE_PROMPT_GAIA
    elif args.dataset.startswith("browsecomp_zh") or args.dataset.startswith("browsecomp_en"):
        judge_prompt = JUDGE_PROMPT_BC
    elif args.dataset.endswith("qa") or args.dataset.endswith("qa_small"):
        judge_prompt = JUDGE_PROMPT_QA
    else:
        judge_prompt = JUDGE_PROMPT_GAIA

    agent = Agent(
        model=OpenAIChat(id=args.model),
        tools=[
            JinaTool(jina_search=False, jina_reader_by_goal=True, jina_reader=False, work_dir='saved_html'),
            # SearchSerperTool(),
            BaiduSearchTool()
        ],
        enable_multi_round=True,
        max_rounds=15,
        max_tokens=30000,
        debug=False
    )

    # Load evaluation dataset
    data_file = os.path.join(pwd_path, 'data', args.dataset + '.jsonl')
    logger.info(f"Loading dataset from {data_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found")

    test_data = load_jsonl(data_file)
    logger.info(f"Total {len(test_data)} instances in the dataset")
    if args.eval_n_limit:
        test_data = test_data[:args.eval_n_limit]
    logger.info(f"Loaded {len(test_data)} test instances")

    # Run single round evaluation
    logger.info("Starting Running")
    results = []

    for instance in tqdm(test_data, desc="Running"):
        result = await evaluate_instance(agent, instance, args.dataset)
        results.append(result)
        await asyncio.sleep(0.1)

    # Save prediction results
    output_file = os.path.join(args.output_dir, 'predictions.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"Predictions saved to {output_file}")

    # Judge results
    logger.info("Starting judgement")
    judged_results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(call_llm_judge, item, judge_prompt, args.dataset): item for item in results}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Judging"):
            judged_results.append(future.result())

    # Calculate statistics
    statistics = single_round_statistics(results)

    # Calculate accuracy
    correct_count = sum(1 for r in judged_results if r["is_correct"])
    accuracy = round(correct_count / len(judged_results) * 100, 2)

    # Print results
    print(f"===========")
    print(f"Accuracy: {accuracy}%")
    print(f"Correct: {correct_count}/{len(judged_results)}")
    print(f"# Invalid {statistics['num_invalid']}  # Extra Length {statistics['extra_length']}")
    print(
        f"Avg. Action {statistics['avg_action']:.2f}  Avg. Visit Action {statistics['avg_visit_action']:.2f}  Avg. Search Action {statistics['avg_search_action']:.2f}  Avg. Other Action {statistics['avg_other_action']:.2f}")
    print(
        f"Avg. Answer Length {statistics['avg_ans_length']:.2f}  Avg. Thinking Length {statistics['avg_think_length']:.2f}")
    print(f"===========")

    # Save final results
    final_result = {
        "dataset": args.dataset,
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(judged_results),
        "statistics": statistics,
        "judged_results": judged_results
    }

    # save restore_result_path to json file
    with open(args.restore_result_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)
    logger.info(f"Final results saved to {args.restore_result_path}")


if __name__ == "__main__":
    asyncio.run(main())
