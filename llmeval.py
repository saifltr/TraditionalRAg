#!/usr/bin/env python3
"""
RAG Evaluation Script using DeepEval
Compares Simple RAG vs Advanced RAG responses
Uses research-backed metrics: G-Eval, Coherence, Faithfulness, Answer Relevancy
"""

import json
import os
from typing import Dict, List
import statistics
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.evaluate.configs import AsyncConfig  # FIXED: Correct import path
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Custom Claude wrapper for DeepEval
class ClaudeModel(DeepEvalBaseLLM):
    def __init__(self, model="claude-sonnet-4-20250514"):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    def load_model(self):
        return self.client
    
    def generate(self, prompt: str) -> str:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self):
        return self.model

def load_json_file(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_qa_pairs(data: Dict) -> List[Dict]:
    """Extract question-answer pairs from JSON structure"""
    qa_pairs = []
    
    # Handle the session-based structure from document 3
    if 'queries' in data:
        for query in data['queries']:
            qa_pairs.append({
                'question': query['question'],
                'answer': query['answer']
            })
    # Handle the messages-based structure from document 4
    elif 'messages' in data:
        for i in range(0, len(data['messages']), 2):
            if i + 1 < len(data['messages']):
                user_msg = data['messages'][i]
                assistant_msg = data['messages'][i + 1]
                if user_msg['role'] == 'user' and assistant_msg['role'] == 'assistant':
                    qa_pairs.append({
                        'question': user_msg['content'],
                        'answer': assistant_msg['content']
                    })
    
    return qa_pairs

def setup_metrics(claude_model):
    """Setup DeepEval metrics for evaluation"""
    
    # G-Eval metrics for comprehensive evaluation
    comprehensiveness = GEval(
        name="Comprehensiveness",
        criteria="Evaluate whether the response thoroughly covers all key aspects of the question, including main concepts, mechanisms, and relevant details.",
        evaluation_steps=[
            "Identify all key concepts and aspects mentioned in the question",
            "Check if the response addresses each of these aspects",
            "Assess the depth and completeness of coverage for each aspect",
            "Determine if any important information is missing"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=claude_model
    )
    
    depth_metric = GEval(
        name="Depth of Explanation",
        criteria="Evaluate whether the response provides detailed explanations of mechanisms, causality, and context rather than just listing facts.",
        evaluation_steps=[
            "Check if the response explains HOW and WHY things work, not just WHAT they are",
            "Assess if causal relationships and mechanisms are explained",
            "Evaluate if contextual information enhances understanding",
            "Determine if explanations go beyond surface-level descriptions"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=claude_model
    )
    
    coherence_metric = GEval(
        name="Coherence",
        criteria="Evaluate the logical structure, flow, and readability of the response.",
        evaluation_steps=[
            "Check if ideas are presented in a logical, easy-to-follow order",
            "Assess whether transitions between sections are smooth and natural",
            "Evaluate if the response has clear hierarchical structure",
            "Determine if the writing maintains consistency throughout"
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=claude_model
    )
    
    synthesis_metric = GEval(
        name="Synthesis Quality",
        criteria="Evaluate whether the response synthesizes information into unified understanding versus just listing facts.",
        evaluation_steps=[
            "Check if information from multiple concepts is integrated coherently",
            "Assess whether connections between ideas are explicitly explained",
            "Evaluate if the response builds a coherent narrative rather than fragmented points",
            "Determine if insights emerge from combining different pieces of information"
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=claude_model
    )
    
    pedagogical_metric = GEval(
        name="Pedagogical Value",
        criteria="Evaluate how effectively the response helps readers truly understand concepts versus just providing information.",
        evaluation_steps=[
            "Check if complex concepts are explained in accessible ways",
            "Assess whether examples or analogies aid understanding",
            "Evaluate if the response anticipates and addresses potential confusion",
            "Determine if the response enables readers to apply knowledge"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=claude_model
    )
    
    # Answer Relevancy metric (built-in DeepEval metric)
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=claude_model
    )
    
    return [comprehensiveness, depth_metric, coherence_metric, 
            synthesis_metric, pedagogical_metric, relevancy_metric]

def extract_scores(test_cases: List[LLMTestCase]) -> Dict:
    """Extract scores from evaluated test cases"""
    
    # Collect all unique metric names from test cases
    all_metric_names = set()
    for test_case in test_cases:
        if hasattr(test_case, 'metrics_data') and test_case.metrics_data:
            for metric_data in test_case.metrics_data:
                # Handle both metrics with .name and without
                metric_name = getattr(metric_data, 'name', metric_data.__class__.__name__)
                all_metric_names.add(metric_name)
    
    scores_by_metric = {name: [] for name in all_metric_names}
    
    for test_case in test_cases:
        if hasattr(test_case, 'metrics_data') and test_case.metrics_data:
            for metric_data in test_case.metrics_data:
                metric_name = getattr(metric_data, 'name', metric_data.__class__.__name__)
                if metric_name in scores_by_metric:
                    scores_by_metric[metric_name].append({
                        'score': metric_data.score,
                        'reason': getattr(metric_data, 'reason', getattr(metric_data, 'reasoning', ""))
                    })
    
    # Calculate aggregates
    aggregates = {}
    for metric_name, scores in scores_by_metric.items():
        if scores:
            score_values = [s['score'] for s in scores]
            aggregates[metric_name] = {
                'mean': statistics.mean(score_values),
                'median': statistics.median(score_values),
                'stdev': statistics.stdev(score_values) if len(score_values) > 1 else 0,
                'scores': scores
            }
    
    # Calculate overall average
    all_scores = []
    for metric_scores in scores_by_metric.values():
        all_scores.extend([s['score'] for s in metric_scores])
    
    if all_scores:
        aggregates['Overall'] = {
            'mean': statistics.mean(all_scores),
            'median': statistics.median(all_scores),
            'stdev': statistics.stdev(all_scores) if len(all_scores) > 1 else 0
        }
    
    return aggregates

def main():
    print("=" * 80)
    print("RAG EVALUATION: Simple RAG vs Advanced RAG")
    print("Using DeepEval Framework with Research-Backed Metrics")
    print("=" * 80)
    
    # Load JSON files
    print("\n📂 Loading data files...")
    simple_rag = load_json_file('simple_rag.json')
    advanced_rag = load_json_file('advanced_rag.json')
    
    # Extract QA pairs
    simple_qa = extract_qa_pairs(simple_rag)
    advanced_qa = extract_qa_pairs(advanced_rag)
    
    print(f"✓ Simple RAG: {len(simple_qa)} Q&A pairs")
    print(f"✓ Advanced RAG: {len(advanced_qa)} Q&A pairs")
    
    # Setup Claude model for evaluation
    print("\n🔧 Setting up Claude as evaluation model...")
    claude_model = ClaudeModel()
    
    # Setup metrics
    print("📊 Initializing evaluation metrics...")
    metrics = setup_metrics(claude_model)
    print(f"✓ {len(metrics)} metrics ready:")
    for metric in metrics:
        # Handle both GEval (has .name) and AnswerRelevancyMetric (no .name)
        metric_name = getattr(metric, 'name', metric.__class__.__name__)
        print(f"  - {metric_name}")
    
    # Ensure both have same questions
    num_questions = min(len(simple_qa), len(advanced_qa))
    
    print(f"\n🔍 Evaluating {num_questions} question-answer pairs...\n")
    
    # Create test cases for Simple RAG
    print("=" * 80)
    print("EVALUATING SIMPLE RAG")
    print("=" * 80)
    simple_test_cases = []
    for idx, qa in enumerate(simple_qa[:num_questions]):
        print(f"\n[{idx + 1}/{num_questions}] {qa['question'][:80]}...")
        test_case = LLMTestCase(
            input=qa['question'],
            actual_output=qa['answer']
        )
        simple_test_cases.append(test_case)
    
    # Evaluate Simple RAG
    print("\n⚙️ Running DeepEval evaluation on Simple RAG...")
    async_config = AsyncConfig(run_async=False)
    
    simple_results = evaluate(
        test_cases=simple_test_cases,
        metrics=metrics,
        async_config=async_config
    )
    
    # Create test cases for Advanced RAG
    print("\n" + "=" * 80)
    print("EVALUATING ADVANCED RAG")
    print("=" * 80)
    advanced_test_cases = []
    for idx, qa in enumerate(advanced_qa[:num_questions]):
        print(f"\n[{idx + 1}/{num_questions}] {qa['question'][:80]}...")
        test_case = LLMTestCase(
            input=qa['question'],
            actual_output=qa['answer']
        )
        advanced_test_cases.append(test_case)
    
    # Evaluate Advanced RAG
    print("\n⚙️ Running DeepEval evaluation on Advanced RAG...")
    advanced_results = evaluate(
        test_cases=advanced_test_cases,
        metrics=metrics,
        async_config=async_config
    )
    
    # Extract and aggregate scores
    print("\n" + "=" * 80)
    print("CALCULATING AGGREGATE RESULTS")
    print("=" * 80)
    
    simple_aggregates = extract_scores(simple_test_cases)
    advanced_aggregates = extract_scores(advanced_test_cases)
    
    # Print comparison table
    print("\n{:<30} {:>15} {:>15} {:>15}".format("Metric", "Simple RAG", "Advanced RAG", "Difference"))
    print("-" * 80)
    
    # Get all metric names dynamically from aggregates
    all_metrics = set(simple_aggregates.keys()) | set(advanced_aggregates.keys())
    # Sort metrics, put Overall last
    sorted_metrics = sorted([m for m in all_metrics if m != 'Overall']) + (['Overall'] if 'Overall' in all_metrics else [])
    
    for metric_name in sorted_metrics:
        if metric_name in simple_aggregates and metric_name in advanced_aggregates:
            simple_mean = simple_aggregates[metric_name]['mean']
            advanced_mean = advanced_aggregates[metric_name]['mean']
            diff = advanced_mean - simple_mean
            
            print("{:<30} {:>15.4f} {:>15.4f} {:>15.4f}".format(
                metric_name,
                simple_mean,
                advanced_mean,
                diff
            ))
    
    # Print winner
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    simple_overall = simple_aggregates.get('Overall', {}).get('mean', 0)
    advanced_overall = advanced_aggregates.get('Overall', {}).get('mean', 0)
    
    if advanced_overall > simple_overall:
        winner = "🏆 ADVANCED RAG WINS"
        margin = advanced_overall - simple_overall
    else:
        winner = "🏆 SIMPLE RAG WINS"
        margin = simple_overall - advanced_overall
    
    print(f"\n{winner}")
    print(f"Margin: {margin:.4f} points")
    print(f"\nSimple RAG Average Score: {simple_overall:.4f}")
    print(f"Advanced RAG Average Score: {advanced_overall:.4f}")
    
    # Save detailed results
    results = {
        'evaluation_framework': 'DeepEval',
        'metrics_used': [getattr(m, 'name', m.__class__.__name__) for m in metrics],
        'simple_rag': {
            'aggregates': simple_aggregates,
            'num_questions': num_questions
        },
        'advanced_rag': {
            'aggregates': advanced_aggregates,
            'num_questions': num_questions
        },
        'winner': 'advanced_rag' if advanced_overall > simple_overall else 'simple_rag',
        'margin': abs(advanced_overall - simple_overall)
    }
    
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Detailed results saved to: evaluation_results.json")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()