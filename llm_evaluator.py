import json
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langfuse.callback import CallbackHandler

# Pydantic schema for LLM-as-a-judge response
class EvaluationResult(BaseModel):
    score: int = Field(description="A score between 1 and 5 indicating the quality of the response.")
    reasoning: str = Field(description="The reasoning behind the score.")

class LLMEvaluator:
    """
    A comprehensive suite for evaluating LLM outputs across multiple dimensions:
    1. Lexical (Exact Match, Inclusion)
    2. LLM-as-a-Judge (Qualitative scoring, hallucination detection)
    3. Custom heuristics
    """
    def __init__(self, judge_llm: Optional[BaseChatModel] = None):
        # Default judge to Google GenAI if provided, or expect it to be passed
        self.judge_llm = judge_llm 

    def evaluate_exact_match(self, prediction: str, reference: str) -> bool:
        """Evaluates whether the prediction exactly matches the reference."""
        return prediction.strip().lower() == reference.strip().lower()

    def evaluate_inclusion(self, prediction: str, reference: str) -> bool:
        """Evaluates whether the reference string is included in the prediction."""
        return reference.strip().lower() in prediction.strip().lower()
    
    def evaluate_with_judge(self, criteria: str, instruction: str, prediction: str, reference: Optional[str] = None) -> EvaluationResult:
        """
        Uses an LLM as a judge to evaluate a prediction based on criteria.
        Expects self.judge_llm to support structured output (e.g. Gemini 1.5 Pro).
        """
        if not self.judge_llm:
            raise ValueError("A judge LLM must be initialized to use 'evaluate_with_judge'.")
            
        system_prompt = (
            "You are an impartial expert evaluator. Evaluate the provided prediction based on the given criteria.\n"
            "Return a rigorous score from 1-5 and the reasoning."
        )
        
        user_content = f"Criteria: {criteria}\nInstruction: {instruction}\nPrediction: {prediction}\n"
        if reference:
            user_content += f"Reference: {reference}\n"
            
        # Enforce structured output via Pydantic
        structured_llm = self.judge_llm.with_structured_output(EvaluationResult)
        
        # Initialize Langfuse handler
        langfuse_handler = CallbackHandler()
        
        response = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ], config={"callbacks": [langfuse_handler]})
        
        return response

    def run_evaluation_suite(self, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Runs a suite of evaluations on a dataset.
        dataset format: [{'instruction': '...', 'prediction': '...', 'reference': '...'}, ...]
        """
        results = {
            "exact_match_accuracy": 0.0,
            "inclusion_accuracy": 0.0,
            "judge_average_score": 0.0,
            "details": []
        }
        
        total_score = 0
        valid_judge_runs = 0
        
        for data in dataset:
            prediction = data.get("prediction", "")
            reference = data.get("reference", "")
            instruction = data.get("instruction", "")
            
            em = self.evaluate_exact_match(prediction, reference)
            inc = self.evaluate_inclusion(prediction, reference)
            
            detail = {
                "instruction": instruction,
                "prediction": prediction,
                "reference": reference,
                "exact_match": em,
                "inclusion": inc
            }
            
            if self.judge_llm:
                try:
                    judge_res = self.evaluate_with_judge(
                        criteria="Is the prediction helpful, accurate, and faithful to the reference?",
                        instruction=instruction,
                        prediction=prediction,
                        reference=reference
                    )
                    detail["judge_score"] = judge_res.score
                    detail["judge_reasoning"] = judge_res.reasoning
                    total_score += judge_res.score
                    valid_judge_runs += 1
                except Exception as e:
                    detail["judge_error"] = str(e)
            
            results["details"].append(detail)
            
            if em: results["exact_match_accuracy"] += 1
            if inc: results["inclusion_accuracy"] += 1
            
        num_items = len(dataset)
        if num_items > 0:
            results["exact_match_accuracy"] /= num_items
            results["inclusion_accuracy"] /= num_items
        if valid_judge_runs > 0:
            results["judge_average_score"] = total_score / valid_judge_runs
            
        return results

if __name__ == "__main__":
    # Example usage:
    print("Initialize the evaluator... (Set your GOOGLE_API_KEY environment variable)")
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    # evaluator = LLMEvaluator(judge_llm=llm)
    # dataset = [
    #     {"instruction": "What is 2+2?", "prediction": "It is exactly 4.", "reference": "4"},
    #     {"instruction": "Capital of France?", "prediction": "London", "reference": "Paris"}
    # ]
    # metrics = evaluator.run_evaluation_suite(dataset)
    # print(json.dumps(metrics, indent=2))
