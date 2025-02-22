from langchain_community.llms import Replicate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langsmith import Client
from langchain_openai import ChatOpenAI
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import pandas as pd
import requests
import json
import os

from constants import PAYLOAD

class MetricType(str, Enum):
    NORMAL = "NORMAL"
    PLOT = "PLOT"

class PlotCodeEval(BaseModel):
    Comparison: bool = Field(..., description="Whether they produce the same output or not")
    Reasoning: str = Field(..., description="Reasoning for the comparison")

class GPT_Evaluation:
    def __init__(self):
        self.client = Client()
        self.payload = PAYLOAD
        self.url = ""
        self.type = ""
        self.llm_openai = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        self.llm_llama = Replicate(
            model="meta/meta-llama-3.1-405b-instruct",
            model_kwargs=dict(temperature=0),
        )
        
        self.model = self.llm_llama
    
    def fill_context(self, df_data: pd.DataFrame = None, file_name = None) -> str:
        if self.type == "DF":
            self.payload['context'] = json.dumps([{
                "df_name": file_name,
                "df_description": "",
                "data": df_data.to_json()
            }])
        elif self.type == "SQL":
            self.payload['context'] = os.getenv("CONTEXT_SQL")
        else:
            raise ValueError("GPT not recognized")
    
    def store_data(self, df: pd.DataFrame, dataset_name):
        try:
            dataset_id = self.client.create_dataset(dataset_name=dataset_name).id
        except:
            print("dataset already exists")
            dataset_id = [data.id for data in self.client.list_datasets(dataset_name=dataset_name)][0]
        
        stored_questions = []
        for example in self.client.list_examples(dataset_id=dataset_id):
            stored_questions.append(example.inputs)
        
        expected_inputs = []
        expected_outputs = []
        for _, row in df.fillna('').iterrows():
            question = row['input_question']
            answer = row['output_answer']
            if question not in [obj['question'] for obj in stored_questions + expected_inputs]:
                expected_inputs.append({'question': question})
                expected_outputs.append({'answer': answer})

        if expected_inputs and expected_outputs:
            self.client.create_examples(
                inputs=expected_inputs,
                outputs=expected_outputs,
                dataset_id=dataset_id
            )
        else:
            print("No new data to store")
    
    def generate_new_answer(self, question: str) -> str:
        self.payload["query"] = question
        result = requests.post(self.url + "/chat", json=self.payload).json()
        return {
            "answer": result.get("text"),
            "plot_code": json.loads(result.get("context")[0])['plot_code']
        }

    def final_answer_eval(self, inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        question = inputs['question'].replace("{", "{{").replace("}", "}}")
        real_answer = reference_outputs['answer'].replace("{", "{{").replace("}", "}}")
        generated_answer = outputs['response'].replace("{", "{{").replace("}", "}}")
        
        eval_instructions = "You are an expert professor specialized in grading student's answers to questions."
        user_content = (
            "You are grading the following question:\n"
            f"{question}\n"
            "Here is the real answer:\n"
            f"{real_answer}\n"
            "You are grading the following predicted answer:\n"
            f"{generated_answer}\n"
            "Respond Only with CORRECT or INCORRECT, No additional texts or comments.\n"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", eval_instructions),
            ("user", user_content)
        ])

        chain = prompt | self.model | StrOutputParser()
        response = chain.invoke({})
        return response == "CORRECT"
    
    def plot_code_eval(self, inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        question = inputs['question']
        real_answer = reference_outputs['answer'].replace("{", "{{").replace("}", "}}")
        generated_answer = outputs['response'].replace("{", "{{").replace("}", "}}")
        
        eval_instructions = "You are an expert Python programmer and code evaluator. Your task is to compare two pieces of Python code and determine whether they produce the same output and check if they do the same Conditional Filtering.\n"
        user_content = (
            "[BEGIN DATA]\n************\n"
            f"[Question]: {question}\n"
            "************\n"
            f"[Real Answer]: {real_answer}\n"
            "************\n"
            f"[Generated Answer]: {generated_answer}\n"
            "[END DATA]\n"
            "Respond Only with CORRECT or INCORRECT, No additional texts or comments."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", eval_instructions),
            ("user", user_content)
        ])

        chain = prompt | self.model | StrOutputParser()
        response = chain.invoke({})
        return response == "CORRECT"
    
    def normal_target(self, inputs: str) -> dict:
        answer = self.generate_new_answer(inputs["question"])['answer']
        return {"response": answer}
    
    def plot_target(self, inputs: str) -> dict:
        plot_code = self.generate_new_answer(inputs["question"])['plot_code']
        return {"response": plot_code}
    
    def evaluate(self, dataset_name: str, type: MetricType = MetricType.NORMAL, print_results: bool = True):
        os.makedirs("results", exist_ok=True)
        
        result = None
        if type == MetricType.NORMAL.value:
            results = self.client.evaluate(
                self.normal_target,
                data=dataset_name,
                evaluators=[self.final_answer_eval],
                experiment_prefix=f"Normal-eval-{self.type}-{datetime.now().strftime('%Y%m%d')}"
            )
        elif type == MetricType.PLOT.value:
            results = self.client.evaluate(
                self.plot_target,
                data=dataset_name,
                evaluators=[self.plot_code_eval],
                experiment_prefix=f"Plot-eval-{self.type}-{datetime.now().strftime('%Y%m%d')}"
            )
        else:
            raise ValueError("Metric type not recognized")

        if print_results and results:
            results.to_pandas().to_csv(f"results/eval-{dataset_name}-{datetime.now().strftime('%Y%m%d')}.csv", index=False)