from langchain_community.llms import Replicate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from datetime import datetime
import pandas as pd
import requests
import json
import os

from constants import PAYLOAD

class GPT_Evaluation:
    def __init__(self):
        self.client = Client()
        self.payload = PAYLOAD
        self.url = ""
        self.type = ""
        self.llm = Replicate(
            model="meta/meta-llama-3.1-405b-instruct",
            model_kwargs=dict(temperature=0),
        )
    
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
        return result.get("text")

    def final_answer_eval(self, inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        eval_instructions = "You are an expert professor specialized in grading student's answers to questions."
        user_content = (
            "You are grading the following question:\n"
            f"{inputs['question']}\n"
            "Here is the real answer:\n"
            f"{reference_outputs['answer']}\n"
            "You are grading the following predicted answer:\n"
            f"{outputs['response']}\n"
            "Respond Only with CORRECT or INCORRECT, No additional texts or comments.\n"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", eval_instructions),
            ("user", user_content)
        ])

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({})
        return response == "CORRECT"
    
    def target(self, inputs: str) -> dict:
        return {"response": self.generate_new_answer(inputs["question"])}
    
    def evaluate(self, dataset_name: str, print_results: bool = True):
        os.makedirs("results", exist_ok=True)
        results = self.client.evaluate(
            self.target,
            data=dataset_name,
            evaluators=[
                self.final_answer_eval
            ],
            experiment_prefix=f"eval-{self.type}-{datetime.now().strftime('%Y%m%d')}"
        )
        if print_results:
            results.to_pandas().to_csv(f"results/eval-{dataset_name}-{datetime.now().strftime('%Y%m%d')}.csv", index=False)