from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import os, json
load_dotenv()

from gpt_evaluation import GPT_Evaluation
from data_extractor import DataExtractor
from constants import payload


df = pd.read_json("./df_expected_qa.json")
df.rename(columns={"question": "input_question", "answer": "output_answer"}, inplace=True)
df.to_csv("data/df_expected_qa.csv", index=False)


url = os.getenv("DF_URL")
print(url)
file_path = "df_context/car_price_dataset_1k.csv"
df_dataset_name = f"DF Dataset {os.path.basename(file_path)}"
df_data = pd.read_csv(file_path)
json_df = {
    "df_name": "car_price_dataset_1k",
    "df_description": "",
    "data": df_data.to_json()
}


payload["context"] = json.dumps([json_df])
evaluator = GPT_Evaluation(payload, url)


df = pd.read_csv(f"data/df_expected_qa.csv")
evaluator.store_data(df, df_dataset_name)


results = evaluator.evaluate("DF", df_dataset_name)
results.to_pandas().to_csv(f"results/eval-DF-{datetime.now().strftime('%Y%m%d')}.csv", index=False)
