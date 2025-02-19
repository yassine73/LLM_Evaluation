import pandas as pd
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from gpt_evaluation import GPT_Evaluation

dir = "./data"
GPT = str(sys.argv[1]).upper()

os.makedirs(os.path.join(dir, "DF"), exist_ok=True)
os.makedirs(os.path.join(dir, "SQL"), exist_ok=True)

gpt_evaluation = GPT_Evaluation()
if GPT == "DF":
    gpt_evaluation.url = os.getenv("DF_URL")
    for file_name in os.listdir(os.path.join(dir, GPT)):
        gpt_evaluation.type = GPT
        if file_name.endswith((".csv", ".xlsx")):
            try:
                json_qa = os.path.join(dir, GPT, file_name.replace(".csv", ".json").replace(".xlsx", ".json"))
                print(json_qa)
                df = pd.read_json(json_qa)
                df = df.rename(columns={"question": "input_question", "answer": "output_answer"})
                gpt_evaluation.store_data(df, f"{GPT} Dataset {file_name}")
                if file_name.endswith(".csv"):
                    df = pd.read_csv(os.path.join(dir, GPT, file_name))
                else:
                    df = pd.read_excel(os.path.join(dir, GPT, file_name))
                gpt_evaluation.fill_context(df, file_name)
                gpt_evaluation.evaluate(f"{GPT} Dataset {file_name}")
            except Exception as e:
                raise e
        else:
            print(f"File {file_name} not recognized")
elif GPT == "SQL":
    URL = os.getenv("SQL_URL")
else:
    raise ValueError("GPT not recognized")
        
        