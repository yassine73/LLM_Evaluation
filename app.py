import pandas as pd
import os
import sys
from dotenv import load_dotenv
import argparse
load_dotenv()

from gpt_evaluation import GPT_Evaluation, MetricType

arg_parser = argparse.ArgumentParser("GPT Evaluation")
arg_parser.add_argument("-d", "--dataframe", action="store_true", help="-f or --dataframe to Evaluate DataFrame")
arg_parser.add_argument("-p", "--plot_code", action="store_true", help="-p or --plot_code to Evaluate Plot Code")
arg_parser.add_argument("-s", "--sql", action="store_true", help="-s or --sql to Evaluate SQL")
args = arg_parser.parse_args()

dir = "./data"
os.makedirs(os.path.join(dir, "DF"), exist_ok=True)
os.makedirs(os.path.join(dir, "SQL"), exist_ok=True)

GPT = "DF" if args.dataframe else "SQL" if args.sql else None
PLOT_CODE = True if args.plot_code else False
gpt_evaluation = GPT_Evaluation()
if GPT == "DF":
    gpt_evaluation.url = os.getenv("DF_URL")
    for file_name in os.listdir(os.path.join(dir, GPT)):
        gpt_evaluation.type = GPT
        if file_name.endswith((".csv", ".xlsx")) and file_name == "Regional_Tourism_Indicators.csv":
            dataset_name = f"{GPT} {file_name}"
            try:
                if PLOT_CODE:
                    dataset_name = f"{GPT} Plot {file_name}"
                    json_qa_plot = os.path.join(dir, GPT, file_name.replace(".csv", "_plot.json").replace(".xlsx", "_plot.json"))
                    df_plot = pd.read_json(json_qa_plot)
                    df_plot = df_plot.rename(columns={"question": "input_question", "answer": "output_answer"})
                    gpt_evaluation.store_data(df_plot, dataset_name)
                    
                    if file_name.endswith(".csv"):
                        df = pd.read_csv(os.path.join(dir, GPT, file_name))
                    else:
                        df = pd.read_excel(os.path.join(dir, GPT, file_name))
                    gpt_evaluation.fill_context(df, file_name)
                    gpt_evaluation.evaluate(dataset_name, MetricType.PLOT.value)
                else:
                    json_qa = os.path.join(dir, GPT, file_name.replace(".csv", ".json").replace(".xlsx", ".json"))
                    df = pd.read_json(json_qa)
                    df = df.rename(columns={"question": "input_question", "answer": "output_answer"})
                    gpt_evaluation.store_data(df, dataset_name)
                    if file_name.endswith(".csv"):
                        df = pd.read_csv(os.path.join(dir, GPT, file_name))
                    else:
                        df = pd.read_excel(os.path.join(dir, GPT, file_name))
                    gpt_evaluation.fill_context(df, file_name)
                    gpt_evaluation.evaluate(dataset_name, MetricType.NORMAL.value)
            except Exception as e:
                raise e
        else:
            print(f"File {file_name} not recognized")
elif GPT == "SQL":
    URL = os.getenv("SQL_URL")
else:
    raise ValueError("GPT not recognized")