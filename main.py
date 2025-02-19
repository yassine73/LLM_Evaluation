from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import os
import argparse
load_dotenv()

from gpt_evaluation import GPT_Evaluation
from data_extractor import DataExtractor

def parse_arguments():
    parser = argparse.ArgumentParser(description='Bot Answer Validation Test Suite')
    parser.add_argument('-f', '--fetch-data', action='store_true', help='Fetch data from database')
    return parser.parse_args()

def main():
    args = parse_arguments()
    data_extractor = DataExtractor()
    
    payload = {
        "query": "hi",
        "chat_history": [],
        "context": os.getenv("CONTEXT_SQL"),
        "index_name": "",
        "generate_conversation_title": False,
        "user_info": {}
    }
    url = os.getenv("SQL_URL")
    sql_dataset_name = "SQL Dataset"
    evaluator = GPT_Evaluation(payload, url)
    
    if args.fetch_data:
        # data_extractor.fetch_and_store_data()
        # directory = "data/SQL"
        # for excel_file in os.listdir(directory):
        #     if excel_file.endswith(".csv"):
        df = pd.read_csv(f"data/sql_data.csv")
        evaluator.store_data(df, sql_dataset_name)
    
    results = evaluator.evaluate("SQL", sql_dataset_name)
    results.to_pandas().to_csv(f"results/eval-SQL-{datetime.now().strftime('%Y%m%d')}.csv", index=False)

if __name__ == "__main__":
    main()