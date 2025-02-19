import pandas as pd
from psycopg2.extras import RealDictCursor
import psycopg2
import os
from pathlib import Path

DATA_DIR = Path('data')

class DataExtractor:
    def __init__(self):
        self.bot_sql_id = os.getenv('BOT_SQL_ID')
        self.bot_type = "SQL"
        self.db_params = {
            'dbname': os.getenv('POSTGRES_DB'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'host': os.getenv('POSTGRES_HOST'),
            'port': os.getenv('POSTGRES_PORT')
        }
        
    def get_db_connection(self):
        """Create and return a database connection"""
        try:
            conn = psycopg2.connect(**self.db_params)
            return conn
        except psycopg2.Error as e:
            raise e

    def fetch_conversations(self, bot_id: str) -> pd.DataFrame:
        query = (
        "SELECT "
            "ci.conversation_id, "
            "ci.interaction_id, "
            "ci.message, "
            "ci.message_type, "
            "ci.text_type, "
            "ci.context, "
            "ci.good_answer, "
            "ci.timestamp, "
            "cc.bot_id, "
            "cc.title "
        "FROM conversations_interaction ci "
        "JOIN conversations_conversation cc ON ci.conversation_id = cc.conversation_id "
        "WHERE cc.bot_id = %s "
        "ORDER BY ci.timestamp "
        )
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, (bot_id,))
                    results = cur.fetchall()
                    
            df = pd.DataFrame(results)
            return df
            
        except Exception as e:
            raise

    def fetch_and_store_data(self):
        if not self.bot_sql_id:
            raise ValueError("Bot ID not configured")

        try:
            df = self.fetch_conversations(self.bot_sql_id)
            bot_folder = DATA_DIR / self.bot_type
            bot_folder.mkdir(exist_ok=True)
            
            for conv_id, conv_df in df.groupby('conversation_id'):
                conv_df = conv_df.reset_index(drop=True)
                if conv_df.empty:
                    continue
                filename = f"{conv_id}.csv"
                output_file = bot_folder / filename
                if output_file.exists():
                    existing_df = pd.read_csv(output_file, index_col='id')
                    for idx, row in conv_df.iterrows():
                        interaction_id = row['interaction_id']
                        if interaction_id in existing_df['interaction_id'].values:
                            existing_row = existing_df.loc[existing_df['interaction_id'] == interaction_id].iloc[0]
                            conv_df.loc[idx, 'corrected_message'] = existing_row.get('corrected_message', '')
                            conv_df.loc[idx, 'corrected_context'] = existing_row.get('corrected_context', '')
                conv_df.to_csv(output_file, index_label='id')
            
        except Exception as e:
            raise e
