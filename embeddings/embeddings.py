import pandas as pd

from datetime import datetime, timedelta
import time

from tqdm import tqdm
from openai import OpenAI

import sys
from pathlib import Path

# Automatically detect the repo root (parent of notebook folder)
repo_root = Path().resolve().parent  # if notebook is in 'notebooks/' folder
sys.path.append(str(repo_root))

from config.config import get_environment

from config.config import data_import_pandas, data_export_pandas

# Function to filter non empty target (level) and preprocess
def filter_input(
        df_input: pd.DataFrame
    ):

    df_embed = df_input.copy(deep=True)
    df_embed = df_embed[df_embed['level'].notna()]
    df_embed = df_embed[df_embed['response_cleaning'].astype(str) != '[]']
    df_embed = df_embed[~df_embed['qualification'].str.contains('linkcopy link')]
    df_embed = df_embed[df_embed['qualification'].fillna('').astype(str) != '']
    df_embed = df_embed[df_embed['qualification'].apply(len) > 50].reset_index(drop=True)

    return df_embed


def run():
    ENV = get_environment(
        env_path="../environments",
        env_name="env.json"
    )

    # content_date = datetime.now().date() + timedelta(days=0)
    content_date = ENV['CONTENT_DATE']
    website = ENV['SOURCE']['NAME']
    version = ENV['VERSION']

    EMB_MODEL = ENV['EMBEDDING']['OPENAI']['MODEL']
    EMB_API_KEY = ENV['EMBEDDING']['OPENAI']['API_KEY']


    df_input = data_import_pandas(
        website=website,
        content_date=content_date,
        version=version,
        folder_name='cleaning',
        additional_info='mapping'
    )

    # Filter non empty target (level) and preprocess
    df_embed = filter_input(
        df_input=df_input
    )

    # Embeddings for qualification (batched)

    def get_embeddings_batch(
            input_list: list,
            api_key: str,
            model: str="text-embedding-3-small",
            range_input: int=50,
            sleep_sec: float=0.5
        ):

        client = OpenAI(api_key=api_key)

        embeddings = []
        for i in tqdm(range(0, len(input_list), range_input), desc="Embedding"):
            batch_texts = input_list[i:i+range_input]

            retry = 1
            while True:
                try:            
                    response = client.embeddings.create(
                        model=model,
                        input=batch_texts
                    )

                    # response.usage.prompt_tokens
                    # response.usage.total_tokens

                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    if retry > 3:
                        raise e
                    else:
                        retry += 1
                        print(e)

                time.sleep(sleep_sec)
        return embeddings

    X_qual_embed = get_embeddings_batch(
        input_list=df_embed['qualification'].tolist(),
        api_key=EMB_API_KEY,
        model=EMB_MODEL
    )
    df_embed['qualification_embedding'] = [list(vec) for vec in X_qual_embed]

    data_export_pandas(
        df_output=df_embed,
        website=website,
        content_date=content_date,
        version=version,
        folder_name='embeddings',
        additional_info='embeddings',
    )

    return df_embed