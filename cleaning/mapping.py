

import pandas as pd

from datetime import datetime

import sys
from pathlib import Path

# Automatically detect the repo root (parent of notebook folder)
repo_root = Path().resolve().parent  # if notebook is in 'notebooks/' folder
sys.path.append(str(repo_root))

from config.config import data_import_pandas, data_export_pandas


# Function to explode and mapping response
def merge_mapping(
        website: str,
        content_date: datetime,
        version: int
    ):

    df_merge = data_import_pandas(
        website=website,
        content_date=content_date,
        version=version,
        folder_name='cleaning',
        additional_info='cleaning',
    )

    df_merge = pd.concat([
        df_merge,
        pd.json_normalize(df_merge['r'].str[0])
    ], axis=1)

    df_merge.rename(columns={
        "r": "response_cleaning",
        "y": "min_years",
        "d": "min_degree",
        "x": "related_experience",
    }, inplace=True)

    cleaning_cols = ['level_description', 'qualification', 'country', 'min_years', 'min_degree', 'related_experience']
    for column in cleaning_cols:
        df_merge[column] = df_merge[column].fillna('').astype(str).str.lower().str.strip().str.replace(f"^(null|none|nan|na|n/a)$", "", regex=True)

    data_export_pandas(
        df_output=df_merge,
        website=website,
        content_date=content_date,
        version=version,
        folder_name='cleaning',
        additional_info='mapping',
    )

    return df_merge