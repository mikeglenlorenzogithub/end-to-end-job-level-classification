
import pandas as pd

from datetime import datetime

import sys
from pathlib import Path

# Automatically detect the repo root (parent of notebook folder)
repo_root = Path().resolve().parent  # if notebook is in 'notebooks/' folder
sys.path.append(str(repo_root))

from config.config import data_import_pandas, data_export_pandas

# Function to Remerge all results
def concat_cleaning(
        website: str,
        content_date: datetime,
        version: int,
        borders: list[tuple]
    ):

    df_merge = pd.DataFrame()
    for lower_border, upper_border in borders:
        df_merge_temp = data_import_pandas(
            website=website,
            content_date=content_date,
            version=version,
            folder_name='cleaning',
            additional_info=f'cleaning-{lower_border}_{upper_border}',
        )

        df_merge = pd.concat([
            df_merge,
            df_merge_temp
        ])

    df_merge.reset_index(drop=True, inplace=True)

    data_export_pandas(
        df_output=df_merge,
        website=website,
        content_date=content_date,
        version=version,
        folder_name='cleaning',
        additional_info='cleaning',
    )

    return df_merge