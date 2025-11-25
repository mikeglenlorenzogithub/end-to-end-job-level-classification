# Import Dependencies
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from pathlib import Path

import joblib
import pandas as pd
import sys


# Local ENV Dependencies
import json

# ============================================================== READ LOCAL ENVIRONMENTS ==============================================================>
def get_environment(env_path: str, env_name: str):
    env_complete_path = Path(f"{env_path}/{env_name}")

    with open(env_complete_path, 'r') as f:

        ENV = json.load(f)

    return ENV


# Function to convert scrape date iso into basic
def date_basic(
        content_date: datetime
    ):

    return str(content_date).replace('-', '')


# Function to handle filename as standardized filename
def naming_file(
        website: str,
        content_date: datetime,
        version: str,
        file_extension: str,
        additional_info: str=None
    ):

    if additional_info:
        return f"{date_basic(content_date=content_date)}-{website}-{additional_info}-v{version}.{file_extension}"
    else:
        return f"{date_basic(content_date=content_date)}-{website}-v{version}.{file_extension}"


# Function to generate standardized folder path
def generate_folder_path(
        content_date: datetime,
        folder_name: str,
        base_folder_path: str='../data'
    ):

    return f"{base_folder_path}/{folder_name}/{date_basic(content_date)}"


# Function to ensure folder path before import/export file
def ensure_folder(
        path: str
    ):

    folder_path = Path(path)

    if bool(folder_path.suffixes):
        folder_path = folder_path.parent

    if folder_path.exists():
        print(f"Folder path exist: {folder_path}")
    else:
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Folder path doesn't exist, created: {folder_path}")


# Function to read dumped pandas dataframe files
def data_import_pandas(
        website: str,
        content_date: datetime,
        version: str,
        folder_name: str,
        additional_info: str=None,
        file_extension: str='json',
        base_folder_path: str='../data'
    ):

    folder_path = generate_folder_path(
        content_date=content_date,
        folder_name=folder_name,
        base_folder_path=base_folder_path
    )
    file_name = naming_file(
        website=website,
        content_date=content_date,
        version=version,
        file_extension=file_extension,
        additional_info=additional_info
    )
    file_path = f"{folder_path}/{file_name}"

    try:
        # Ensure folder path
        ensure_folder(folder_path)

        # Import from limited file extension
        if file_extension in ['json']:
            df_input = pd.read_json(file_path)

        elif file_extension in ['xlsx', 'xls']:
            df_input = pd.read_excel(file_path)
    
        else:
            raise ImportError(f'File extension out of options: {file_extension}')

        print(f"Data Import Pandas Success: {file_path}")

    except Exception as e:
        print(f"Data Import Pandas Failed: {file_path}")
        raise e

    return df_input


# Function to dump pandas dataframe
def data_export_pandas(
        df_output: pd.DataFrame,
        website: str,
        content_date: datetime,
        version: str,
        folder_name: str,
        additional_info: str=None,
        incl_excel: bool=False,
        file_extension: str='json',
        base_folder_path: str='../data'
    ):

    folder_path = generate_folder_path(
        content_date=content_date,
        folder_name=folder_name,
        base_folder_path=base_folder_path
    )
    file_name = naming_file(
        website=website,
        content_date=content_date,
        version=version,
        file_extension=file_extension,
        additional_info=additional_info
    )
    file_path = f"{folder_path}/{file_name}"

    try:
        # Ensure folder path
        ensure_folder(folder_path)

        # Export from limited file extension
        if file_extension in ['json']:
            df_output.to_json(file_path, index=False)
            if incl_excel:
                df_output.to_excel(file_path.replace('.json', '.xlsx'), index=False)

        elif file_extension in ['xlsx', 'xls']:
            df_output.to_excel(file_path, index=False)
    
        else:
            raise ImportError(f'File extension out of options: {file_extension}')

        print(f"Data Export Pandas Success: {file_path}")

    except Exception as e:
        print(f"Data Export Pandas Failed: {file_path}")
        raise e

# Function to dump data to JSON
def data_export_json(
        data: Any,
        website: str,
        folder_name: str,
        version: int,
        content_date: datetime=None, # "0000-00-00"
        additional_info: str=None,
        metadata: dict=None,
        base_folder_path: str='../data'
    ):
    """
    Saves data to JSON with metadata
    """

    if not content_date:
        content_date = datetime.now().date()

    folder_path = generate_folder_path(
        content_date=content_date,
        folder_name=folder_name,
        base_folder_path=base_folder_path
    )
    file_name = naming_file(
        website=website,
        content_date=content_date,
        version=version,
        file_extension='json',
        additional_info=additional_info
    )
    file_path = f"{folder_path}/{file_name}"

    # Create payload to export
    if metadata:
        payload = metadata
        payload["content_date"] = str(content_date)
        payload["website"] = website
        payload["additional_info"] = additional_info
        payload["data"] = data

    else:
        payload = {
            "content_date": str(content_date),
            "website": website,
            "additional_info": additional_info,
            "data": data
        }

    # Ensure folder path
    ensure_folder(folder_path)

    # Export json data
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except TypeError as e:
        print(e, "=====> Converting data")
        payload["data"] = str(data)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return file_path


# Function to read dumped JSON files
def data_import_json(
        website: str,
        folder_name: str,
        version: int,
        content_date: datetime=None, # "0000-00-00"
        additional_info: str=None,
        base_folder_path: str='../data'
    ):
    """
    Reads a JSON file and returns the content
    """

    if not content_date:
        content_date = datetime.now().date()

    folder_path = generate_folder_path(
        content_date=content_date,
        folder_name=folder_name,
        base_folder_path=base_folder_path
    )
    file_name = naming_file(
        website=website,
        content_date=content_date,
        version=version,
        file_extension='json',
        additional_info=additional_info
    )
    file_path = f"{folder_path}/{file_name}"

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Function to dump data to pickle
def data_export_pkl(
        pipeline_objects: Any,
        website: str,
        folder_name: str,
        version: int,
        content_date: datetime=None, # "0000-00-00"
        additional_info: str=None,
        base_folder_path: str='../data'
    ):
    """
    Saves data to pickle
    """

    if not content_date:
        content_date = datetime.now().date()

    folder_path = generate_folder_path(
        content_date=content_date,
        folder_name=folder_name,
        base_folder_path=base_folder_path
    )
    file_name = naming_file(
        website=website,
        content_date=content_date,
        version=version,
        file_extension='pkl',
        additional_info=additional_info
    )
    file_path = f"{folder_path}/{file_name}"

    # Ensure folder path
    ensure_folder(folder_path)

    # Export pickle data
    try:
        joblib.dump(pipeline_objects, file_path)
    except Exception as e:
        raise e

    return file_path

# Function to read dumped data from pickle
def data_import_pkl(
        website: str,
        folder_name: str,
        version: int,
        content_date: datetime=None, # "0000-00-00"
        additional_info: str=None,
        base_folder_path: str='../data'
    ):
    """
    Import data from pkl
    """

    if not content_date:
        content_date = datetime.now().date()

    folder_path = generate_folder_path(
        content_date=content_date,
        folder_name=folder_name,
        base_folder_path=base_folder_path
    )
    file_name = naming_file(
        website=website,
        content_date=content_date,
        version=version,
        file_extension='pkl',
        additional_info=additional_info
    )
    file_path = f"{folder_path}/{file_name}"

    # Ensure folder path
    ensure_folder(folder_path)

    # Import pickle data
    try:
        pipeline_objects = joblib.load(file_path)
    except Exception as e:
        raise e

    return pipeline_objects