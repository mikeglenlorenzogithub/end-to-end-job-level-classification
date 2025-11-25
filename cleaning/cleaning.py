
import pandas as pd

from datetime import datetime, timedelta
from math import ceil
from google.genai import types

import sys
from pathlib import Path

# Automatically detect the repo root (parent of notebook folder)
repo_root = Path().resolve().parent  # if notebook is in 'notebooks/' folder
sys.path.append(str(repo_root))

from config.config import get_environment

from config.config import data_import_json, data_export_json, data_import_pandas, data_export_pandas
from cleaning.llm import gemini_process_response
from cleaning.concat import concat_cleaning
from cleaning.mapping import merge_mapping


# Preprocess Input (insert content date etc)
def llm_preprocess_input(
        df_input: pd.DataFrame,
        content_date: datetime,
        version: str
    ):

    # Assign content_date
    df_input['content_date'] = str(content_date)

    # Generate UID
    df_input['uid'] = df_input.index + 1
    df_input['uid'] = df_input['uid'].astype(str)
    df_input['version'] = version

    return

# Convert DataFrame to List of Input
def llm_generate_input(
        df_input: pd.DataFrame,
        process_column: list
    ):

    # Convert input to list
    input_data = df_input[['uid'] + process_column].apply(dict, axis=1).to_list()

    return input_data

# Feed Converted List Input to Gemini as str and return structured output
def llm_request_gemini(
        prompt: str,
        FIELD_DEFS: list,
        input_data: list,
        MODEL: str,
        API_KEY: str
    ):

    response = gemini_process_response(
        model_version=MODEL,
        api_key=API_KEY,
        prompt_template=prompt,
        input=str(input_data),
        column_uid='uid',
        response_key_list=FIELD_DEFS
    )

    return response

# Dump Structured Response to json
def llm_dump_response(
        response: types.GenerateContentResponse,
        content_date: datetime,
        website: str,
        version: str,
        additional_info: str='response'
    ):

    # Convert to Python dict safely
    response_dict = response.model_dump()

    # Optionally save to file
    data_export_json(
        data=response_dict,
        website=website,
        folder_name='llm',
        version=version,
        content_date=content_date, # "0000-00-00"
        additional_info=additional_info
    )

# Merge response to input dataframe and export 
def llm_merge_response(
        df_input: pd.DataFrame,
        response: types.GenerateContentResponse,
        website: str,
        content_date: datetime,
        version: str,
        folder_name: str,
        additional_info: str='response-parsed'
    ):

    # Get token usage
    token_input = response.usage_metadata.prompt_token_count
    token_output = response.usage_metadata.candidates_token_count
    print(f"{additional_info} | Token Input: {token_input} | Token Output: {token_output}")

    # Convert parsed response to dataframe
    df_response = pd.DataFrame(response.parsed)
    df_response.insert(0, "token_input", token_input)
    df_response.insert(1, "token_output", token_output)

    # # Dump converted response to json
    # data_export_pandas(
    #     df_output=df_response,
    #     website=website,
    #     content_date=content_date,
    #     version=version,
    #     folder_name='llm',
    #     additional_info=additional_info,
    #     # incl_excel=True
    # )

    # Merge response
    df_input_merged = pd.merge(
        left=df_input,
        right=df_response,
        on='uid',
        how='left'
    )

    # Export Merged Gemini Response with Input Data
    data_export_pandas(
        df_output=df_input_merged,
        website=website,
        content_date=content_date,
        version=version,
        folder_name=folder_name,
        additional_info=additional_info
    )

    return df_input_merged

# Function to Process Qualification Cleaning
def cleaning_qualification(
        df_input: pd.DataFrame,
        website: str,
        content_date: datetime,
        version: str,
        prompt: str,
        FIELD_DEFS: list,
        MODEL: str,
        API_KEY: str,
        process_column: list=['qualification'],
        additional_info: str='cleaning',
    ):

    print("Preprocessing Input Data")
    try:
        llm_preprocess_input(
            df_input=df_input,
            content_date=content_date,
            version=version
        )

    except Exception as e:
        raise(e)

    print("Generating Input List")
    try:
        input_data = llm_generate_input(
            df_input=df_input,
            process_column=process_column
        )

    except Exception as e:
        raise(e)

    print("Request Gemini Response")
    try:
        response = llm_request_gemini(
            prompt=prompt,
            FIELD_DEFS=FIELD_DEFS,
            input_data=input_data,
            MODEL=MODEL,
            API_KEY=API_KEY
        )

    except Exception as e:
        raise(e)

    print("Dump Gemini Response to JSON")
    try:
        llm_dump_response(
            response=response,
            content_date=content_date,
            website=website,
            version=version,
            additional_info=f'response-{additional_info}'
        )

    except Exception as e:
        raise(e)

    print("Merge Gemini Response to Input Data and Export")
    try:
        df_input_merge = llm_merge_response(
            df_input=df_input,
            response=response,
            website=website,
            content_date=content_date,
            version=version,
            folder_name='cleaning',
            additional_info=additional_info
        )

    except Exception as e:
        raise(e)

    return df_input_merge

# Get country from location
def get_country(
        df_input: pd.DataFrame
    ):

    df_input['country'] = df_input['location'].str.split(',').str[-1].str.strip()

    return


def run():
    ENV = get_environment(
        env_path="../environments",
        env_name="env.json"
    )

    # content_date = datetime.now().date() + timedelta(days=0)
    content_date = ENV['CONTENT_DATE']
    website = ENV['SOURCE']['NAME']
    version = ENV['VERSION']

    MODEL = ENV['LLM']['GEMINI']['MODEL']
    API_KEY = ENV['LLM']['GEMINI']['API_KEY']
    range_input = ENV['LLM']['GEMINI']['RANGE_INPUT']

    concat_only = ENV['CLEANING']['CONCAT_ONLY']

    if concat_only:
        # Concat all cleaning results
        df_merge = concat_cleaning(
            website=website,
            content_date=content_date,
            version=version,
            borders=borders
        )

    else:
        df_input = data_import_pandas(
            website=website,
            content_date=content_date,
            version=version,
            folder_name='parser',
            additional_info='parsed',
        )

        get_country(
            df_input=df_input
        )

        prompt = f"""
        You are a data cleaning and normalization assistant.

        Your task is to take a messy text containing pricing information and convert it into a clean, structured JSON object. 
        Make sure to:
        - Extract the minimum years of experience as integers (numbers only, without dots or commas).
        - Extract the minimum required degree.
        - Extract relevant experience role, e.g software development, ML infrastructure.
        - Fill empty value with empty string.

        Return output **strictly** in JSON with these fields:
        `uid`, `y` as `min_years`, `d` as `min_degree`, `e` as `relevant_experience`.

        Example:

        Input: 
        {{
            "uid": "1",
            "qualification": "- Bachelor’s degree or equivalent practical experience. - 2 years of experience with software development in one or more programming languages, or 1 year of experience with an advanced degree. - 1 year of experience with one or more of the following: speech/audio (e.g., technology duplicating and responding to the human voice), reinforcement learning (e.g., sequential decision making), or specialization in another ML field. - 1 year of experience with ML infrastructure (e.g., model deployment, model evaluation, optimization, data processing, debugging)."
        }}

        Output:
        {{
            "uid": "1",
            "y": "1",
            "d": "bachelor",
            "x": "software development, ML field"
        }}

        Now clean the following input:

        """

        FIELD_DEFS = [
            ("r", [
                ("y", "string"),
                ("d", "string"),
                ("x", "string"),
            ]),
        ]

        # Create looping partitions
        len_input = len(df_input)
        loop_count = ceil(len_input/range_input)

        borders = [(loop*range_input, (loop+1)*range_input) for loop in range(loop_count)][:-1]
        borders = borders + [((loop_count-1)*range_input, len_input)]

        df_merge = pd.DataFrame()

        # Loop for each partitions based on the range input
        for lower_border, upper_border in borders:

            print(f"Processing: {lower_border} - {upper_border}")
            df_input_temp = df_input.iloc[lower_border:upper_border].copy(deep=True)

            df_merge_temp = cleaning_qualification(
                df_input=df_input_temp,
                website=website,
                content_date=content_date,
                version=version,
                prompt=prompt,
                FIELD_DEFS=FIELD_DEFS,
                MODEL=MODEL,
                API_KEY=API_KEY,
                additional_info=f'cleaning-{lower_border}_{upper_border}'
            )
            print(f"Completed: {lower_border} - {upper_border}")

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

    # Mapping response information 
    df_merge = merge_mapping(
        website=website,
        content_date=content_date,
        version=version
    )

    return df_merge