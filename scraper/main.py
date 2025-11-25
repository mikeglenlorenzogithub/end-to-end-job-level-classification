# Google Jobs Scraper
## Dependencies
import requests

import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Automatically detect the repo root (parent of notebook folder)
repo_root = Path().resolve().parent  # if notebook is in 'notebooks/' folder
sys.path.append(str(repo_root))

from config.config import get_environment
from config.config import data_import_json, data_export_pandas
from scraper import scrape_job
from parser import parse_job_list

def run():
    ## ENV
    ENV = get_environment(
        env_path="../environments",
        env_name="env.json"
    )

    # content_date = datetime.now().date() + timedelta(days=0)
    content_date = ENV['CONTENT_DATE']
    website = ENV['SOURCE']['NAME']
    version = ENV['VERSION']

    ## Execute
    # Scrape and Parse
    reparse_only = ENV['SOURCE']['REPARSE_ONLY']
    continue_scraper = ENV['SOURCE']['CONTINUE_SCRAPER']
    session = requests.Session()
    parsed_list = list()
    page = 0

    try:
        while True:
            page += 1

            # Reparse only, no scrape | Continue scraper from latest page
            if reparse_only or continue_scraper:
                try:
                    json_data = data_import_json(
                        website=website,
                        folder_name='scraper',
                        version=version,
                        content_date=content_date,
                        additional_info=f"scrape-page{page}"
                    )
                    soup = BeautifulSoup(
                        json_data["data"],
                        "html.parser"
                    )

                except Exception as e:
                    # Scrape
                    if reparse_only:
                        print(e)
                        break

                    # Continue Scraper
                    else:
                        soup = scrape_job(
                            session=session,
                            page=page,
                            scrape_date=content_date,
                            website=website,
                            version=version,
                        )

            else:
                # Scrape
                soup = scrape_job(
                    session=session,
                    page=page,
                    scrape_date=content_date,
                    website=website,
                    version=version,
                )

            # Parse
            parsed_list_temp = parse_job_list(
                soup=soup,
                page=page
            )

            parsed_list = parsed_list + parsed_list_temp

            # Identifier to stop iteration
            if len(soup.select('a[aria-label="Go to next page"]')) == 0:
                break

            # if page == 2:
            #     break

        df_parse = pd.DataFrame(parsed_list)
        data_export_pandas(
            df_output=df_parse,
            website=website,
            content_date=content_date,
            version=version,
            folder_name='parser',
            additional_info='parsed',
            incl_excel=True
        )

    except Exception as e:
        print(e)

        df_parse = pd.DataFrame(parsed_list)
        data_export_pandas(
            df_output=df_parse,
            website=website,
            content_date=content_date,
            version=version,
            folder_name='parser',
            additional_info='parsed',
            incl_excel=True
        )

    return