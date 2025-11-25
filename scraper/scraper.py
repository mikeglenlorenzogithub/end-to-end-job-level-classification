
# Google Jobs Scraper
## Dependencies
import requests
import time
import random

from bs4 import BeautifulSoup
from datetime import datetime

import sys
from pathlib import Path

# Automatically detect the repo root (parent of notebook folder)
repo_root = Path().resolve().parent  # if notebook is in 'notebooks/' folder
sys.path.append(str(repo_root))

from config.config import data_export_json

## Mining
def scrape_job(
        scrape_date: datetime,
        website: str,
        version: int,
        session: requests,
        page: int,
        max_retry: int=3,
        time_sleep_min: int=1,
        time_sleep_max: int=3,
        timeout: int=10
    ):

    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9,id;q=0.8',
        'cache-control': 'max-age=0',
        'downlink': '1.35',
        'priority': 'u=0, i',
        'rtt': '300',
        'sec-ch-prefers-color-scheme': 'dark',
        'sec-ch-ua': '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
        'sec-ch-ua-arch': '"x86"',
        'sec-ch-ua-bitness': '"64"',
        'sec-ch-ua-form-factors': '"Desktop"',
        'sec-ch-ua-full-version': '"142.0.7444.60"',
        'sec-ch-ua-full-version-list': '"Chromium";v="142.0.7444.60", "Google Chrome";v="142.0.7444.60", "Not_A Brand";v="99.0.0.0"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-model': '""',
        'sec-ch-ua-platform': '"Windows"',
        'sec-ch-ua-platform-version': '"19.0.0"',
        'sec-ch-ua-wow64': '?0',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
        'x-browser-channel': 'stable',
    }

    url_scrape = f'https://www.google.com/about/careers/applications/jobs/results?page={page}'

    print(f"Scraping Page: {page} | URL: {url_scrape}")

    for retry in range(max_retry):
        time.sleep(random.uniform(time_sleep_min, time_sleep_max))
        try:
            response = session.get(
                url_scrape,
                headers=headers,
                timeout=timeout,
            )
            # Validate Status Code
            if response.status_code == 200:
                break
            else:
                raise ValueError(f"Status Code: {response.status_code} | URL: {url_scrape}")

        except Exception as e:
            if retry == max(range(max_retry)):
                raise e
            else:
                print(f"Retry request: {url_scrape} | Retry count: {retry+1} | Error: {e}")
                session = requests.Session()

    soup = BeautifulSoup(response.content, "html.parser")
    soup2 = soup.find(name="h2", string="Jobs search results").find_parent("div")
    total_jobs = soup2.select_one('div[role="status"]').get('aria-label')

    data_export_json(
        data=soup2,
        website=website,
        folder_name='scraper',
        version=version,
        content_date=scrape_date, # "0000-00-00"
        additional_info=f"scrape-page{page}",
        metadata={
            "total_jobs": total_jobs,
            "page": page,
            "url_scrape": url_scrape
        }
    )

    return soup2