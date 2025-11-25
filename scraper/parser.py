
# Google Jobs Parser
## Dependencies

from bs4 import BeautifulSoup

def parse_job_list(
        soup: BeautifulSoup,
        page: int
    ):

    items = soup.find_all(name="span", string="Learn more")
    parsed_list = list()
    base_url = "https://www.google.com/about/careers/applications/"

    for item in items:
        job_url = item.find_next(name="a").get("href")
        job_url = f"{base_url}{job_url}"

        print(f"Parsing Page: {page} | Item: {len(parsed_list)+1} | URL: {job_url}")
        item2 = item.find_parent(name="li")

        job_id = item2.select_one("div").attrs["jsdata"].split(";")[1]
        job_loc = item2.find(name="i", string="place").find_next(name="span").text
        job_name = item.find_parent(name="div").find_parent(name="div").find_parent(name="div").find(name="h3").text
        try:
            job_lvl = item2.find(name="i", string="bar_chart").find_next(name="span").text
        except (TypeError, AttributeError) as e:
            print(e)
            job_lvl = None

        try:
            job_lvl_desc = item2.find(name="i", string="bar_chart").find_next(name="span").find_next(name="div").find(name="h2").find_next("div").text
        except (TypeError, AttributeError) as e:
            print(e)
            job_lvl_desc = None

        job_qua = " ".join([
            f"- {i.text}" for i in item2.find(name="h4", string="Minimum qualifications").find_next(name="ul").select("li")
        ])

        data_dict = dict()
        data_dict["id"] = job_id
        data_dict["url"] = job_url
        data_dict["name"] = job_name
        data_dict["location"] = job_loc
        data_dict["level"] = job_lvl
        data_dict["level_description"] = job_lvl_desc
        data_dict["qualification"] = job_qua

        # Append parsed list
        parsed_list.append(data_dict)

    return parsed_list