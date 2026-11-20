from ddgs import DDGS
import requests
import trafilatura
import pandas as pd
import time

headers = {
        "User-Agent": "ResearchBot/1.0 (educational research)"
}

def search_web(query):
    # Maximum 10 try to fetch any web data
    max_retry = 10
    count = 1
    
    # Fetch using DuckDuckGo web crawler. Not for prodcution use.
    while count <=max_retry:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results = 10
                ))
                break
        except Exception as e:
            print(f"Search faiiled at attempt : {count}.\n Error : {e}")
        count+=1
    print(f"Found {len(results)} search results")
    
    # store the data
    research_data = []
    
    for result in results:
        url = result.get("href")
        title = result.get("title")
        print(f"Fetching : {title}")
        
        try :
            
            # Fetch the url
            response = requests.get(
                url,
                headers = headers,
                timeout = 5
            )
            
            response.raise_for_status()

            # Fetch the content
            content = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=True
            )
            if content:
                research_data.append({
                    "title": title,
                    "url": url,
                    "content": content
                })
        except Exception as e:
            print("Error:", e)
        
        # After fecthing one url sleep before fetching new url
        time.sleep(1)
    convert_to_csv_json(research_data)


def convert_to_csv_json(research_data:list):
    df = pd.DataFrame(research_data)
    df.to_csv(
        './files/research_data.csv',
        index = False
    )
    
    df.to_json(
        './files/research_data.json',
        orient = 'records',
        indent = 4
    )
    print("\nResearch completed!")
    print("Pages collected:", len(research_data))
        
        
    