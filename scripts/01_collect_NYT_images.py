import os
import pandas as pd
from urllib.request import urlretrieve

api_key = os.getenv("NYT_API_KEY")
your_query = 

url = f'https://api.nytimes.com/svc/search/v2/articlesearch.json?q={your_query}&api-key={api_key}'

query = requests.get(url)
data = response.json()

all_docs = [item for sublist in articles['multimedia'] for item in sublist]

# Filter entries with 'subtype'== 'xlarge'
xlarge_docs = [entry for entry in all_docs if entry['subtype'=='xlarge']]
xlarge_media_df = pd.DataFrame(xlarge_docs)

articles = data['response']['docs']
articles = pd.DataFrame(articles)

all_multimedia = all_multimedia = [item for sublist in articles['multimedia'] for item in sublist]

# Filter entries with 'subtype' == 'xlarge'
xlarge_multimedia = [entry for entry in all_multimedia if entry['subtype'] == 'xlarge']
xlarge_multimedia_df = pd.DataFrame(xlarge_multimedia)

base_img_link = "https://static01.nyt.com/"

for index, row in xlarge_multimedia_df[0:2].iterrows():
  img_url = base_img_link + row['url']
  urlretrieve(img_url, f"{index}.jpg")