'''
PART 1: ETL the dataset and save in `data/`

Here is the imbd_movie data:
https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true

It is in JSON format, so you'll need to handle accordingly and also figure out what's the best format for the two analysis parts. 
'''

import os
import pandas as pd
import json

# Create '/data' directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

# Load datasets and save to '/data'
url = "https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true"
df = pd.read_json(url, lines=True)

# Save to CSV
csv_path = os.path.join(data_dir, "imdb_movies_data.csv")
df.to_csv(csv_path, index=False)

# Save to json
jsonl_path = os.path.join(data_dir, "imdb_movies_data.jsonl")
df.to_json(jsonl_path, orient="records", lines=True)
