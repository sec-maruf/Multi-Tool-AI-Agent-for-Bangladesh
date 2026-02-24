""" Create a file src/create_databases.py. This script will:

Load each dataset from Hugging Face using the datasets library.

Convert to a pandas DataFrame.

Clean column names (lowercase, underscores).

Save as a SQLite table in the databases/ folder. """

import pandas as pd
import sqlite3
from datasets import load_dataset
import os

# Ensure databases folder exists
os.makedirs("databases", exist_ok=True)

datasets_info = [
    {
        "name": "institutions",
        "hf_path": "Mahadih534/Institutional-Information-of-Bangladesh",
        "db_file": "databases/institutions.db",
        "table": "institutions"
    },
    {
        "name": "hospitals",
        "hf_path": "Mahadih534/all-bangladeshi-hospitals",
        "db_file": "databases/hospitals.db",
        "table": "hospitals"
    },
    {
        "name": "restaurants",
        "hf_path": "Mahadih534/Bangladeshi-Restaurant-Data",
        "db_file": "databases/restaurants.db",
        "table": "restaurants"
    }
]

for ds in datasets_info:
    print(f"Processing {ds['name']}...")
    dataset = load_dataset(ds["hf_path"], split="train")
    df = dataset.to_pandas()
    # Clean column names
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    
    conn = sqlite3.connect(ds["db_file"])
    df.to_sql(ds["table"], conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows to {ds['db_file']}")