import pandas as pd

Some_data = pd.read_parquet("testEOSfiltered2.parquet")
Some_data.to_csv("parquet_test2.csv")
