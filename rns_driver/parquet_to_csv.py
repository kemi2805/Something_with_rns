import pandas as pd

Some_data = pd.read_parquet("testEOS11.parquet")
Some_data.to_csv("parquet_test.csv")
