import json
import csv
import pandas as pd
import sys


with open(sys.argv[1]) as json_data:
    df = json.load(json_data)

data = df["data"]
col_names = []
for i in range(len(df["meta"]["view"]["columns"])):
    col_names.append(df["meta"]["view"]["columns"][i]['name'])

with open('test.csv','w') as f:
    writer = csv.writer(f)
    writer.writerow(col_names)
    writer.writerows(data)