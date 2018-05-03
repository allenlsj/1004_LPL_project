# -*- coding: utf-8 -*-
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
    for row in data:
        try:
            writer.writerow([str(s).encode("utf-8") for s in row])
        except:
            pass