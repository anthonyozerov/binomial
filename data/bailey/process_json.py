import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

os.makedirs('CsvData', exist_ok=True)
for file in os.listdir('JsonData'):
    with open('JsonData/' + file, 'r') as f:
        data = json.load(f)
    folder = file.split(".")[0]
    print(folder)
    os.makedirs('CsvData/' + folder, exist_ok=True)
    for key, value in tqdm(data.items()):
        csv_name = key.replace(" ", "_").replace('/', '[div]') + ".csv"
        if len(csv_name) > 100:
            csv_name = csv_name[:97] + ".csv"
        rows = []
        for ID, measurement in value.items():
            row = {}
            row['source'] = ID.replace('#', '')
            row['value'] = measurement[0]
            row['error-'] = np.sqrt(measurement[1][0])
            row['error+'] = np.sqrt(measurement[1][1])
            row['uncertainty'] = (row['error-'] + row['error+']) / 2
            row['year'] = measurement[8]
            rows.append(row)
        df = pd.DataFrame(rows)
        # print(csv_name)
        df.to_csv('CsvData/' + folder + '/' + csv_name, index=False)