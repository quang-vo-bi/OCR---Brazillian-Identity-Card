import glob
import numpy as np
import pandas as pd
import csv
from unidecode import unidecode


class DataReader:
    def __init__(self, classes, dataType):
        self.classes = classes
        self.dataType = dataType
        self.data = {'id': [], 'class': [], 'type': [], 'path': []}

    def get_data(self):
        return self.data

    def to_table(self, dataPath):
        # Append data
        for i in self.classes:
            for p in glob.glob(dataPath + '/' + i + '/*'):
                file_name = p[p.rfind("/") + 1:]

                # id and class
                self.data['id'].append(file_name[:file_name.find("_")])
                self.data['class'].append(i)

                # features and ground truths
                dt_type = file_name[file_name.find("_") + 1:file_name.rfind(".")]
                if dt_type in self.dataType:
                    self.data['type'].append(dt_type)
                else:
                    self.data['type'].append(np.nan)
                # data path
                self.data['path'].append(p)

        # Convert to pandas
        self.data = pd.DataFrame(self.data)
        self.data = self.data.dropna()
        self.data = self.data.pivot(index=['id', 'class'], columns='type', values='path').reset_index()
        self.data.columns.name = None

    def csv_to_tuple(self, colName):
        results = []
        results_text = []
        for fname in self.data[colName].tolist():
            with open(fname, encoding="ISO-8859-1") as f:
                headers = next(f)
                rowValue = []
                rowValueText = []
                for box in csv.reader(f):
                    rowValue.append(([int(s.strip()) if s.strip().isdigit() else None for s in box[:4]], box[4].strip()))
                    rowValueText.append(unidecode(box[4].strip().lower()))
            results.append(rowValue)
            results_text.append(' '.join(rowValueText))
        self.data[colName] = results
        self.data[f'{colName}_text'] = results_text
        return
