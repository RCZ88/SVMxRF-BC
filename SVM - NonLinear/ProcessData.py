import pandas as pd
import numpy as np

fileName:str = input("Enter file Name: ")
data = pd.read_csv(f'{fileName}.csv')

def mapData(data, fileName):
    if fileName == 'diabetic_data':
        ms:dict[str, int] = {}
        countRef:int = 1
        for label, value in data.iloc['medical_specialty']:
            if ms.get(value) is None and ms.get(value) != '?':
                ms[value] = countRef
        age:dict[int, int] = {}
        countRef = 1
        for label, value in data.iloc['age']:
            if age.get(value) is None and age.get(value) != '?':
                age.update({value[1]: countRef})
                age[int(value[1])*10] = countRef
        race:dict[str, int] = {'Caucasian': 1, 'AfricanAmerican':2, 'Other':0}
        symp:dict[str, int] = {'Steady':1,'No':0, 'Yes':2 }




