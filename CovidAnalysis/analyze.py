
import pandas as pd
import os
my_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(f'{my_dir}/data/covid.csv')

print(data.head())

print(sum(data['actual_diagnostic'].values))