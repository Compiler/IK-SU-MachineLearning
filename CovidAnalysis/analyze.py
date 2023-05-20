
import pandas as pd
import os
my_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(f'{my_dir}/data/covid.csv')

infected_ppl = []
males = 0
females = 0
for i in range(data.shape[0]):
    if data.loc[i]['actual_diagnostic'] == 1: 
        infected_ppl.append(data.loc[i])
        males += data.loc[i]['sex'] == 0
        females += data.loc[i]['sex'] == 1
num_females = sum(data['sex'])
num_males = data.shape[0] - num_females
print(data.loc[(data['sex'] == 1) & (data['actual_diagnostic'] == 1)])
print(sum(data['actual_diagnostic']), "test kits needed")
print(sum(data['actual_diagnostic']), "test kits needed")
print(num_males, "tested and", males, 'infected = ', males / num_males)
print(num_females, "tested and", females, 'infected = ', females / num_females)
print("Females are more likely to be infected")