import numpy as np
import pandas as pd

data = pd.read_csv('/home/zqtan/hyper embed/mammal_closure.tsv')
shuffled_idx = data.index.values
np.random.shuffle(shuffled_idx)

split_ratio = 0.7
split_idx = int(len(data) * split_ratio)

data_part1 = data.loc[shuffled_idx[:split_idx]]
data_part2 = data.loc[shuffled_idx[split_idx:]]

data_part1.to_csv('/home/zqtan/hyper embed/mammal_closure_part1.tsv', index=False)
data_part2.to_csv('/home/zqtan/hyper embed/mammal_closure_part2.tsv', index=False)