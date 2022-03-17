import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


columns = ['random', 'random_from_samples', 'katsa', 'xiao']
lines = ['4', '8', '16', '32', '64', '128', '256']

#data = np.random.rand(len(lines), len(columns))

df = pd.DataFrame(data=np.random.rand(len(lines), len(columns)), columns = columns, index=lines)
print (df)


#fig, ax = plt.figure()
color_dict = {'random': 'blue', 'random_from_samples':'red', 'katsa':'cyan', 'xiao':'green'}
df.plot.bar(color=color_dict)
plt.show()
