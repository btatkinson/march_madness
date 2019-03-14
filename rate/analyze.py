import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# df = pd.read_csv('elok.csv')
df = pd.read_csv('sub_errors.csv')
df = df.groupby(['Day']).mean()
df = df.reset_index()
df = df.drop(columns='Unnamed: 0')

# plt.plot(df['MOV K'],df['MOV Error'], 'r', label="MOV")
# plt.plot(df.Week,df.WL, 'k', label="Win-Loss")
# plt.plot(df.Day,df['WLM Error'], 'c', label="Win-Loss Margin")
plt.plot(df.Day,df['Elo Error'], 'r', label="Elo")
plt.plot(df.Day,df['Dim Error'], 'm', label="Diminishing K")
plt.plot(df.Day,df['MOV Error'], 'b', label="Margin of Victory")
plt.plot(df.Day,df['DimV Error'], 'y', label="Dim + MOV")
# plt.plot(df.Week,df.Combo, 'g', label="Combo")

plt.legend()
plt.xlabel("Day Number")
plt.ylabel("Season Error")
plt.show()





# end
