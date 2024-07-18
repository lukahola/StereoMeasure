import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pyplot import MultipleLocator
import numpy as np
import math

pf = pd.read_excel(r"C:\Users\ezxtz6\OneDrive - The University of Nottingham\Documents\Nottingham\Code\StereoMeasure\measurement.xlsx", sheet_name = "30mm")
# data = [ [] for i in pf.index]
data = []
means = []
rmse = []
mae = []
for s in pf.columns:
    if type(s) == int:
        data.append(pf[s].values[0: -2])
        means.append(pf[s].values[-2])
    else: continue

sum_abs = 0
sum_pow = 0
for i in range(len(data)):
    for j in range(len(data[i])):
        sum_abs += abs(data[i][j] - means[i])
        sum_pow += (data[i][j] - means[i]) ** 2
    mae.append(sum_abs / len(data[i]))
    rmse.append(math.sqrt(sum_pow / len(data[i])))
print(rmse)
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.boxplot(data, patch_artist=True, boxprops={'facecolor': 'skyblue'}, label='Data')
ax1.plot(range(1, 11), means, 'm*-', label='Means')
# plt.plot(range(1, 11), mae, 'r.-', label='MAE')
# ax1.plot(range(1, 11), rmse, 'c.-', label='RMSE')
ax1.axhline(y=7.5, color='silver', linestyle='-.', label='Truth')
# plt.axhline(y=7.5*0.1, color='red', linestyle='dotted')
ax1.set_xlabel("Distance/mm")
ax1.set_ylabel("Result/mm")
ax1.set_ylim([-1, 25])
ax1.yaxis.set_major_locator(MultipleLocator(2.5))
for index, error in enumerate(rmse):
    if index % 2 == 1:
        plt.plot([index+1, 10], [error, error], color='gray', linestyle = '--')
ax1.set_xticks(ticks=range(1, 11), labels=range(50, 550, 50))
plt.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(range(1, 11), rmse, 'c.-', label='RMSE')
ax2.set_ylabel("MSE Error/%")
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=7.5, decimals=0))
ax2.set_ylim(-1, 25)
ax2.yaxis.set_major_locator(MultipleLocator(1.8))

plt.title("Stereo Vision Measurement Result--30mm")
plt.legend(loc='upper right')
plt.show()
