import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation.csv")
df.plot(x="build", y="rmse")
plt.savefig("evaluation_plot.png")