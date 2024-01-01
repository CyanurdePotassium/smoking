# load the model
from joblib import load

model = load('../../models/RFC.joblib')

# create importances dataframe and sort it
import pandas as pd

importances = pd.DataFrame(
    dict(
        names = model.feature_names_in_, 
        weighs = model.feature_importances_.round(2)
    )
)

importances = importances.sort_values('weighs')

# create importances plot
import matplotlib.pyplot as plt

fig, bar = plt.subplots()

fig.set_figheight(10)
fig.set_figwidth(14)

bar.barh(importances['names'], importances['weighs'], align = 'center', height = 1)
bar.set_title("Variables importance")

# save the plot
fig.savefig("../../reports/figures/importances.png")