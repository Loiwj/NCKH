import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns

import matplotlib.pyplot as plt

# Given array
array = np.array([[89 , 0], [ 1 ,99]])

# Convert the array into percentage
array = array / array.sum(axis=1, keepdims=True)

# Convert the array into a DataFrame
df_cm = pd.DataFrame(array, index=["rooster", "hen"], columns=["rooster", "hen"])

# Plot the confusion matrix
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='.2f') # font size

plt.show()
