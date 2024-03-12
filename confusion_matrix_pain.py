import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
# Mảng đã cho
array = np.array([[87, 2], [2, 98]])



# Convert the array into a DataFrame
df_cm = pd.DataFrame(array, index=["rooster", "hen"], columns=["rooster", "hen"])

# Plot the confusion matrix
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size

plt.show()
