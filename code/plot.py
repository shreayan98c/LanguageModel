import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'Classification Accuracy': [100 - 25.55, 100 - 8.148, 100 - 6.296, 100 - 7.407],
    'Training Size': [450 + 275, 900 + 550, 1800 + 1100, 3600 + 2200],
})

plt.title('Training Size vs Classification Accuracy')
plt.xlabel('Training Size (Number of samples in training data)')
plt.ylabel('Classification Accuracy (%age)')
plt.plot(df['Training Size'], df['Classification Accuracy'])
plt.show()
