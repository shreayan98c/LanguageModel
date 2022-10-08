import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# gen-spam model
accuracy_vs_size = pd.DataFrame({
    'Classification Accuracy': [100 - 25.55, 100 - 8.148, 100 - 6.296, 100 - 7.407],
    'Training Size': [450 + 275, 900 + 550, 1800 + 1100, 3600 + 2200],
})

accuracy_vs_file_size = pd.DataFrame({
    'Classification Accuracy': [100 - 25.55, 100 - 8.148, 100 - 6.296, 100 - 7.407],
    'Training Size': [1, 2, 4, 8],
})

plt.title('Training Size vs Classification Accuracy')
plt.xlabel('Training Size (Number of samples in training data)')
plt.ylabel('Classification Accuracy (%age)')
plt.plot(accuracy_vs_size['Training Size'], accuracy_vs_size['Classification Accuracy'])
plt.show()

plt.title('Training File Size vs Classification Accuracy')
plt.xlabel('Training File Size (Number of times of data in training data)')
plt.ylabel('Classification Accuracy (%age)')
plt.plot(accuracy_vs_file_size['Training Size'], accuracy_vs_file_size['Classification Accuracy'])
plt.xticks(accuracy_vs_file_size['Training Size'], ['genspam', 'times-2', 'times-4', 'times-8'])
plt.show()


# en-sp model
accuracy_vs_size = pd.DataFrame({
    'Classification Accuracy': [100 - 25.55, 100 - 20.24, 100 - 18.6, 100 - 15, 100 - 8.148, 100 - 6.296],
    'Training Size': [1000, 2000, 5000, 10000, 20000, 50000],
})

accuracy_vs_file_size = pd.DataFrame({
    'Classification Accuracy': [100 - 25.55, 100 - 20.24, 100 - 18.6, 100 - 15, 100 - 8.148, 100 - 6.296],
    'Training Size': [1, 2, 5, 10, 20, 50],
})

plt.title('Training Size vs Classification Accuracy')
plt.xlabel('Training Size (Number of samples in training data)')
plt.ylabel('Classification Accuracy (%age)')
plt.plot(accuracy_vs_size['Training Size'], accuracy_vs_size['Classification Accuracy'])
plt.show()

plt.title('Training File Size vs Classification Accuracy')
plt.xlabel('Training File Size (Number of times of data in training data)')
plt.ylabel('Classification Accuracy (%age)')
plt.plot(accuracy_vs_file_size['Training Size'], accuracy_vs_file_size['Classification Accuracy'])
plt.xticks(accuracy_vs_file_size['Training Size'], ['en-sp.1K', '2K', '5K', '10K', '20K', '50K'], rotation=90)
plt.show()
