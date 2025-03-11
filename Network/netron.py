import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the confusion matrix
cm = np.array([
    [887, 24, 33, 27, 12, 1, 7, 4, 38, 24],
    [21, 898, 3, 6, 3, 2, 2, 2, 10, 48],
    [62, 3, 628, 47, 69, 44, 45, 28, 4, 7],
    [23, 6, 53, 610, 54, 162, 49, 24, 6, 7],
    [6, 6, 59, 26, 738, 32, 44, 54, 7, 0],
    [13, 2, 30, 167, 32, 708, 20, 24, 5, 3],
    [8, 3, 35, 55, 22, 22, 854, 2, 1, 1],
    [13, 1, 27, 35, 56, 60, 2, 827, 1, 4],
    [51, 32, 8, 14, 4, 5, 4, 8, 865, 11],
    [17, 56, 5, 9, 7, 4, 2, 4, 16, 890]
])

# Create figure and axes
plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')

# Add labels
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Show the plot
plt.tight_layout()
plt.show()