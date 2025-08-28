#A Confusion Matrix is a 2x2 table (for binary classification) that shows how well the classification model performed.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Example: Actual vs Predicted
y_true = [1,0,1,1,0,1,0,0,1,0]   # actual labels
y_pred = [1,0,1,0,0,1,1,0,1,0]   # predicted labels

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Display visually
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap="Blues")
plt.show()


#Precision is important when False Positives are costly (e.g., spam email filter).

#Recall is important when False Negatives are costly (e.g., disease detection).