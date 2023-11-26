import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def evaluate(model_name, ground_truth, predictions):
    print("\nTest Ground truth: ", ground_truth)
    print("Test model prediction: ", predictions)

    # Calculate precision, recall and f1 for each class
    labels = sorted(list(set(ground_truth)))
    precision_each_class = precision_score(ground_truth, predictions, average=None, labels=labels)
    recall_each_class = recall_score(ground_truth, predictions, average=None, labels=labels)
    f1_each_class = f1_score(ground_truth, predictions, average=None, labels=labels)
    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    for class_label, prec, rec, f1 in zip(labels, precision_each_class, recall_each_class, f1_each_class):
        print(f"Class: {class_label}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
        precision_sum += prec
        recall_sum += rec
        f1_sum += f1

    # Calculate average precision, recall and f1
    avg_precision = precision_sum / len(labels)
    avg_recall = recall_sum / len(labels)
    avg_f1 = f1_sum / len(labels)
    print("\nAverage precision:", avg_precision)
    print("Average recall:", avg_recall)
    print("Average F1 Score:", avg_f1)

    # Plot the confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(40, 35))
    params = {'axes.labelsize': 30, 'axes.titlesize': 30, 'font.size': 20, 'legend.fontsize': 20,
              'xtick.labelsize': 20, 'ytick.labelsize': 20}
    plt.rcParams.update(params)
    sns.heatmap(cm, annot=True, linewidths=1, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_' + str(model_name) + '.png')
    return avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    ground_truth = ["a", "b", "c", "d"]
    predictions = ["b", "b", "d", "d"]
    avg_precision, avg_recall, avg_f1 = evaluate('id3', ground_truth, predictions)

