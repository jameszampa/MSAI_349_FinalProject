import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def evaluate(model_name, ground_truth, predictions):
    print("\nTest Ground truth: ", ground_truth)
    print("Test model prediction: ", predictions)

    # Calculate precision, recall and f1 for each class
    labels = list(set(predictions))
    precision_each_class = precision_score(ground_truth, predictions, average=None, labels=labels)
    recall_each_class = recall_score(ground_truth, predictions, average=None, labels=labels)
    f1_each_class = f1_score(ground_truth, predictions, average=None, labels=labels)
    for class_label, prec, rec, f1 in zip(labels, precision_each_class, recall_each_class, f1_each_class):
        print(f"Class: {class_label}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    # Calculate average precision, recall and f1
    avg_precision = precision_score(ground_truth, predictions, average='micro')
    avg_recall = recall_score(ground_truth, predictions, average='micro')
    avg_f1 = f1_score(ground_truth, predictions, average='micro')
    print("\nAverage precision:", avg_precision)
    print("Average recall:", avg_recall)
    print("Average F1 Score:", avg_f1)

    # Plot the confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_' + str(model_name) + '.png')
    return avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    ground_truth = [6, 5, 10, 0, 3, 21, 10, 14, 3, 7, 8, 8, 21, 12, 7, 4, 22, 0, 7, 7, 2, 0, 21, 4, 10, 15, 2, 15, 7, 1, 7, 8, 13, 19,
     3, 21, 13, 3, 18, 14, 15, 23, 8, 15, 14, 5, 17, 4, 19, 13]
    predictions = [11, 19, 4, 19, 16, 15, 19, 2, 11, 18, 18, 18, 10, 3, 16, 18, 18, 23, 8, 14, 5, 6, 18, 11, 22, 1, 5, 18, 14, 20, 16,
     20, 18, 13, 1, 3, 0, 1, 18, 1, 18, 18, 18, 10, 19, 14, 18, 18, 12, 18]
    avg_precision, avg_recall, avg_f1 = evaluate('id3', ground_truth, predictions)

