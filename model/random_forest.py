import math
import random
from decision_tree import ID3
from utils.read import read_features_labels
from utils.read import read_df
from utils.evaluate import evaluate

TARGET = 'label'


class RandomForest:

    @staticmethod
    def random_forest(examples, n_tree=30):
        """
        random_forest:
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"
            eg.
            [dict(a=0, b=1, c=1, d=0, Class=1), dict(a=0, b=0, c=1, d=0, Class=0),
            dict(a=0, b=1, c=0, d=0, Class=1), dict(a=1, b=0, c=1, d=0, Class=0),
            dict(a=1, b=1, c=0, d=0, Class=0), dict(a=1, b=1, c=0, d=1, Class=0),
            dict(a=1, b=1, c=1, d=0, Class=0)]
          - n_tree: the number of trees in a random forest, default value is 30

        Takes in an array of examples, and returns a forest (a list of trees)
        trained on the examples.
        """
        forest = []
        # create forest
        for i in range(0, n_tree):
            # sample n examples from examples with replacement
            tree_examples = [item.copy() for item in random.choices(examples, k=len(examples))]
            # sample log(2)m + 1 attributes from attribute list
            attribute_list = [key for key in examples[0].keys() if key != TARGET]
            m = round(math.log2(len(attribute_list)) + 1)
            tree_attributes = random.sample(attribute_list, k=m)
            # only include tree_attributes in examples
            for example in tree_examples:
                for attribute in list(example.keys()):
                    if attribute not in tree_attributes and attribute != TARGET:
                        del example[attribute]
            # create an ID3 tree
            tree = ID3.ID3(tree_examples)
            forest.append(tree)
        return forest

    @staticmethod
    def random_forest_evaluate(forest, example):
        """
        random_forest_evaluate:
          - forest: a list of nodes, each node is the root node of a tree
          - example: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"
            eg. dict(a=0, b=0, c=1, d=0)

        Takes in a forest and one example.  Returns the Class value that the forest
        assigns to the example.
        """
        forest_prediction_labels = []
        for tree in forest:
            label = ID3.evaluate(tree, example)
            forest_prediction_labels.append(label)
        return max(set(forest_prediction_labels), key=forest_prediction_labels.count)

    @staticmethod
    def random_forest_test(forest, examples):
        """
        random_forest_test:
          - forest: a list of nodes, each node is the root node of a tree
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        Takes in a trained forest and a test set of examples.  Returns the accuracy (fraction
        of examples the tree classifies correctly).
        """
        success = 0
        for example in examples:
            if forest is not None:
                ans = RandomForest.random_forest_evaluate(forest, example)
                if ans == example[TARGET]:
                    success += 1
        return success / len(examples)


def prepare_data():
    train_df = read_df('../sign_mnist_train_tiny.csv')
    test_df = read_df('../sign_mnist_test_tiny.csv')
    # Transform data
    train_data = [dict(zip(train_df.columns, row)) for row in train_df.values]
    for row in test_df.values:
        a = dict(zip(test_df.columns[1:], row[1:]))
    test_data = [dict(zip(test_df.columns[1:], row[1:])) for row in test_df.values]
    test_label = [row[0] for row in test_df.values]
    return train_data, test_data, test_label


if __name__ == '__main__':
    # Load data
    train_data, test_data, test_label = prepare_data()
    # Train
    forest = RandomForest.random_forest(train_data)
    # Test
    predictions = []
    for test_feature in test_data:
        pred = RandomForest.random_forest_evaluate(forest, test_feature)
        predictions.append(pred)
    # Evaluate
    precision, recall, f1 = evaluate('random_forest', test_label, predictions)

