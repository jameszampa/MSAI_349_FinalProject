import math
import copy
import random


PRUNING_METHOD = 'REP'  # REP: reduced error pruning, CVP: critical value pruning
TARGET = 'label'

class Node:

    def __init__(self):
        self.label = None
        self.attribute = None
        self.children = {}


class ID3:

    @staticmethod
    def ID3(examples, criterion='information_gain', max_feature_number=None):
        # Use information gain to determine which attribute to split on. Always split on the attribute with the highest information gain.
        # If there are missing values replace them with the most common value for that attribute.
        # If there is zero information gain prefer to split that is non-trival (i.e. one where all examples do not have the same attribute value).
        """
        ID3:
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        Takes in an array of examples, and returns a tree (an instance of Node)
        trained on the examples.  Each example is a dictionary of attribute:value pairs,
        and the target class variable is a special attribute with the name "Class".
        Any missing attributes are denoted with a value of "?"
        """
        # Create a node t for the tree
        t = Node()
        # examples = Helper.fix_missing_values(examples)
        # Label t with the most common value of the target attribute in the examples
        t.label = Helper.get_most_common_value(examples)
        # If all examples are the same class return t
        if Helper.is_same_class(examples):
            return t
        # If attributes is empty return t
        attributes = [key for key in examples[0].keys() if key != TARGET]
        if len(attributes) == 0:
            return t
        # Otherwise...
        # Let A be the attribute that has the highest information gain
        best_attribute, best_attribute_values = Helper.get_best_attribute(examples, criterion, max_feature_number)
        # Assign t the decision attribute A
        t.attribute = best_attribute
        print (best_attribute)
        # For each possible value "a" in A do
        for best_attribute_value in best_attribute_values:
            # Add a new tree branch below t corresponding to the test A = a
            # Let examples_a be the subset of examples that have value a for A
            examples_a = Helper.get_examples_a(examples, best_attribute, best_attribute_value)
            # If examples_a is empty
            if len(examples_a) == 0:
                # Below this new branch add a leaf node with label = most common target value in the examples
                leaf = Node()
                leaf.label = Helper.get_most_common_value(examples)
                t.children[best_attribute_value] = leaf
            else:
                # remove the attribute from the new D
                examples_a_copy = copy.deepcopy(examples_a)
                for item in examples_a_copy:
                    item.pop(best_attribute)
                t.children[best_attribute_value] = ID3.ID3(examples_a_copy)
        # return the tree
        print (t)
        return t

    @staticmethod
    def test(node, examples):
        """
        test:
          - node: a node in a decision tree
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
        of examples the tree classifies correctly).
        """
        accuracy = 0
        for example in examples:
            if ID3.evaluate(node, example) == example[TARGET]:
                accuracy += 1
        return accuracy / len(examples)

    @staticmethod
    def evaluate(node, example):
        """
        evaluate:
          - node: a node in a decision tree
          - example: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        Takes in a tree and one example.  Returns the Class value that the tree
        assigns to the example.
        """
        if len(node.children) == 0:
            return node.label
        else:
            a = example[node.attribute]
            if a not in node.children:
                return node.label
            else:
                return ID3.evaluate(node.children[a], example)

    @staticmethod
    def unit_test_id3():
        data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
        tree = ID3.ID3(data)
        if tree is not None:
            ans = ID3.evaluate(tree, dict(a=1, b=0))
            if ans != 1:
                print("ID3 test failed.")
            else:
                print("ID3 test succeeded.")
        else:
            print("ID3 test failed -- no tree returned")


class Helper:

    @staticmethod
    # Returns the most common value in the list
    def mode(examples: list):
        max_example = None
        max_count = 0
        for example in set(examples):
            count = examples.count(example)
            if count > max_count:
                max_example = example
                max_count = count
        return max_example

    @staticmethod
    def get_most_common_value(examples):
        """
        get_most_common_value:
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        This function returns the most common value of the target attribute in the examples
        """
        return Helper.mode([example[TARGET] for example in examples])

    @staticmethod
    def is_same_class(examples):
        """
        is_same_class:
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        This function returns True if all the examples belong to the same class, and False otherwise
        """
        return len(set([example[TARGET] for example in examples])) == 1

    @staticmethod
    def get_information_gain(examples, attribute):
        """
        get_information_gain:
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"
          - attribute: the attribute to compute the information gain on
        """
        # Compute the entropy of the dataset
        total_number_of_examples = len(examples)
        # Computer the number of unique classes
        unique_classes = set([example[TARGET] for example in examples])
        # Compute the number of examples for each class
        number_of_examples_for_each_class = {}
        for example in examples:
            if example[TARGET] not in number_of_examples_for_each_class:
                number_of_examples_for_each_class[example[TARGET]] = 0
            number_of_examples_for_each_class[example[TARGET]] += 1
        # Compute the entropy of the dataset
        entropy = 0
        for unique_class in unique_classes:
            probability = number_of_examples_for_each_class[unique_class] / total_number_of_examples
            entropy += probability * math.log(probability, 2)
        entropy *= -1

        # Find the attribute with the highest information gain
        # Compute the number of examples for each value of the attribute
        number_of_examples_for_each_attribute_value = {}
        for example in examples:
            if example[attribute] not in number_of_examples_for_each_attribute_value:
                number_of_examples_for_each_attribute_value[example[attribute]] = 0
            number_of_examples_for_each_attribute_value[example[attribute]] += 1

        # Compute the information gain of the attribute
        information_gain = entropy
        for attribute_value, number_of_examples in number_of_examples_for_each_attribute_value.items():
            probability = number_of_examples / total_number_of_examples
            examples_with_attribute_value = [example for example in examples if example[attribute] == attribute_value]
            entropy_of_attribute_value = 0
            for unique_class in unique_classes:
                probability_of_class = len([example for example in examples_with_attribute_value if
                                            example[TARGET] == unique_class]) / number_of_examples
                if probability_of_class != 0:
                    entropy_of_attribute_value += probability_of_class * math.log(probability_of_class, 2)
            entropy_of_attribute_value *= -1
            information_gain -= probability * entropy_of_attribute_value
        return information_gain

    @staticmethod
    def get_gini_gain(examples, attribute):
        """
        get_gini_gain:
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"
          - attribute: the attribute to compute the gini gain on
        """
        # Compute the gini impurity of the dataset
        total_number_of_examples = len(examples)
        # Computer the number of unique classes
        unique_classes = set([example[TARGET] for example in examples])
        # Compute the number of examples for each class
        number_of_examples_for_each_class = {}
        for example in examples:
            if example[TARGET] not in number_of_examples_for_each_class:
                number_of_examples_for_each_class[example[TARGET]] = 0
            number_of_examples_for_each_class[example[TARGET]] += 1
        # Compute the gini impurity of the dataset
        gini_impurity = 0
        for unique_class in unique_classes:
            probability = number_of_examples_for_each_class[unique_class] / total_number_of_examples
            gini_impurity += 1 - probability**2

        # Find the attribute with the highest gini gain
        # Compute the number of examples for each value of the attribute
        number_of_examples_for_each_attribute_value = {}
        for example in examples:
            if example[attribute] not in number_of_examples_for_each_attribute_value:
                number_of_examples_for_each_attribute_value[example[attribute]] = 0
            number_of_examples_for_each_attribute_value[example[attribute]] += 1

        # Compute the gini gain of the attribute
        gini_gain = gini_impurity
        for attribute_value, number_of_examples in number_of_examples_for_each_attribute_value.items():
            probability = number_of_examples / total_number_of_examples
            examples_with_attribute_value = [example for example in examples if example[attribute] == attribute_value]
            gini_of_attribute_value = 0
            for unique_class in unique_classes:
                probability_of_class = len([example for example in examples_with_attribute_value if
                                            example[TARGET] == unique_class]) / number_of_examples
                if probability_of_class != 0:
                    gini_of_attribute_value += 1 - probability_of_class**2
            gini_gain -= probability * gini_of_attribute_value
        return gini_gain

    @staticmethod
    def get_best_attribute(examples, criterion, max_feature_number=None):
        """
        get_best_attribute:
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        This function returns the attribute that has the highest information gain
        """
        unique_attributes = set()
        for example in examples:
            for key, _ in example.items():
                unique_attributes.add(key)

        best_attribute = None
        best_gain = 0
        if max_feature_number:
            feature_number = min(max_feature_number, len(unique_attributes))
            attributes_list = random.sample(unique_attributes, feature_number)
        else:
            attributes_list = unique_attributes
        for attribute in attributes_list:
            if attribute == TARGET:
                continue
            if criterion == 'gini_gain':
                info_gain = Helper.get_gini_gain(examples, attribute)
            else:
                info_gain = Helper.get_information_gain(examples, attribute)
            if info_gain > best_gain:
                best_attribute = attribute
                best_gain = info_gain

        if best_attribute is None:
            # choose attribute that is non-trivial
            for attribute in unique_attributes:
                if attribute == TARGET:
                    continue
                if not Helper.is_same_class(examples):
                    best_attribute = attribute
                    break
        assert best_attribute is not None

        # get the unique values best attribute can take
        best_attribute_values = set()
        for example in examples:
            best_attribute_values.add(example[best_attribute])

        return best_attribute, best_attribute_values

    @staticmethod
    def get_examples_a(examples, best_attribute, a):
        """
        get_examples_a:
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"
          - a: the value of the attribute to get the examples for

        This function returns the subset of examples that have value a for the attribute
        """
        return [example for example in examples if example[best_attribute] == a]

    @staticmethod
    def fix_missing_values(examples):
        """
        fix_missing_values:
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        attribute value missing: replace all missing values with the most common value for that attribute
        class missing: discard the data
        """
        # get the most common non-missing value for each attribute
        attribute_most_common_value = {}
        attributes = [key for key in examples[0].keys() if key != TARGET]
        for attribute in attributes:
            attribute_values = [item[attribute] for item in examples if
                                item[attribute] != '?' and item[attribute] is not None]
            if len(attribute_values) != 0:
                most_common_value = max(set(attribute_values), key=attribute_values.count)
                attribute_most_common_value[attribute] = most_common_value
            else:
                # all values in this attribute is missing, we discard this attribute
                for item in examples:
                    item.pop(attribute)
        # deal with the missing value
        for example in examples:
            for attribute, value in example.items():
                if value == '?' or value is None:
                    if attribute == TARGET:
                        # class is missing: discard this data
                        examples.remove(example)
                        break
                    else:
                        # attribute value is missing: replace the missing value with the most common value for that
                        # attribute
                        example[attribute] = attribute_most_common_value[attribute]
        return examples

    @staticmethod
    def is_info_below(examples, node, threshold):
        """
        is_info_below:
          - node: a node in a decision tree
          - threshold: the threshold to use for the information gain

        Given node, return True if any node below has an information gain more than threshold, and False otherwise
        """
        if len(node.children) == 0:
            return False
        for child in node.children.values():
            if len(child.children) == 0:
                continue
            if Helper.get_information_gain(examples, child.attribute) > threshold:
                return True
            elif Helper.is_info_below(examples, child, threshold):
                return True
        return False

    @staticmethod
    def get_most_common_value_below(class_values, node):
        """
        get_most_common_value_below:
          - node: a node in a decision tree

        Given a node return the most common class of all leaf nodes below it
        """
        if len(node.children) == 0:
            return node.label
        for child in node.children.values():
            class_value = Helper.get_most_common_value_below(class_values, child)
            if class_value not in class_values:
                class_values[class_value] = 0
            class_values[class_value] += 1

        for key, value in class_values.items():
            if value == max(class_values.values()):
                return key


class Prune:

    def __init__(self):
        pass

    @staticmethod
    def prune(node, examples):
        """
        prune:
          - node: a node in a decision tree
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"
          - mode: pruning methods, default is REP
            REP: reduced error pruning
            CVP: critical value pruning

        Takes in a trained tree and a validation set of examples.  Prunes nodes in order
        to improve accuracy on the validation data; the precise pruning strategy is up to you.
        """
        if PRUNING_METHOD == 'CVP':
            return Prune.critical_value_pruning(node, examples)
        elif PRUNING_METHOD == 'REP':
            return Prune.reduced_error_pruning(node, examples)
        else:
            raise ValueError('Invalid pruning method')

    @staticmethod
    def critical_value_pruning(node, examples, threshold=0.1):
        """
        critical_value_pruning:
          - node: a node in a decision tree
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        This function prunes the tree using critical value pruning
        """
        if len(node.children) == 0:
            return node

        info_gained = Helper.get_information_gain(examples, node.attribute)
        if info_gained > threshold:
            for child in node.children.values():
                if len(child.children) == 0:
                    continue
                Prune.critical_value_pruning(child, examples, threshold)
        else:
            if Helper.is_info_below(examples, node, threshold):
                for child in node.children.values():
                    if len(child.children) == 0:
                        continue
                    Prune.critical_value_pruning(child, examples, threshold)
            else:
                node_label = Helper.get_most_common_value_below({}, node)
                node.children = {}
                node.label = node_label
        return node

    @staticmethod
    def reduced_error_pruning(node, examples):
        """
        reduced_error_pruning:
          - node: a node in a decision tree
          - examples: a dictionary of attribute:value pairs,
            and the target class variable is a special attribute with the name "Class"

        This function prunes the tree using reduced error pruning
        """
        # Reduced Error Pruning
        if len(node.children) == 0:
            # do nothing to a leaf node
            return
        if len(examples) == 0:
            # do nothing with empty data
            return
        for value, child_node in node.children.items():
            if len(child_node.children) != 0:
                # the child node is a non-leaf node, aka, an attribute node
                new_examples = copy.deepcopy([item for item in examples if item[node.attribute] == value])
                Prune.reduced_error_pruning(child_node, new_examples)
        # preparation work: save the value for node in case of roll back
        node_children = node.children
        node_label = node.label
        labels_in_examples = [item[TARGET] for item in examples]
        before_prune_accuracy = ID3.test(node, examples)
        # prune the node: change it to a leaf node (remove all children) with the most common label
        node.children = {}
        node.label = max(set(labels_in_examples), key=labels_in_examples.count)
        # test accuracy
        after_prune_accuracy = ID3.test(node, examples)
        if after_prune_accuracy <= before_prune_accuracy:
            # bad pruning: roll back the node
            node.children = node_children
            node.label = node_label
        print(node)
        return node

    @staticmethod
    def test_pruning():
        data = [dict(a=0, b=1, c=1, d=0, Class=1), dict(a=0, b=0, c=1, d=0, Class=0),
                dict(a=0, b=1, c=0, d=0, Class=1), dict(a=1, b=0, c=1, d=0, Class=0),
                dict(a=1, b=1, c=0, d=0, Class=0), dict(a=1, b=1, c=0, d=1, Class=0),
                dict(a=1, b=1, c=1, d=0, Class=0)]
        validation_data = [dict(a=0, b=0, c=1, d=0, Class=1), dict(a=1, b=1, c=1, d=1, Class=0)]
        tree = ID3.ID3(data)
        Prune.prune(tree, validation_data)
        if tree is not None:
            ans = ID3.evaluate(tree, dict(a=0, b=0, c=1, d=0))
            if ans != 1:
                print("pruning test failed.")
            else:
                print("pruning test succeeded.")
        else:
            print("pruning test failed -- no tree returned.")




