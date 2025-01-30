#!/usr/bin/env python3

"""
Module for decision tree management with implementations of Node, Leaf,
and Decision_Tree classes. This module allows the creation of a decision tree
with methods to compute the maximum depth of subtrees.
"""
import numpy as np


class Node:
    """
    Class representing a node in a decision tree.

    Attributes:
    - feature: The feature used for splitting the node.
    - threshold: The threshold used for splitting the node.
    - left_child: Left child node.
    - right_child: Right child node.
    - is_leaf: Boolean indicating if the node is a leaf.
    - is_root: Boolean indicating if the node is the root.
    - sub_population: Data associated with the node.
    - depth: The depth of the node in the tree.
    """
    def __init__(
        self, feature=None, threshold=None, left_child=None, right_child=None,
        is_root=False, depth=0
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = {}
        self.upper = {}

    def max_depth_below(self):
        """
        Computes the maximum depth of the subtrees
        starting from the current node.

        Returns:
        - The maximum depth among the left and right subtrees.
        """
        if self.is_leaf:
            return self.depth
        left_depth = (
            self.left_child.max_depth_below() if self.left_child else -1
        )
        right_depth = (
            self.right_child.max_depth_below() if self.right_child else -1
        )
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the nodes below the current node.

        Parameters:
        - only_leaves (bool): If True, counts only leaf nodes.
        If False, counts all nodes.

        Returns:
        - The number of nodes below this node (leaf or non-leaf).
        """
        count = 1 if not only_leaves or self.is_leaf else 0
        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def left_child_add_prefix(self, text):
        """
        Adds a prefix for the left child in the string representation.

        Parameters:
        - text (str): The text representation of the child node.

        Returns:
        - str: The text with the appropriate prefix for the left child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds a prefix for the right child in the string representation.

        Parameters:
        - text (str): The text representation of the child node.

        Returns:
        - str: The text with the appropriate prefix for the right child.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        if new_text.endswith("\n"):
            new_text = new_text[:-1]
        return new_text

    def __str__(self):
        """
        Returns a string representation of the tree structure
        starting from the current node.

        Returns:
        - str: A string representation of the node and its subtrees.
        """
        result = f"[feature={self.feature}, threshold={self.threshold}]\n"
        if self.is_root:
            result = "root " + result
        else:
            result = "-> node " + result

        if self.left_child:
            result += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))

        if self.depth == 0:
            result += "\n"
        return result

    def get_leaves_below(self):
        """
        Returns the list of all leaves of the tree

        Returns:
        - str: A string representation of list of all leaves of the tree
        """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            if self.left_child:
                leaves.extend(self.left_child.get_leaves_below())
            if self.right_child:
                leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Updates the lower and upper bounds for the node and its children.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                elif child == self.right_child:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Compute the indicator function for
        the node based on its lower and upper bounds.
        """
        def is_large_enough(x):
            """
            Check if all features are greater than or equal to lower bounds.
            """
            check = np.array([
                np.greater(x[:, key], self.lower[key])
                for key in self.lower.keys()
            ])
            return np.all(check, axis=0)

        def is_small_enough(x):
            """
            Check if all features are less than or equal to upper bounds.
            """
            check = np.array([
                np.less_equal(x[:, key], self.upper[key])
                for key in self.upper.keys()
            ])
            return np.all(check, axis=0)

        self.indicator = lambda x: np.all(np.array([
            is_large_enough(x), is_small_enough(x)
        ]), axis=0)

    def pred(self, x):
        """
        Recursively predict the value for a
        single individual by traversing the tree.

        Depending on the feature value of the individual, this method decides
        whether to follow the left or right child.

        Args:
            x (np.ndarray): A 1D NumPy array
            representing the features of an individual.

        Returns:
            The predicted value from a leaf node.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Class representing a leaf in a decision tree.

    Attributes:
    - value: The value of the leaf.
    - depth: The depth of the leaf in the tree.
    """
    def __init__(self, value, depth=None):
        """
        Initializes a leaf with a value and depth.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf.

        Returns:
        - The depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Counts only the leaf node itself, since it doesn't have children.

        Parameters:
        - only_leaves (bool): This parameter is ignored
        for leaf nodes since it will always count the leaf.

        Returns:
        - int: Always returns 1 as it counts the leaf node itself.
        """
        return 1

    def __str__(self):
        """
        Returns a string representation of the leaf node.

        Returns:
        - str: A string representation in the format
        '-> leaf [value={value}]' where value is the value of the leaf.
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        Returns the list of all leaves of the tree

        Returns:
        - str: A string representation of list of all leaves of the tree
        """
        return [self]

    def update_bounds_below(self):
        """
        Pass
        """
        pass

    def pred(self, x):
        """
        Predict the value for a single individual at the leaf node.

        Args:
            x (np.ndarray): A 1D NumPy array
            representing the features of an individual.

        Returns:
            The value stored in the leaf node (prediction).
        """
        return self.value


class Decision_Tree():
    """
    Class representing a decision tree.

    Attributes:
    - max_depth: Maximum depth of the tree.
    - min_pop: Minimum size of sub-populations in nodes.
    - seed: Seed for random number generation.
    - split_criterion: Criterion for splitting nodes.
    - root: Root node of the tree.
    """
    def __init__(
        self, max_depth=10, min_pop=1, seed=0, split_criterion="random",
        root=None
    ):
        """
        Initializes a decision tree with the given parameters.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Returns the maximum depth of the tree starting from the root.

        Returns:
        - The maximum depth of the root subtree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the nodes in the tree, starting from the root node.

        Parameters:
        - only_leaves (bool): If True, counts only leaf nodes.
        If False, counts all nodes.

        Returns:
        - int: The total number of nodes in the tree .
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Returns a string representation of the entire
        decision tree starting from the root.

        Returns:
        - str: A string representation of the tree,
        beginning with the root node and recursively showing its children.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Returns the list of all leaves of the tree

        Returns:
        - str: A string representation of list of all leaves of the tree
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Updates the lower and upper bounds for the node and its children.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Compute an efficient prediction function for the decision tree.

        Updates tree bounds, leaf indicators, and creates a vectorized
        prediction method that can handle multiple individuals simultaneously.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        if not leaves:
            self.predict = lambda A: np.array([])
        else:
            self.predict = lambda A: np.array([
                leaf.value for leaf in leaves
            ])[
                np.argmax([leaf.indicator(A) for leaf in leaves], axis=0)
            ]

    def pred(self, x):
        """
        Predict the value for a single
        individual by starting from the root node.

        Args:
            x (np.ndarray): A 1D NumPy array
            representing the features of an individual.

        Returns:
            The predicted value from a leaf node.
        """
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """
        Fit the decision tree to the given data.

        Args:
            explanatory: 2D array of feature values.
            target: 1D array of target values.
            verbose: If 1, print training results.

        Trains the tree, updates prediction function,
        and optionally prints results.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(
        self.explanatory,self.target
    )}""")

    def np_extrema(self, arr):
        """
        Find minimum and maximum values in an array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Generate a random split criterion for a node.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """
        Recursively fit a node and its children in the decision tree.
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = np.logical_and(
            node.sub_population,
            self.explanatory[:, node.feature] <= node.threshold
        )

        # Is left node a leaf?
        is_left_leaf = (
            np.sum(left_population) < self.min_pop
            or node.depth + 1 == self.max_depth
            or np.all(
                self.target[left_population]
                == self.target[left_population][0]
            )
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf?
        is_right_leaf = (
            np.sum(right_population) < self.min_pop
            or node.depth + 1 == self.max_depth
            or np.all(
                self.target[right_population]
                == self.target[right_population][0]
            )
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf node with the most common class as its value.
        """
        value = np.argmax(np.bincount(self.target[sub_population]))
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Create a new internal node as a child of the current node.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Calculate the accuracy of the model on test data.
        """
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size

    def possible_thresholds(self, node, feature):
        """
        Compute possible thresholds for splitting a node on a given feature.
        """
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[: -1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Compute the best split for a single feature based on Gini impurity.

        Args:
            node: The current node.
            feature: The index of the feature to
            compute the split criterion for.

        Returns:
            A tuple (best_threshold, best_gini_avg), where:
                - best_threshold is the threshold
                that minimizes the Gini impurity.
                - best_gini_avg is the corresponding Gini impurity average.
        """
        thresholds = self.possible_thresholds(node, feature)
        if len(thresholds) == 0:
            raise ValueError(
                f"No thresholds available for feature {feature} "
                f"at node {node}."
            )
        feature_values = self.explanatory[node.sub_population, feature]
        target_values = self.target[node.sub_population]

        n_classes = len(np.unique(self.target))
        n_samples = len(node.sub_population)

        left_gini_impurities = np.zeros(len(thresholds))
        right_gini_impurities = np.zeros(len(thresholds))
        avg_gini_impurity = np.zeros(len(thresholds))

        for idx, threshold in enumerate(thresholds):
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_classes = target_values[left_mask]
            right_classes = target_values[right_mask]

            left_class_counts = np.array(
                [np.sum(left_classes == i) for i in range(n_classes)]
            )
            right_class_counts = np.array(
                [np.sum(right_classes == i) for i in range(n_classes)]
            )

            left_proportions = (
                left_class_counts / len(left_classes)
                if len(left_classes) > 0
                else np.zeros(n_classes)
            )

            right_proportions = (
                right_class_counts / len(right_classes)
                if len(right_classes) > 0
                else np.zeros(n_classes)
            )

            left_gini = 1 - np.sum(left_proportions ** 2)
            right_gini = 1 - np.sum(right_proportions ** 2)

            left_gini_impurities[idx] = left_gini
            right_gini_impurities[idx] = right_gini

        for idx in range(len(thresholds)):
            left_size = np.sum(feature_values <= thresholds[idx])
            right_size = np.sum(feature_values > thresholds[idx])

            total_size = left_size + right_size
            if total_size == 0:
                avg_gini_impurity[idx] = float('inf')
            else:
                left_weight = left_size / total_size
                right_weight = right_size / total_size
                avg_gini_impurity[idx] = (
                    left_weight * left_gini_impurities[idx]
                    + right_weight * right_gini_impurities[idx]
                )

        best_threshold_idx = np.argmin(avg_gini_impurity)
        best_threshold = thresholds[best_threshold_idx]
        best_gini_impurity = avg_gini_impurity[best_threshold_idx]

        return best_threshold, best_gini_impurity

    def Gini_split_criterion(self, node):
        """
        Find the best split for a node based on Gini impurity.
        """
        X = np.array([
            self.Gini_split_criterion_one_feature(node, i)
            for i in range(self.explanatory.shape[1])
        ])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]
