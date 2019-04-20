import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class TreeNode:
    """
    Class for each node in tree structure
    """

    def __init__(self, X=None, split_attr=None, split_value=None, left=None,
                 right=None):
        self.X = X  # Data
        self.split_attr = split_attr    # Feature to split on
        self.split_value = split_value  # Value on split feature
        self.left = left                # Pointer to left node
        self.right = right              # Pointer to right node
        self.n_nodes = 0                # Number of total nodes in tree


class IsolationTreeEnsemble:
    """
    Ensemble class for multiple isolation trees
    """

    def __init__(self, sample_size, n_trees=10):
        self.n = sample_size            # Sub-sample size of entire dataset
        self.n_trees = n_trees          # Number of iTrees in ensemble
        self.trees = []                 # List of root tree nodes

    def fit(self, X: np.ndarray, improved=False) -> list:
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        # Take values from pandas dataframe
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Calculate height limit
        l = math.ceil(np.log2(self.n))

        # Fit each iteration of trees
        for n in range(self.n_trees):
            # Sample from X
            sample_rows = np.random.randint(len(X),
                                            size=self.n)
            X_prime = X[sample_rows, :]
            self.trees.append(IsolationTree(height_limit=l,
                                            current_height=0)
                              .fit(X_prime, improved=improved))

            # Count number of nodes by walking tree
            self.trees[-1].n_nodes = self.count_nodes(self.trees[-1])

        return self.trees

    def path_length(self, X: np.ndarray, tree: TreeNode,
                    current_length: int = 0) -> None:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        # If no split attr, we're at a terminating node
        if tree.split_attr is None:
            np.add.at(self.total_lengths, X[:, -1].astype(int),
                      current_length + self.avg_path(len(X)))
            return

        # Get split attr
        a = tree.split_attr

        # Take appropriate path
        self.path_length(X[X[:, a] < tree.split_value], tree.left,
                         current_length + 1)
        self.path_length(X[X[:, a] >= tree.split_value], tree.right,
                         current_length + 1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        # Field to keep track of lengths
        self.total_lengths = np.zeros(len(X))
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Add column of indicies to refer to in last column of X
        X = np.hstack((X, np.arange(len(X)).reshape((len(X), 1))))

        # Calculate path length for each observation from each tree
        for tree in self.trees:
            self.path_length(X, tree)

        # Return score
        return np.apply_along_axis(lambda h: 2**(-((h / len(self.trees))
                                                   / self.avg_path(self.n))),
                                   axis=0,
                                   arr=self.total_lengths)

    def predict_from_anomaly_scores(self, scores: np.ndarray,
                                    threshold: float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores > threshold).astype(int)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        A shorthand for calling anomaly_score() and
        predict_from_anomaly_scores().
        """
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores=scores,
                                                threshold=threshold)

    def avg_path(self, n: int) -> float / int:
        """
        Compute expected average path length given size n
        """
        if n > 2:
            return 2 * (math.log(n - 1) + 0.5772156649) \
                   - ((2 * (n - 1)) / n)
        elif n == 2:
            return 1
        else:
            return 0

    def count_nodes(self, tree: TreeNode) -> int:
        """
        Walk tree to count total nodes
        """
        if tree is None: return 0
        count = 1
        count += self.count_nodes(tree.left)
        count += self.count_nodes(tree.right)
        return count


class IsolationTree:
    """
    Full tree object, creating tree structure with TreeNode objects
    """

    def __init__(self, height_limit, current_height, split_attr=None,
                 split_value=None, left=None, right=None):
        self.height_limit = height_limit      # Depth of tree
        self.current_height = current_height  # Track current height
        self.split_attr = split_attr
        self.split_value = split_value
        self.left = left
        self.right = right

    def fit(self, X: np.ndarray, improved=False) -> TreeNode:
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original
        code.
        """

        if self.current_height >= self.height_limit or len(X) <= 1:
            return TreeNode(X=X)
        else:
            Q = X.shape[1]
            if improved:
                # Randomly sample half of the features
                q = np.random.choice(range(Q), int(Q * 0.35), replace=False)
                # Randomly select a value to split on between min/max range
                p = self.value_select(X, q)
                # Split left and right trees
                left = self.split_sides(X, q, 'left', p)
                right = self.split_sides(X, q, 'right', p)
                # Grab the size of the smallest out of the left and right sides
                sizes = np.vectorize(lambda l, r: min([l.size, r.size]))(
                    left, right)

                # Grab index of feature with one split being the smallest size
                index = sizes.argmin()

                # Get final feature, split value, and left/right data values
                q = index
                p = p[index]
                X_l = left[index]
                X_r = right[index]

            else:
                # Else just choose feature at random with random value within
                # range
                q = np.random.randint(Q)
                p = np.random.uniform(min(X[:, q]), max(X[:, q]))
                X_l = X[X[:, q] < p]
                X_r = X[X[:, q] >= p]

        self.root = TreeNode(split_attr=q,
                             split_value=p,
                             left=IsolationTree(
                                 current_height=self.current_height + 1,
                                 height_limit=self.height_limit)
                             .fit(X_l, improved),
                             right=IsolationTree(
                                 current_height=self.current_height + 1,
                                 height_limit=self.height_limit)
                             .fit(X_r, improved))

        return self.root

    def value_select(self, X: np.ndarray, qs: int) -> np.array:
        """
        Randomly select a value to split on between min/max range
        """
        return np.vectorize(
            lambda q: np.random.uniform(min(X[:, q]), max(X[:, q])))(qs)

    def split_sides(self, X: np.ndarray, cols: int, side: str,
                    values: float) -> np.ndarray
        """
        Split data based on split value
        """
        if side == 'left':
            return np.vectorize(lambda q, p: X[X[:, q] > p],
                                otypes=[np.ndarray])(cols, values)
        else:
            return np.vectorize(lambda q, p: X[X[:, q] > p],
                                otypes=[np.ndarray])(cols, values)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1.0
    prev_TPR = 0

    while threshold >= 0:
        predictions = (scores > threshold).astype(int)
        cm = confusion_matrix(y, predictions)

        TN, FP, FN, TP = cm.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        if TPR >= desired_TPR:
            return threshold, FPR
        if prev_TPR == 0:
            if TPR > 0:
                prev_TPR = TPR
                decrement = 0.0001
            else:
                decrement = 0.001

        threshold -= decrement
