import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    def count_H(R):
        p0 = np.sum(R * (target_vector == 0), axis=1) / np.sum(R, axis=1)
        p1 = np.sum(R * (target_vector == 1), axis=1) / np.sum(R, axis=1)
        return 1 - p0 ** 2 - p1 ** 2

    def count_gini(thresholds_local):
        Rl = sort_feature_vector.reshape(1, -1) < thresholds_local.reshape(-1, 1)
        Rr = sort_feature_vector.reshape(1, -1) > thresholds_local.reshape(-1, 1)

        R = target_vector
        answer = -np.sum(Rl, axis=1) / np.sum(R) * count_H(Rl) - np.sum(Rr, axis=1) / np.sum(R) * count_H(Rr)
        return answer

    index_sort = np.argsort(feature_vector)
    sort_feature_vector, target_vector = feature_vector[index_sort], target_vector[index_sort]

    thresholds = (np.roll(np.unique(sort_feature_vector), 1) + np.unique(sort_feature_vector))[1:] / 2

    if len(thresholds) == 0:
        return [], [], -np.inf, -np.inf

    gini = count_gini(thresholds)

    best_index = np.argmax(gini)

    threshold_best = thresholds[best_index]
    gini_best = gini[best_index]

    return thresholds, gini, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):  # 2
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self.min_samples_split is not None and len(sub_y) <= self.min_samples_split:  # 4
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):  # 3
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                        ratio[key] = current_count / current_click
                    else:
                        ratio[key] = 0
                sorted_categories = sorted(ratio.keys(), key=lambda x: ratio[x])
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])  # 1

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        else:
            feature_split = node["feature_split"]
            if self.feature_types[feature_split] == "real":
                threshold = node["threshold"]
            elif self.feature_types[feature_split] == "categorical":
                threshold = node["categories_split"]
                if x[feature_split] in threshold:
                    return self._predict_node(x, node["left_child"])
                else:
                    return self._predict_node(x, node["right_child"])
            else:
                raise ValueError
            if x[feature_split] < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)