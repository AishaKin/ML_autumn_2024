import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    """
    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    
    def calculate_impurity(y):
        if len(y) == 0:
            return 0
        p0 = np.mean(y == 0)
        p1 = 1 - p0
        return 1 - p1 ** 2 - p0 ** 2
    
    sorted_ind = np.argsort(feature_vector)
    R = feature_vector[sorted_ind]
    y = target_vector[sorted_ind]
    
    thresholds = (R[1:] + R[:-1]) / 2
    ginis = np.zeros(len(thresholds))  # Инициализация массива с нулями
    gini_best = -np.inf  # Начальное значение, чтобы гарантировать обновление
    threshold_best = None

    # Основной цикл по всем порогам
    for i, t in enumerate(thresholds):   
        R_l = R[R < t]
        R_r = R[R >= t]

        if len(R_l) == 0 or len(R_r) == 0:  # Пропуск пустых подвыборок
            continue
        
        y_l = y[R < t]
        y_r = y[R >= t]
        
        H_R = calculate_impurity(y)
        H_Rl = calculate_impurity(y_l)
        H_Rr = calculate_impurity(y_r)

        Q = H_R - (len(R_l) / len(R)) * H_Rl - (len(R_r) / len(R)) * H_Rr
        ginis[i] = Q  # Запись значения Джини в массив

        if Q > gini_best:
            gini_best = Q
            threshold_best = t
            
    return thresholds[ginis > 0], ginis[ginis > 0], threshold_best, gini_best


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