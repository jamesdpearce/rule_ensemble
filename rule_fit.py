from __future__ import division
from math import floor, log10
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
 
class rule_fit:
 
    def __get_tree_encoding(self, X, tree, nodes):
        tree_encoding = tree.decision_path(X.astype(np.float32))[:,nodes].copy()
        return tree_encoding
 
    def __round_to_n_sigdigits(self, x, n):
        return round(x, -int(floor(log10(abs(x)))) + (n-1))
 
    def __get_rules(self, tree):
 
        stack = [(0, [])]
        rules = {}
 
        while stack:
            node_id, rule = stack.pop()
            feature = self.feature_names[tree.feature[node_id]]
            threshold = self.__round_to_n_sigdigits(tree.threshold[node_id], self.precision)
            child_left, child_right = tree.children_left[node_id], tree.children_right[node_id]
            left_rule = rule + ['{0} <= {1}'.format(feature, threshold)]
            right_rule = rule + ['{0} > {1}'.format(feature, threshold)]
 
            if (child_left != child_right):
                stack.append((child_left, left_rule))
                stack.append((child_right, right_rule))
                if len(rule) != 0:
                    rules[node_id] = ' & '.join(rule)
            else:
                rules[node_id] = ' & '.join(rule)
 
        return rules
 
    def __remove_duplicate_rules(self):
        rule_set = set()
        for rules in self.rule_ensemble.values():
            for idx, rule in rules.items():
                if rule in rule_set:
                    del rules[idx]
            rule_set = rule_set.union(set(rules.values()))
 
    def __get_decision_matrix(self, X):
        mat_to_stack = []
        feature_names = []
        for tree, rules in self.rule_ensemble.items():
            mat_to_stack.append(self.__get_tree_encoding(X, tree, rules.keys()))
            feature_names += rules.values()
        return hstack(mat_to_stack), np.array(feature_names)
 
    def __get_rule_ensemble(self):
        rule_ensemble = {}
        estimators = np.array(self.tree_ensemble.estimators_).flatten()
        for estimator in estimators:
            tree = estimator.tree_
            rule_ensemble[tree] = self.__get_rules(tree)
        return rule_ensemble
 
    def __down_sample_rules(self, X, rules, n_rules):
        if n_rules >= X.shape[1]:
            return X, rules
        random_idx = np.random.choice(range(X.shape[1]), n_rules, replace = False).astype(int)
        return X[:,random_idx].copy(), rules[random_idx].copy()
 
    def __init__(self, tree_ensemble, linear_model, feature_names, linear_feature_idx = [],
                 precision = 2, verbose = 0, regression = False, max_rules = None,
                 prefit_ensemble = False, use_sparse_matrix = True):
        self.tree_ensemble = tree_ensemble
        self.linear_model = linear_model
        self.feature_names = feature_names
        self.linear_feature_idx = linear_feature_idx
        self.linear_feature_names = np.array(feature_names)[linear_feature_idx].tolist()
        self.precision = precision
        self.verbose = verbose
        self.rule_ensemble = None
        self.rules = None
        self.coefs = None
        self.importances = None
        self.max_rules = max_rules
        self.prefit_ensemble = prefit_ensemble
        self.use_sparse_matrix = use_sparse_matrix
        self.regression = regression
 
    def fit(self, X, y):
 
        if not self.prefit_ensemble:
            if self.verbose: print "Fitting tree ensemble"
            self.tree_ensemble.fit(X,y)
 
        if self.verbose: print "Extracting rules"
        self.rule_ensemble = self.__get_rule_ensemble()
        self.__remove_duplicate_rules()
 
        if self.verbose: print "Calculating decision matrix"
        Z, self.rules = self.__get_decision_matrix(X)
        if self.max_rules:
            Z, self.rules = self.__down_sample_rules(Z, self.rules, self.max_rules)
 
        if self.verbose: print "Fitting linear model"
        if not self.use_sparse_matrix:
            Z = Z.toarray()
        if len(self.linear_feature_idx) > 0:
            if self.use_sparse_matrix:
                Z = hstack([Z, csr_matrix(X[:, self.linear_feature_idx])])
            else:
                Z = np.hstack([Z, X[:, self.linear_feature_idx]])
        self.linear_model.fit(Z,y)
 
        coef_index = self.rules.tolist()
        if self.linear_feature_names:
            coef_index += self.linear_feature_names
        self.coefs = pd.Series(self.linear_model.coef_.flatten(), index = coef_index)
 
        self.importances = pd.Series(np.abs(self.coefs.values)*Z.toarray().std(0), index = coef_index)
 
    def predict_proba(self, X):
        if self.verbose: print "Evaluating rules"
        Z, _ = self.__get_decision_matrix(X)
        if len(self.linear_feature_idx) > 0:
            Z = np.hstack([Z, X[:, self.linear_feature_idx]])
 
        if self.verbose: print "Evaluating linear model"
        if self.regression:
            y_hat = self.linear_model.predict(Z)
        else:
            y_hat = self.linear_model.predict_proba(Z)
 
        return y_hat
