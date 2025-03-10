import numpy as np

classification_models = {"knn": {'n_neighbors': range(1, 101), 'weights': ["uniform", "distance"], 'p': [1, 2]},
          'svc': {'penalty': ["l2"], 'loss': ["hinge", "squared_hinge"], 'dual': [True], 'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]},
          'logistic': {'penalty': ["l2"], 'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.], 'dual': [False]},
          'rf': {'n_estimators': [100], 'criterion': ["gini", "entropy"], 'max_features': np.arange(0.05, 1.01, 0.05), 'min_samples_split': range(2, 21), 'min_samples_leaf':  range(1, 21), 'bootstrap': [True, False]},
          'decision_tree': {'criterion': ["gini", "entropy"], 'max_depth': range(1, 11), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21)}
        }

regression_models = {"rf": {'n_estimators': [100], 'max_features': np.arange(0.05, 1.01, 0.05), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21), 'bootstrap': [True, False]},
                     "knn": {'n_neighbors': range(1, 101), 'weights': ["uniform", "distance"], 'p': [1, 2]},
                     "decision_tree": {'max_depth': range(1, 11), 'min_samples_split': range(2, 21), 'min_samples_leaf': range(1, 21)},
                     "elastic": {'l1_ratio': np.arange(0.0, 1.01, 0.05), 'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
                     }

cleaners = {"scaler": ["minmax", "robust", "standard", "none"], "pca": ["none", "pca"]}

filter_coefficients = np.arange(0.1, 0.9, 0.1)