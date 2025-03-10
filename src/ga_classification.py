import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ga_base import GeneticAlgorithm
from models import classification_models
from performance_metric import calc_final_classification

class GeneticAlgorithmRegressor (GeneticAlgorithm):


    def __init__(self, model, cols, population=20, generations=20, cv=5, parents=4):
        super().__init__(model, cols, population, generations, cv, parents)


        self.models = classification_models
        self.scoring = "f1"

        self.create_population()


    def create_model(self, genome):

        model = None

        if self.model == "knn":
            model = KNeighborsClassifier(n_neighbors=genome['n_neighbors'], weights=genome['weights'], p=genome['p'])
        elif self.model == "svc":
            model = LinearSVC(penalty=genome["penalty"], loss=genome["loss"], dual=genome["dual"], tol=genome["tol"], C=genome["C"])
        elif self.model == "logistic":
            model = LogisticRegression(penalty=genome["penalty"], C=genome["C"], dual=genome["dual"])
        elif self.model == "rf":
            model = RandomForestClassifier(n_estimators=genome["n_estimators"], criterion=genome["criterion"], max_features=genome["max_features"], min_samples_split=genome["min_samples_split"], min_samples_leaf=genome["min_samples_leaf"], bootstrap=genome["bootstrap"])
        elif self.model == "decision_tree":
            model = DecisionTreeClassifier(criterion=genome["criterion"], max_depth=genome["max_depth"], min_samples_split=genome["min_samples_split"], min_samples_leaf=genome["min_samples_leaf"])
        
        return model
    

    def predict(self, X_test, y_test):

        X_train = self.clean_input_data(self.X_train, self.best_genome)
        X_test = self.clean_input_data(X_test, self.best_genome)

        model = self.create_model(self.best_genome["model"])

        model.fit(X_train, self.y_train)
        predictions = model.predict(X_test)

        calc_final_classification(y_test, predictions)