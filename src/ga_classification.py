import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from ga_base import GeneticAlgorithm
from models import classification_models
from performance_metric import calc_final_classification
from typing import Dict, Any

class GeneticAlgorithmClassifier(GeneticAlgorithm):
    """
        Genetic Algorithm for classification problems
        Inherits from base genetic algorithm class
    """

    def __init__(self, model: str, cols: int, population: int = 20, generations: int = 20, cv: int = 5, parents: int = 4) -> None:
        """
            Model initialisation
            Inherits from the parent class initialisation, as well as introducing classification-specific parameters for the models used, and scoring metric
            Creates an initial population for the first generation

            Args:
                model (str): Model to be used
                cols (int): Number of columns in the dataset
                population (int): Number of individuals in the population
                generations (int): Number of generations
                cv (int): Number of cross-validation splits
                parents (int): Number of parents to be selected for the next generation
        """
        super().__init__(model, cols, population, generations, cv, parents)

        self.models = classification_models
        self.scoring = "f1"

        self.create_population()

    def create_model(self, genome: Dict[str, Any]) -> BaseEstimator:
        """
            Create a model based on the genome
            overrides the parent class method for classification-specific problems

            Args:
                genome (dict): Genome of the individual

            Returns:
                model: ML model to be used
        """
        model: BaseEstimator = None

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

    def predict(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
            Make predictions based on the best genome
            overrides the parent class method for classification-specific problems
            Prints the final performance metrics to terminal, and creates confusion matrix

            Args:
                X_test (pd.DataFrame): Test dataset
                y_test (pd.Series): Test labels
        """
        X_train: np.ndarray = self.clean_input_data(self.X_train, self.best_genome)
        X_test: np.ndarray = self.clean_input_data(X_test, self.best_genome)

        model: BaseEstimator = self.create_model(self.best_genome["model"])

        model.fit(X_train, self.y_train)
        predictions: np.ndarray = model.predict(X_test)

        calc_final_classification(y_test, predictions)


if __name__ == '__main__':
    data_path: str = r"/home/david/Documents/git/ga_remake/data/dataset_phishing_reduced.csv"
    df: pd.DataFrame = pd.read_csv(data_path)

    # mapping = {'phishing': 1, 'legitimate': 0}
    # column = df['status'].map(mapping)
    # df['status'] = column

    labels: pd.Series = df["status"]
    data: pd.DataFrame = df.drop("status", axis=1)

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    print("Compiled")

    genetic_algorithm: GeneticAlgorithmClassifier = GeneticAlgorithmClassifier("rf", cols=X_train.shape[1], parents=4, population=10, generations=10, cv=2)
    genetic_algorithm.fit(X_train, y_train)
    print("predicting")
    genetic_algorithm.predict(X_test, y_test)