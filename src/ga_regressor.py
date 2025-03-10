import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from ga_base import GeneticAlgorithm
from models import regression_models
from performance_metric import calc_final_regression

from typing_extensions import override

class GeneticAlgorithmRegressor (GeneticAlgorithm):
    """
        Genetic Algorithm for regression problems
        Inherits from base genetic algorithm class
    """

    def __init__(self, model, cols, population=20, generations=20, cv=5, parents=4):
        """
            Model initialisation
            Inherits from the parent class initialisation, as well as introducing regression-specific parameters for the models used, and scoring metric
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

        self.models = regression_models
        self.scoring = "r2"

        self.create_population()


    @override
    def create_model(self, genome):
        """
            Create a model based on the genome
            overrides the parent class method for regression-specific problems

            Args:
                genome (dict): Genome of the individual

            Returns:
                model: ML model to be used
        """

        model = None

        if self.model == "rf":
            model = RandomForestRegressor(n_estimators=genome["n_estimators"], max_features=genome["max_features"], min_samples_split=genome["min_samples_split"], min_samples_leaf=genome["min_samples_leaf"], bootstrap=genome["bootstrap"])
        elif self.model == "decision_tree":
            model = DecisionTreeRegressor(max_depth=genome["max_depth"], min_samples_split=genome["min_samples_split"], min_samples_leaf=genome["min_samples_leaf"])
        elif self.model == "elastic":
            model = ElasticNet(l1_ratio=genome["l1_ratio"], tol=genome["tol"])
        elif self.model == "knn":
            model = KNeighborsRegressor(n_neighbors=genome["n_neighbors"], weights=genome["weights"], p=genome["p"])
        
        return model
    

    @override
    def predict(self, X_test, y_test):
        """
            Make predictions based on the best genome
            overrides the parent class method for regression-specific problems
            Prints the final performance metrics to terminal

            Args:
                X_test (DataFrame): Test dataset
                y_test (DataFrame): Test labels
        """

        X_train = self.clean_input_data(self.X_train, self.best_genome)
        X_test = self.clean_input_data(X_test, self.best_genome)

        model = self.create_model(self.best_genome["model"])

        model.fit(X_train, self.y_train)
        predictions = model.predict(X_test)

        calc_final_regression(y_test, predictions)


if __name__=='__main__':

    data_path = r"/home/david/Documents/git/ga_remake/data/dataset_phishing_reduced.csv"
    df = pd.read_csv(data_path)

    # mapping = {'phishing': 1, 'legitimate': 0}
    # column = df['status'].map(mapping)
    # df['status'] = column

    labels = df["status"]
    data = df.drop("status", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    genetic_algorithm = GeneticAlgorithmRegressor("rf", cols=X_train.shape[1], parents=4, population=10, generations=10, cv=2)
    genetic_algorithm.fit(X_train, y_train)
    genetic_algorithm.predict(X_test, y_test)