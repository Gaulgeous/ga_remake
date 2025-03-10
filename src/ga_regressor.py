import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from ga_base import GeneticAlgorithm
from models import regression_models
from performance_metric import calc_final_regression

class GeneticAlgorithmRegressor (GeneticAlgorithm):


    def __init__(self, model, cols, population=20, generations=20, cv=5, parents=4):
        super().__init__(model, cols, population, generations, cv, parents)

        self.models = regression_models
        self.scoring = "r2"

        self.create_population()


    def create_model(self, genome):

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
    

    def predict(self, X_test, y_test):

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

    print("Compiled")

    genetic_algorithm = GeneticAlgorithmRegressor("rf", cols=X_train.shape[1], parents=4, population=10, generations=10, cv=2)
    genetic_algorithm.fit(X_train, y_train)
    print("predicting")
    genetic_algorithm.predict(X_test, y_test)