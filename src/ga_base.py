import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from random import choices, randint
from statistics import mean
from copy import copy
from math import exp
from joblib import Parallel, delayed

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score

from models import cleaners
from sklearn.base import BaseEstimator
from typing import List, Dict, Any, Tuple


class GeneticAlgorithm:
    """
        Base model class for the genetic algorithm
        Contains the core functions for the genetic algorithm, including creating the initial population,
        Intended to be over-ridden by classifier and regressor models 
    """
    def __init__(self, model: str, cols: int, population: int = 20, generations: int = 20, cv: int = 5, parents: int = 4) -> None:
        """
            Initialize the genetic algorithm with the given parameters

            Args:
                model (str): Model to use for the genetic algorithm
                cols (int): Number of columns in the dataset
                population (int): Number of genomes in the population
                generations (int): Number of generations to run the algorithm
                cv (int): Number of cross-validation splits
                parents (int): Number of parents to select for the next
        """
        self.model: str = model
        self.cols: int = cols
        self.population_size: int = population
        self.generations: int = generations
        self.cv: int = cv
        self.parents: int = parents

        self.population: List[Dict[str, Any]] = []
        self.X_train: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.best_genome: Dict[str, Any] = None

    def create_genome(self) -> Dict[str, Any]:
        """
            Create a random genome for the genetic algorithm
            Randomly select drop margins, cleaners and model parameters

            Args:
                None

            Returns:
                genome (Dict[str, Any]): Randomised genome from the available models and cleaning
        """
        genome: Dict[str, Any] = {}
        clean: Dict[str, Any] = {}

        for key in self.models[self.model]:
            value: Any = choices(self.models[self.model][key], k=1)[0]
            genome[key] = value

        for cleaner in cleaners:
            value: Any = choices(cleaners[cleaner], k=1)[0]
            clean[cleaner] = value

        drop_margins: np.ndarray = np.array([randint(0, 1) for _ in range(self.cols)])
        if sum(drop_margins) < 2:
            a: int = randint(0, len(drop_margins) - 1)
            b: int = randint(0, len(drop_margins) - 1)
            while b == a:
                b = randint(0, len(drop_margins) - 1)
            drop_margins[a] = 1
            drop_margins[b] = 1
            drop_margins = np.asarray(drop_margins)

        return {"drop_margins": drop_margins, "cleaners": clean, "model": genome}

    def create_model(self, genome: Dict[str, Any]) -> BaseEstimator:
        """
            Placeholder function for creating ML models for each genome
            This is intended to be over-ridden within the classifier and regression classes
        """
        raise NotImplementedError("create_model method must be implemented in subclass")

    def create_population(self) -> None:
        """
            Create the initial population of the genetic algorithm
            For each genome, create a random set of drop margins, cleaners and model parameters
        """
        for _ in range(self.population_size):
            genome: Dict[str, Any] = self.create_genome()
            self.population.append(genome)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
            Fit the genetic algorithm to the given data
            Create the initial population, and iterate over the generations
            For each generation, sort the population by fitness, create parents, crossovers and mutations
            The final genome is the best genome from the last generation

            Args:
                X_train (pd.DataFrame): Training dataset
                y_train (pd.Series): Training labels
        """
        self.X_train = X_train
        self.y_train = y_train

        for generation in range(self.generations):
            sorted_population: List[Dict[str, Any]] = self.sort_by_fitness()

            if generation != self.generations - 1:
                if generation % 5 == 0:
                    print()
                    print(f"generation {generation}")
                    print()

                parents: List[Dict[str, Any]] = self.create_parents(sorted_population)
                crossovers: List[Dict[str, Any]] = self.create_crossovers(parents)
                parents.extend(crossovers)
                mutations: List[Dict[str, Any]] = self.create_mutations(parents)
                parents.extend(mutations)
                self.population = parents
            else:
                self.best_genome = sorted_population[0]
                print("best drop_margins {0}".format(self.best_genome["drop_margins"]))
                print("best cleaners {0}".format(self.best_genome["cleaners"]))
                print("best model {0}".format(self.best_genome["model"]))

    def create_parents(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
            Create a list of parents for the next generation
            Parents are selected based on the fitness of the genome
            The probability of selection is determined by the exponential of the genome's fitness

            Args:
                population (list): Population to select parents from

            Returns:
                parents (list): List of parents for the next generation
        """
        parents: List[Dict[str, Any]] = []

        while len(parents) < self.parents:
            new_parent: Dict[str, Any] = choices(population, k=1, weights=[exp(-x*0.1) for x in range(self.population_size)])[0]
            present: int = 0
            for parent in parents:
                if np.array_equal(new_parent["drop_margins"], parent["drop_margins"]) and new_parent["cleaners"] == parent["cleaners"] and new_parent["model"] == parent["model"]:
                    present = 1
            if not present:
                parents.append(new_parent)

        return parents

    def create_crossovers(self, parents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
            Create crossovers between pairs of parents to generate offspring

            Args:
                parents (list): List of parents

            Returns:
                crossovers (list): List of offspring generated from crossovers
        """
        crossovers: List[Dict[str, Any]] = []
        parent_posses: List[int] = [i for i in range(len(parents))]
        pairs: List[Tuple[int, int]] = list(itertools.combinations(parent_posses, 2))

        for pair in pairs:
            parent_a: Dict[str, Any] = parents[pair[0]]
            parent_b: Dict[str, Any] = parents[pair[1]]

            alternator: int = 0

            drop_margins_a: List[int] = []
            drop_margins_b: List[int] = []

            cleaners_a: Dict[str, Any] = {}
            cleaners_b: Dict[str, Any] = {}

            model_a: Dict[str, Any] = {}
            model_b: Dict[str, Any] = {}

            for i in range(len(parent_a["drop_margins"])):
                if alternator:
                    drop_margins_a.append(parent_a["drop_margins"][i])
                    drop_margins_b.append(parent_b["drop_margins"][i])
                else:
                    drop_margins_a.append(parent_b["drop_margins"][i])
                    drop_margins_b.append(parent_a["drop_margins"][i])
                alternator = not alternator

            for cleaner in parent_a["cleaners"]:
                if alternator:
                    cleaners_a[cleaner] = parent_a["cleaners"][cleaner]
                    cleaners_b[cleaner] = parent_b["cleaners"][cleaner]
                else:
                    cleaners_a[cleaner] = parent_b["cleaners"][cleaner]
                    cleaners_b[cleaner] = parent_a["cleaners"][cleaner]
                alternator = not alternator

            for key in parent_a["model"]:
                if alternator:
                    model_a[key] = parent_a["model"][key]
                    model_b[key] = parent_b["model"][key]
                else:
                    model_a[key] = parent_b["model"][key]
                    model_b[key] = parent_a["model"][key]
                alternator = not alternator

            child_a: Dict[str, Any] = {"drop_margins": drop_margins_a, "cleaners": cleaners_a, "model": model_a}
            child_b: Dict[str, Any] = {"drop_margins": drop_margins_b, "cleaners": cleaners_b, "model": model_b}

            crossovers.append(child_a)
            crossovers.append(child_b)

        return crossovers

    def create_mutations(self, population: List[Dict[str, Any]], n_mutations: int = 2) -> List[Dict[str, Any]]:
        """
            Create mutated offspring for the given population
            Until the required population size is reached, keep appending mutations
            For each n_mutation, randomly select a segment of the genome and mutate it

            Args:
                population (list): Population to mutate
                n_mutations (int): Number of mutations to perform

            Returns:
                mutations (list): Mutated offspring
        """
        mutations: List[Dict[str, Any]] = []

        while len(population) + len(mutations) < self.population_size:
            genome: Dict[str, Any] = population[randint(0, len(population) - 1)].copy()

            for _ in range(n_mutations):
                segment: int = randint(0, len(genome)-1)

                if segment == 0:
                    index: int = randint(0, len(genome["drop_margins"]) - 1)
                    genome["drop_margins"][index] = not genome["drop_margins"][index]

                elif segment == 1:
                    index: int = randint(0, len(genome["cleaners"]) - 1)
                    key: str = list(genome["cleaners"].keys())[index]
                    new_value: str = choices(cleaners[key], k=1)[0]
                    genome["cleaners"][key] = new_value

                elif segment == 2:
                    index: int = randint(0, len(genome["model"]) - 1)
                    key: str = list(genome["model"].keys())[index]
                    new_value: str = choices(self.models[self.model][key], k=1)[0]
                    genome["model"][key] = new_value

            mutations.append(genome)

        return mutations

    def sort_by_fitness(self) -> List[Dict[str, Any]]:
        """
            Sort the population by fitness, and return the repositioned population
            Fitness is determined by the GA's given metric

            Args:
                None

            Returns:
                repositioned (list): Repositioned population
        """
        fitnesses: Dict[str, float] = {}
        repositioned: List[Dict[str, Any]] = []

        for genome_pos in range(self.population_size):
            fitness: float = self.calc_fitness(self.population[genome_pos])
            fitnesses[str(genome_pos)] = fitness

        print(f"Best fitness: {max(fitnesses.values())}")

        for _ in range(self.population_size):
            max_key: str = max(fitnesses, key=fitnesses.get)
            repositioned.append(self.population[int(max_key)].copy())
            fitnesses.pop(max_key)

        return repositioned

    def scale_data(self, data: np.ndarray, scaler: str, pca: str) -> np.ndarray:
        """
            Scale input data according to the required scaler and PCA

            Args:
                data (np.ndarray): Input data
                scaler (str): Scaler to use
                pca (str): Whether to use PCA

            Returns:
                data (np.ndarray): Scaled data
        """
        if scaler == "robust":
            scaler = RobustScaler()
            data = scaler.fit_transform(data)
        elif scaler == "standard":
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        elif scaler == "minmax":
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)

        if pca == "pca":
            pca = PCA(n_components="mle")
            data = pca.fit_transform(data)

        return data

    def clean_input_data(self, data: pd.DataFrame, genome: Dict[str, Any]) -> np.ndarray:
        """
            Clean input data according to the specifications of the genome, and return as array

            Args:
                data (pd.DataFrame): Input data
                genome (Dict[str, Any]): Genome containing the specifications for the data cleaning

            Returns:
                data (np.ndarray): Cleaned data
        """
        data: np.ndarray = np.asarray(data)
        data = data[:, np.asarray(genome["drop_margins"]).astype('bool')]
        data = self.scale_data(data, genome["cleaners"]["scaler"], genome["cleaners"]["pca"])
        return data

    def calc_fitness(self, genome: Dict[str, Any]) -> float:
        """
            Calculate the fitness of a genome
            Clean the input data and create the ML model based upon the genome's specifications
            Perform cross-validation and return the mean score

            Args:
                genome (Dict[str, Any]): Genome containing the specifications for the model and data cleaning

            Returns:
                float: Mean cross-validation score
        """
        X_train: np.ndarray = self.clean_input_data(self.X_train, genome)
        model: BaseEstimator = self.create_model(genome["model"])
        scores: List[float] = cross_val_score(estimator=model, X=X_train, y=self.y_train, cv=self.cv, scoring=self.scoring)
        return mean(scores)

    def predict(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
            Placeholder function for predicting over the test dataset
            Intended to be overwritten within the classifier and regression classes

            Args:
                X_test (pd.DataFrame): Test dataset
                y_test (pd.Series): Test labels
        """
        raise NotImplementedError("predict method must be implemented in subclass")


if __name__ == '__main__':
    data_path = r"data/dataset_phishing_reduced.csv"
    df = pd.read_csv(data_path)

    # mapping = {'phishing': 1, 'legitimate': 0}
    # column = df['status'].map(mapping)
    # df['status'] = column

    labels = df["status"]
    data = df.drop("status", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    print("Compiled")

    genetic_algorithm = GeneticAlgorithm("rf", cols=X_train.shape[1], parents=3, population=20, generations=20, cv=2)
    genetic_algorithm.fit(X_train, y_train)
    print("predicting")
    genetic_algorithm.predict(X_test, y_test)