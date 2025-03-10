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
from sklearn.base import BaseEstimator
from typing import List, Dict, Any, Tuple

from models import cleaners
from data_cleaning import clean_input_data
from model_making import create_genome


class GeneticAlgorithm:
    """
        Base model class for the genetic algorithm
        Contains the core functions for the genetic algorithm, including creating the initial population,
        Intended to be over-ridden by classifier and regressor models 
    """
    def __init__(self, model: str, cols: int, population: int, generations: int, cv: int, parents: int) -> None:
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
            genome: Dict[str, Any] = create_genome(self.models[self.model], self.cols)
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
                print("best drop_margins {0}".format(self.best_genome["filters"]["drop_margins"]))
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
                comparison: List[bool] = []
                for segment in new_parent:
                    for key in new_parent[segment]:
                        if isinstance(new_parent[segment][key], np.ndarray):
                            result = np.array_equal(new_parent[segment][key], parent[segment][key])
                            if result == np.True_ or result == True:
                                comparison.append(True)
                            else:
                                comparison.append(False)
                        else:
                            result = new_parent[segment][key] == parent[segment][key]
                            if result == np.True_ or result == True:
                                comparison.append(True)
                            else:
                                comparison.append(False)
                if all(comparison):
                    present = 1
                    break
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

            child_a: Dict[str, Any] = {}
            child_b: Dict[str, Any] = {}

            alternator: int = randint(0, 1)

            for segment in parent_a:
                new_seg_a: Dict[str, Any] = {}
                new_seg_b: Dict[str, Any] = {}
                for key in parent_a[segment]:
                    if alternator:
                        new_seg_a[key] = parent_a[segment][key]
                        new_seg_b[key] = parent_b[segment][key]
                    else:
                        new_seg_a[key] = parent_b[segment][key]
                        new_seg_b[key] = parent_a[segment][key]
                    alternator = not alternator

                child_a[segment] = new_seg_a
                child_b[segment] = new_seg_b

            crossovers.append(child_a)
            crossovers.append(child_b)

        return crossovers

    def get_genome_dict(self, segment: str, key: str, drop_margins: List[int]) -> Dict[str, Any]:
        """
            Get the dictionary for a given segment of the genome

            Args:
                segment (str): Segment of the genome to get the dictionary for

            Returns:
                Dict[str, Any]: Dictionary for the segment
        """
        if segment == "filters":
            index: int = randint(0, len(drop_margins) - 1)
            drop_margins[index] = not drop_margins[index]
            return drop_margins
        elif segment == "cleaners":
            return choices(cleaners[key], k=1)[0]
        elif segment == "model":
            return choices(self.models[self.model][key], k=1)[0]

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

            # todo: generalise this
            for _ in range(n_mutations):
                segment_int: int = randint(0, len(genome)-1)
                segment: str = list(genome.keys())[segment_int]

                index: int = randint(0, len(genome[segment]) - 1)
                key: str = list(genome[segment].keys())[index]

                new_value: Any = self.get_genome_dict(segment, key, genome["filters"]["drop_margins"])
                genome[segment][key] = new_value

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
        X_train: np.ndarray = clean_input_data(self.X_train, genome, self.y_train)
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