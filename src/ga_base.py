import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from random import choices, randint
from statistics import mean
from copy import copy
from math import exp
from joblib import parallel, delayed

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score


from models import cleaners


class GeneticAlgorithm:
    """
        Base model class for the genetic algorithm
        Contains the core functions for the genetic algorithm, including creating the initial population,
        Intended to be over-ridden by classifier and regressor models 
    """

    def __init__(self, model, cols, population=20, generations=20, cv=5, parents=4):
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

        self.model = model
        self.cols = cols
        self.population_size = population
        self.generations = generations
        self.cv = cv
        self.parents = parents

        # Unlabelled variables that appear during fitting
        self.population = []
        self.X_train = None
        self.y_train = None
        self.best_genome = None

        
    def create_genome(self):
        """
            Create a random genome for the genetic algorithm
            Randomly select drop margins, cleaners and model parameters

            Args:
                None

            Returns:
                genome (dict): Randomised genome from the available models and cleaning
        """

        genome = {}
        clean = {}

        for key in self.models[self.model]:
            value = choices(self.models[self.model][key], k=1)[0]
            genome[key] = value

        for cleaner in cleaners:
            value = choices(cleaners[cleaner], k=1)[0]
            clean[cleaner] = value

        drop_margins = np.array([randint(0, 1) for _ in range(self.cols)])
        if sum(drop_margins) < 2:
            a = randint(0, len(drop_margins) - 1)
            b = randint(0, len(drop_margins) - 1)
            while b == a:
                b = randint(0, len(drop_margins) - 1)
            drop_margins[a] = 1
            drop_margins[b] = 1
            drop_margins = np.asarray(drop_margins)

        return {"drop_margins": drop_margins, "cleaners": clean, "model": genome}


    def create_model(self, genome):
        """
            Placeholder function for creating ML models for each genome
            This is intended to be over-ridden within the classifer and regression classes
        """

        raise NotImplementedError("create_model method must be implemented in subclass")


    def create_population(self):
        """
            Create the initial population of the genetic algorithm
            For each genome, create a random set of drop margins, cleaners and model parameters
        """
        for _ in range(self.population_size):
            genome = self.create_genome()
            self.population.append(genome)


    def fit(self, X_train, y_train):
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

            sorted_population = self.sort_by_fitness()

            if generation != self.generations - 1:

                if generation % 5 == 0:
                    print()
                    print(f"generation {generation}")
                    print()

                # Create the next generation
                new_generation = self.create_parents(sorted_population)
                new_generation.extend(self.create_crossovers(new_generation))
                new_generation.extend(self.create_mutations(new_generation))
                self.population = new_generation

            else:
                
                # Select best performing genome from the last generation
                self.best_genome = sorted_population[0]
                print("best drop_margins {0}".format(self.best_genome["drop_margins"]))
                print("best cleaners {0}".format(self.best_genome["cleaners"]))
                print("best model {0}".format(self.best_genome["model"]))

            
    def create_parents(self, population):
        """
            Create a list of parents for the next generation
            Parents are selected based on the fitness of the genome
            The probability of selection is determined by the exponential of the genome's fitness

            Args:
                population (list): Population to select parents from

            Returns:
                parents (list): List of parents for the next generation
        """
        
        parents = []

        while len(parents) < self.parents:
            new_parent = choices(population, k=1, weights=[exp(-x*0.1) for x in range(self.population_size)])[0]
            present = 0

            # See if the parent is already within the list, if not, append it
            for parent in parents:
                if np.array_equal(new_parent["drop_margins"], parent["drop_margins"]) and new_parent["cleaners"] == parent["cleaners"] and new_parent["model"] == parent["model"]:
                    present = 1
                    break

            if not present:
                parents.append(new_parent)

        return parents
    

    def create_crossovers(self, parents):
        crossovers = []
        parent_posses = range(len(parents))
        pairs = itertools.combinations(parent_posses, 2)

        for pair in pairs:
            parent_a = parents[pair[0]]
            parent_b = parents[pair[1]]

            drop_margins_a, drop_margins_b = [], []
            cleaners_a, cleaners_b = {}, {}
            model_a, model_b = {}, {}

            for i, (dm_a, dm_b) in enumerate(zip(parent_a["drop_margins"], parent_b["drop_margins"])):
                if i % 2 == 0:
                    drop_margins_a.append(dm_a)
                    drop_margins_b.append(dm_b)
                else:
                    drop_margins_a.append(dm_b)
                    drop_margins_b.append(dm_a)

            for cleaner in parent_a["cleaners"]:
                if cleaner in parent_b["cleaners"]:
                    if i % 2 == 0:
                        cleaners_a[cleaner] = parent_a["cleaners"][cleaner]
                        cleaners_b[cleaner] = parent_b["cleaners"][cleaner]
                    else:
                        cleaners_a[cleaner] = parent_b["cleaners"][cleaner]
                        cleaners_b[cleaner] = parent_a["cleaners"][cleaner]

            for key in parent_a["model"]:
                if key in parent_b["model"]:
                    if i % 2 == 0:
                        model_a[key] = parent_a["model"][key]
                        model_b[key] = parent_b["model"][key]
                    else:
                        model_a[key] = parent_b["model"][key]
                        model_b[key] = parent_a["model"][key]

            child_a = {"drop_margins": drop_margins_a, "cleaners": cleaners_a, "model": model_a}
            child_b = {"drop_margins": drop_margins_b, "cleaners": cleaners_b, "model": model_b}

            crossovers.append(child_a)
            crossovers.append(child_b)

        return crossovers
    

    def create_mutations(self, population, n_mutations=2):
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

        mutations = []

        # Loop to iterate until desired population size is reached
        while len(population) + len(mutations) < self.population_size:

            genome = population[randint(0, len(population) - 1)].copy()

            for _ in range(n_mutations):

                # Select a random segment of the genome to mutate
                segment = randint(0, len(genome)-1)

                # Mutate feature mask
                if segment == 0:
                    index = randint(0, len(genome["drop_margins"]) - 1)
                    genome["drop_margins"][index] = not genome["drop_margins"][index]

                # Mutate the scaler or PCA
                elif segment == 1:
                    index = randint(0, len(genome["cleaners"]) - 1)
                    key = list(genome["cleaners"].keys())[index]
                    new_value = choices(cleaners[key], k=1)[0]
                    genome["cleaners"][key] = new_value

                # Mutate the ML model
                elif segment == 2:
                    index = randint(0, len(genome["model"]) - 1)
                    key = list(genome["model"].keys())[index]
                    new_value = choices(self.models[self.model][key], k=1)[0]
                    genome["model"][key] = new_value

            mutations.append(genome)

        return mutations

                    
    def sort_by_fitness(self):
        """
            Sort the population by fitness, and return the repositioned population
            Fitness is determined by the GA's given metric

            Args:
                None

            Returns:
                repositioned (list): Repositioned population
        """
        
        fitnesses = {}
        repositioned = []

        # Calculate fitness for each genome in the population
        for genome_pos in range(self.population_size):
            fitness = self.calc_fitness(self.population[genome_pos])
            fitnesses[str(genome_pos)] = fitness

        print(f"Best fitness: {max(fitnesses.values())}")

        # Create a repositioned list of the population, by order of fitness
        for _ in range(self.population_size):
            max_key = max(fitnesses, key=fitnesses.get)
            repositioned.append(self.population[int(max_key)].copy())
            fitnesses.pop(max_key)
      
        return repositioned
    

    def scale_data(self, data, scaler, pca):
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
    

    def clean_input_data(self, data, genome):
        """
            Clean input data according to the specifications of the genome, and return as array

            Args:
                data (pd.DataFrame): Input data
                genome (dict): Genome containing the specifications for the data cleaning

            Returns:
                data (np.ndarray): Cleaned data
        """

        data = np.asarray(data)
        data = data[:,np.asarray(genome["drop_margins"]).astype('bool')]

        data = self.scale_data(data, genome["cleaners"]["scaler"], genome["cleaners"]["pca"])

        return data
    

    def calc_fitness(self, genome):
        """
            Calculate the fitness of a genome
            Clean the input data and create the ML model based upon the genome's specifications
            Perform cross-validation and return the mean score

            Args:
                genome (dict): Genome containing the specifications for the model and data cleaning

            Returns:
                float: Mean cross-validation score
        """
        
        X_train = self.clean_input_data(self.X_train, genome)

        model = self.create_model(genome["model"])

        scores = cross_val_score(estimator=model, X=X_train, y=self.y_train, cv=self.cv, scoring=self.scoring)
        
        return mean(scores)
    

    def predict(self, X_test, y_test):
        """
            Placeholder function for predicting over the test dataset
            Intended to be overwritten within the classifier and regression classes

            Args:
                X_test (pd.DataFrame): Test dataset
                y_test (pd.Series): Test labels
        """

        raise NotImplementedError("predict method must be implemented in subclass")
