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

    def __init__(self, model, cols, population=20, generations=20, cv=5, parents=4):

        self.model = model
        self.cols = cols
        self.population_size = population
        self.generations = generations
        self.cv = cv
        self.parents = parents

        self.population = []
        self.X_train = None
        self.y_train = None
        self.best_genome = None

        
    def create_genome(self):

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

        raise NotImplementedError("create_model method must be implemented in subclass")


    def create_population(self):
        for _ in range(self.population_size):
            genome = self.create_genome()
            self.population.append(genome)


    def fit(self, X_train, y_train):

        self.X_train = X_train  
        self.y_train = y_train

        for generation in range(self.generations):

            sorted_population = self.sort_by_fitness()

            if generation != self.generations - 1:
                parents = self.create_parents(sorted_population)
                crossovers = self.create_crossovers(parents)
                parents.extend(crossovers)
                mutations = self.create_mutations(parents)
                parents.extend(mutations)
                self.population = parents
            else:
                if generation % 5 == 0:
                    print()
                    print(f"generation {generation}")
                self.best_genome = sorted_population[0]
                print("best drop_margins {0}".format(self.best_genome["drop_margins"]))
                print("best cleaners {0}".format(self.best_genome["cleaners"]))
                print("best model {0}".format(self.best_genome["model"]))

            
    def create_parents(self, population):
        parents = []

        while len(parents) < self.parents:
            new_parent = choices(population, k=1, weights=[exp(-x*0.1) for x in range(self.population_size)])[0]
            present = 0
            for parent in parents:
                if np.array_equal(new_parent["drop_margins"], parent["drop_margins"]) and new_parent["cleaners"] == parent["cleaners"] and new_parent["model"] == parent["model"]:
                    present = 1
            if not present:
                parents.append(new_parent)

        return parents
    

    def create_crossovers(self, parents):

        crossovers = []
        parent_posses = [i for i in range(len(parents))]
        pairs = list(itertools.combinations(parent_posses, 2))

        for pair in pairs:

            parent_a = parents[pair[0]]
            parent_b = parents[pair[1]]

            alternator = 0

            drop_margins_a = []
            drop_margins_b = []

            cleaners_a = {}
            cleaners_b = {}

            model_a = {}
            model_b = {}

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

            child_a = {"drop_margins": drop_margins_a, "cleaners": cleaners_a, "model": model_a}
            child_b = {"drop_margins": drop_margins_b, "cleaners": cleaners_b, "model": model_b}

            crossovers.append(child_a)
            crossovers.append(child_b)

        return crossovers
    

    def create_mutations(self, population, n_mutations=2):

        mutations = []

        while len(population) + len(mutations) < self.population_size:
            genome = population[randint(0, len(population) - 1)].copy()

            for _ in range(n_mutations):
                segment = randint(0, len(genome)-1)

                if segment == 0:
                    index = randint(0, len(genome["drop_margins"]) - 1)
                    genome["drop_margins"][index] = not genome["drop_margins"][index]

                elif segment == 1:
                    index = randint(0, len(genome["cleaners"]) - 1)
                    key = list(genome["cleaners"].keys())[index]
                    new_value = choices(cleaners[key], k=1)[0]
                    genome["cleaners"][key] = new_value

                elif segment == 2:
                    index = randint(0, len(genome["model"]) - 1)
                    key = list(genome["model"].keys())[index]
                    new_value = choices(self.models[self.model][key], k=1)[0]
                    genome["model"][key] = new_value

                    
            mutations.append(genome)

        return mutations

                    
    def sort_by_fitness(self):
        
        fitnesses = {}
        repositioned = []

        for genome_pos in range(self.population_size):
            fitness = self.calc_fitness(self.population[genome_pos])
            fitnesses[str(genome_pos)] = fitness

        print(f"Best fitness: {max(fitnesses.values())}")

        for _ in range(self.population_size):
            max_key = max(fitnesses, key=fitnesses.get)
            repositioned.append(self.population[int(max_key)].copy())
            fitnesses.pop(max_key)
      
        return repositioned
    

    def scale_data(self, data, scaler, pca):

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

        data = np.asarray(data)
        data = data[:,np.asarray(genome["drop_margins"]).astype('bool')]

        data = self.scale_data(data, genome["cleaners"]["scaler"], genome["cleaners"]["pca"])

        return data

    def calc_fitness(self, genome):
        
        X_train = self.clean_input_data(self.X_train, genome)

        model = self.create_model(genome["model"])

        scores = cross_val_score(estimator=model, X=X_train, y=self.y_train, cv=self.cv, scoring=self.scoring)
        
        return mean(scores)
    

    def predict(self, X_test, y_test):

        raise NotImplementedError("predict method must be implemented in subclass")


if __name__=='__main__':

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