import numpy as np
from parms import *
from matplotlib import pyplot as plt
from icecream import ic
from vis import get_data


class GA:
    def __init__(self, file_path):
        self.n = N
        self.cities = get_data(file_path)
        self.n_sets = N_SETS
        self.n_parents = N_PARENTS
        self.n_crossover = N_CROSSOVER
        self.crossover_point = CROSSOVER_POINT
        self.n_mutations = N_MUTATIONS
        self.iter = ITER
        self.mute_per = MUTE_PER
        self.curr_pos = self.init_pos()

    def init_pos(self):
        q = []
        for i in range(self.n_sets):
            idxs = np.random.permutation(self.n) # .reshape((self.n, 1))
            q.append(idxs)
        return np.array(q)

    def one_fitness(self, idxs):
        def dis(c1, c2):
            return np.linalg.norm([c2[0] - c1[0], c2[1] - c1[1]])
        cost = 0
        cities = np.array([self.cities[i] for i in idxs])
        for i in range(cities.shape[0] - 1):
            cost += dis(cities[i], cities[i + 1])
        cost += dis(cities[0], cities[-1])
        return cost

    def fitness(self, idxs):
        q = []
        for _idxs in idxs:
            q.append(self.one_fitness(_idxs))
        return np.array(q)

    def get_parents(self, fitness):
        idx_sort_arr = np.argsort(fitness)
        idx_sort_arr = idx_sort_arr[0 : self.n_parents]
        return np.array([self.curr_pos[i] for i in idx_sort_arr])

    def get_val(self, arr, n):
        cross_idx_1 = np.random.randint(0, n)
        val = arr[cross_idx_1]
        arr.pop(cross_idx_1)
        return arr, val

    def indexes(self, n, length):
        arr = [i for i in range(n)]
        mute_idxs = []
        for i in range(length):
            idx = np.random.randint(0, n - i)
            mute_idxs.append(arr[idx])
            arr.pop(idx)
        return np.array(mute_idxs)

    def mutation(self, parents):
        mutation_offspring = np.zeros((self.n_crossover, self.n))
        for i in range(self.n_crossover):
            mutation_offspring[i] = self.mute_shift(parents[i % self.n_parents])
            mutation_offspring[i] = self.mute_switch(mutation_offspring[i])
        return mutation_offspring

    def mute_switch(self, cities):
        for _ in range(self.n_mutations):
            idx_1 = np.random.randint(0, self.n - 1)
            idx_2 = np.random.randint(0, self.n - 1)
            tmp = cities[idx_1]
            cities[idx_1] = cities[idx_2]
            cities[idx_2] = tmp
        return cities

    def mute_shift(self, cities):
        roll_dir = np.random.randint(0, self.n)
        cities = np.roll(cities, roll_dir)
        return cities

    def run(self):
        history = []
        for _ in range(self.iter):
            fitness = self.fitness(self.curr_pos)
            parents = self.get_parents(fitness)
            mutation_offspring = self.mutation(parents)
            # set to next iteration
            self.curr_pos[: self.n_parents] = parents
            self.curr_pos[self.n_parents:] = mutation_offspring
            # save best ones
            history.append(fitness[0])
            best_result = parents[0]
            ic(fitness[0])
        # self.show_results(history)
        best_result = np.append(best_result, best_result[0])
        route = []
        for i in range(best_result.shape[0] - 1):
            idx_1 = best_result[i]
            idx_2 = best_result[i + 1]
            route.append([self.cities[idx_1], self.cities[idx_2]])
        ic(route)
        return np.array(route)

    def show_results(self, history):
        history = np.array(history)
        x = np.array([idx for idx, _ in enumerate(history)])
        plt.plot(x, history)
        plt.show()
        print("Best result: {:.2f}".format(history[-1]))
