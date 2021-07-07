import numpy as np


class GA:
    def __init__(self, arm):
        self.arm = arm
        # -------------------
        # data from parms
        self.target = TARGET
        self.n = N
        self.n_sets = N_SETS
        self.n_parents = N_PARENTS
        self.n_crossover = N_CROSSOVER
        self.crossover_point = CROSSOVER_POINT
        self.n_mutations = N_MUTATIONS
        self.iter = ITER
        self.mute_per = MUTE_PER
        self.save_dir = SAVE_DIR
        self.csv_name = CSV_NAME
        # -------------------
        # initialize the position
        # if len(self.target) == 3:
        #     self.curr_pos = self.arm.init_pos()
        # elif len(self.target) == 2:
        self.ang_z = self.arm.cal_ang_z()
        self.target = np.array([np.sqrt(sum([i ** 2 for i in self.target[:2]])), self.target[2]])
        self.curr_pos = self.init_pos()
        self.ss = SS

    def init_pos(self):
        angs = np.zeros((self.n_sets, self.n))
        for iter in range(self.n_sets):
            for idx, ang in enumerate(self.arm.ang_limits):
                ang_tmp = np.random.uniform(low=ang[0], high=ang[1], size=(1,1))
                angs[iter, idx] = ang_tmp
        return angs

    def fitness(self, angs):
        tmp = np.zeros((self.n_sets, 3))
        for idx, _angs in enumerate(angs):
            for (l, ang, _s) in zip(self.arm.links, _angs, self.ss):
                tmp[idx, 2] += ang * _s
                tmp[idx, 0] += l * np.cos(tmp[idx, 2])
                tmp[idx, 1] += l * np.sin(tmp[idx, 2])
        tmp = tmp[:, 0:2] # drops the angles
        return np.array([np.linalg.norm(np.array([i, j]) - self.target) for (i, j) in tmp])

    def get_parents(self, fitness):
        idx_sort_arr = np.argsort(fitness)
        idx_sort_arr = idx_sort_arr[0 : self.n_parents]
        return np.array([self.curr_pos[i] for i in idx_sort_arr])

    def crossover_1(self, parents):
        # possibly simplest crossover method
        tmp = np.zeros((self.n_crossover, self.n))
        for idx in range(self.n_crossover):
            idx_1 = idx % self.n_parents
            idx_2 = (idx + 1) % self.n_parents
            tmp[idx, : self.crossover_point] = parents[idx_1, : self.crossover_point]
            tmp[idx, self.crossover_point:] = parents[idx_2, self.crossover_point:]
        return tmp

    def crossover_2(self, parents):
        """
        "crossover_2" proved to be the best
        """
        # possibly simplest crossover method
        tmp = np.zeros((self.n_crossover, self.n))
        for idx in range(self.n_crossover):
            nums = [i for i in range(self.n_parents)]
            nums, idx_1 = self.get_val(nums, self.n_parents)
            nums, idx_2 = self.get_val(nums, self.n_parents - 1)
            tmp[idx, : self.crossover_point] = parents[idx_1, : self.crossover_point]
            tmp[idx, self.crossover_point:] = parents[idx_2, self.crossover_point:]
        return tmp

    def crossover_3(self, parents):
        tmp = np.zeros((self.n_crossover, self.n))
        for idx in range(self.n_crossover):
            arr = [i for i in range(self.n)]
            idx_1 = idx % self.n_parents
            idx_2 = (idx + 1) % self.n_parents
            for i in range(self.n // 2):
                i *= 2
                arr, cross_idx_1 = self.get_val(arr, self.n - i)
                arr, cross_idx_2 = self.get_val(arr, self.n - i - 1)
                tmp[idx, cross_idx_1] = parents[idx_1, cross_idx_1]
                tmp[idx, cross_idx_2] = parents[idx_2, cross_idx_2]
        return tmp

    def crossover_4(self, parents):
        """ almost the save method as "crossover_3" ---
         --- this time parent indexes are also drawed """
        tmp = np.zeros((self.n_crossover, self.n))
        for idx in range(self.n_crossover):
            arr = [i for i in range(self.n)]
            nums = [i for i in range(self.n_parents)]
            for i in range(self.n // 2):
                i *= 2
                arr, cross_idx_1 = self.get_val(arr, self.n - i)
                arr, cross_idx_2 = self.get_val(arr, self.n - i - 1)
                nums, idx_1 = self.get_val(nums, self.n_parents - i)
                nums, idx_2 = self.get_val(nums, self.n_parents - i - 1)
                tmp[idx, cross_idx_1] = parents[idx_1, cross_idx_1]
                tmp[idx, cross_idx_2] = parents[idx_2, cross_idx_2]
        return tmp

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

    def mutation(self, crossover_offspring):
        for i in range(self.n_crossover):
            crossover_offspring[i] = self.mutation_interval(crossover_offspring[i])
        return crossover_offspring

    def mutation_interval(self, angs):
        # use every mutation
        mute_idxs = self.indexes(self.n, self.n_mutations)
        for i in mute_idxs:
            one = np.sign(self.arm.ang_limits[i, 0])
            two = np.sign(self.arm.ang_limits[i, 1])
            low_val = -abs(angs[i] * self.mute_per)
            high_val = abs(low_val)
            r = np.random.uniform(low=low_val, high=high_val)
            angs[i] += r
            if angs[i] < self.arm.ang_limits[i, 0] * one:
                angs[i] = self.arm.ang_limits[i, 0]
            if angs[i] > self.arm.ang_limits[i, 1] * two:
                angs[i] = self.arm.ang_limits[i, 1]
        return angs

    def run(self):
        history = []
        for _ in range(self.iter):
            fitness = self.fitness(self.curr_pos)
            # ===================================================
            # if check_valid(self.curr_pos, self.target, LINKS, BOUND, self.ang_z):
            #     ic(True)
            # ===================================================
            parents = self.get_parents(fitness)
            crossover_offspring = self.crossover_2(parents)
            mutation_offspring = self.mutation(crossover_offspring)
            # set to next iteration
            self.curr_pos[: self.n_parents] = parents
            self.curr_pos[self.n_parents:] = mutation_offspring
            history.append(fitness[0])
            best_result = fitness[0]
            if _ == self.iter - 1:
                best_result = parents[0]
                # ic(parents[0])
        # ic("{:.2f}".format(history[-1]))
        # self.show_results(history)
        # self.save2csv(history)

        # ======================================================================
        # TEST
        # tmp_x, tmp_z, tmp_ang = 0, 0, 0
        # for (l, ang, _s) in zip(LINKS, parents[0], self.ss):
        #     tmp_ang += ang * _s
        #     tmp_x += l * cos(tmp_ang)
        #     tmp_z += l * sin(tmp_ang)
        # curr_target = [tmp_x * cos(self.ang_z), tmp_x * sin(self.ang_z), tmp_z]
        # ic(curr_target)
        # self.target = [self.target[0] * cos(self.ang_z), self.target[0] * sin(self.ang_z), self.target[1]]
        # # if check_self_cross(points):
        # #     return True
        # ======================================================================
        if check_valid([parents[0]], self.target, LINKS, BOUND, self.ang_z):
            return 10 ** 2, 10 ** 2
        return best_result, fitness[0]

    def save2csv(self, history):
        f = open(os.path.join(self.save_dir, self.csv_name + ".csv"), "w")
        for dis in history:
            f.write(str(dis) + "\n")

    def show_results(self, history):
        history = np.array(history)
        x = np.array([idx for idx, _ in enumerate(history)])
        plt.plot(x, history)
        plt.show()
        print("Best result: {:.2f}".format(history[-1]))