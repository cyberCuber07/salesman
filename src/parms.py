
X_min, Y_min = 0, 0
X_max, Y_max = 400, 400

N_SETS = 1 * 10 ** 2
N_PARENTS = int(N_SETS * 0.2)
N_CROSSOVER = N_SETS - N_PARENTS

N = 20
CROSSOVER_POINT = N // 2
N_MUTATIONS = 1

ITER = 1 * 10 ** 2
# MUTE_PER == 5 * 10 ** -1 resulted in fast convergence
MUTE_PER = 5 * 10 ** -1
