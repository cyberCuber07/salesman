from ga import GA
import sys
from vis import vis


if __name__ == "__main__":
    ga = GA(file_path=sys.argv[1])
    best_route = ga.run()
    # vis(ga.cities, best_route)