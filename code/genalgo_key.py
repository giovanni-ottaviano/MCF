#!/usr/bin/env python3

#Genetic Algorithm to decrypt a message - polyalphabetic Vigenere chiper with unknown key length

import os
import time
import random
import argparse
from math import trunc
from numpy import mean
import multiprocessing as mp
from tabulate import tabulate
import genalgo_keyfixed as ga


def new_pop_sizes(scores, popdim):

    """
    Returns a list with new dimension of subpopulation, basing on scores parameter.

    This version preserve the dimension of the whole population.

    """

    #check
    try:
        iterator = iter(scores)
    except:
        raise AssertionError ("scores must be an iterable")

    if not isinstance(popdim, int):
        raise TypeError ("popdim attribute must be an integer. Given {}".format(popdim))
    elif popdim < 0:
        raise ValueError ("popdim attribute must be positive. Given {}".format(popdim))



    tot = sum(scores)

    new_dim = [ int(round(popdim*f/tot))  for f in scores ]

    #check if total population is the same. Otherwise increment best population or decrement worst
    diff = popdim - sum(new_dim)

    if diff > 0:
        best_pos = scores.index(max(scores))
        new_dim[best_pos] += diff

    elif diff < 0:
        worst_pos = scores.index(min(scores))
        new_dim[worst_pos] += diff #is negative


    return new_dim



def check_length(pop, index1, index2):

    """
    Check if all elements in interval have same dimension.

    This function uses [index1, index2) as interval.

    """

    size = len(pop[index1])

    for el in pop[index1:index2]:
        if len(el) != size:
            return False

    return True


def compute_scores(pop, string, indexes, cores=1):

    """
    Returns a list with computed fitnesses for a population of keys with
    different lengths.

    Indexes is a list-like object with the beginning of each subpopulation.
    This version accepts number of cores in order to use multiprocessing.

    """

    nkeys = len(indexes) - 1

    if cores == 1:
        scores = [ [ ind.get_fitness(string) for ind in pop[indexes[i] : indexes[i+1]] ] for i in range(nkeys) ]
    else:
        scores = []
        pool = mp.Pool(processes=cores)

        with pool:
            for i in range(nkeys):
                scores.append(pool.starmap(ga.individual.get_fitness, [ (ind, string) for ind in pop[indexes[i] : indexes[i+1]] ] ))

        pool.join()


    return scores

################################################################################
###                                   MAIN                                   ###
################################################################################

def main():

    """
    Main function - Genetic Algorithm, polyalphabetic Vigenere chiper (unknown key length).
    """

    #get input values and set variables
    parser = argparse.ArgumentParser(description="Genetic algo for cryptography -> polyalph. with unknown key length")
    parser.add_argument('filename', help="File with encrypted text", type=str)
    parser.add_argument('popsize', help="Initial subpopulation size", type=ga.positiveint)
    parser.add_argument('ngen', help="Number of generation", type=ga.positiveint)
    parser.add_argument('--pelite', help="Fraction of elite", type=ga.floatrange(0,1), default=0.1)
    parser.add_argument('--pxover', help="Crossover probability", type=ga.floatrange(0,1), default=0.75)
    parser.add_argument('--mutprob', help="Mutation probability", type=ga.floatrange(0,1), default=0.05)
    parser.add_argument('--randseed', help="Random seed", type=int, default=None)
    parser.add_argument('--klmin', help="Minimum key length (first in)", type=ga.positiveint, default=4)
    parser.add_argument('--klmax', help="Maximum key length (last in)", type=ga.positiveint, default=15)
    parser.add_argument('--cores', help="Number of cores for multiprocessing",
                        type=ga.positiveint, default=len(os.sched_getaffinity(0)))

    args = parser.parse_args()

    filename = args.filename
    npop = args.popsize
    ngen = args.ngen
    p_elite = args.pelite
    p_xover = args.pxover
    mut_prob = args.mutprob
    seed = args.randseed
    klmin = args.klmin      #first key length in
    klmax = args.klmax + 1  #first key length out
    cores = args.cores


    #check values
    if not os.path.isfile(filename):
        raise FileNotFoundError ("File {} does not exists.".format(filename))

    if klmin > klmax:
        print("WARNING: Wrong key size order -> Main will swap them.")
        klmin, klmax = klmax, klmin

    #check number of CPUs
    if cores > len(os.sched_getaffinity(0)):
        print("WARNING: Invalid number of cores. {} will be changed to {}".format(cores, len(os.sched_getaffinity(0))))
        print()
        cores = len(os.sched_getaffinity(0)) #number of cores assigned to this process


    #print chosen parameters
    values = [npop, ngen, p_elite, p_xover, mut_prob] + ["None" if seed == None else seed] + [klmin,
              klmax-1, cores, filename]

    names = ["Population size per key", "Generations", "Fraction of elite",
             "Crossover probability", "Mutation probability", "Random seed",
             "Min key length", "Max key length", "Cores", "Encrypted text in file"]

    appo_tabulate = [[name, val] for name, val in zip(names, values)]

    print("Genetic Algorithm to decrypt a message - polyalphabetic Vigenere chiper with unknown key length.")
    print("...")
    print(tabulate(appo_tabulate, headers=["Parameter", "Value"]))
    print("...")
    print("...")


    #set random seed (for reproducibility)
    random.seed(seed)

    #set method for multiprocessing
    mp.set_start_method('fork')

    #load encoded message and clean it
    text = ga.clean_text(ga.get_text(filename))


    #instanciate lists
    pop, scores = [], []                      #population, fitnesses
    max_list, ave_list = [], []               #max and average fitness for each iteration
    nkeys = klmax - klmin                     #number of keys
    index = [i*npop for i in range(nkeys+1)]  #beginning of a specific population (last one is the end)

    #make population
    for keysize in range(klmin, klmax):
        pop.extend(ga.generate_population(npop, keysize))

    #compute fitnesses
    scores = compute_scores(pop, text, index, cores=cores)

    #evaluate
    max_list.append( [max(scores[i]) for i in range(nkeys)] )           #find max of subpopulation
    ave_list.append( [trunc(mean(scores[i])) for i in range(nkeys)] )   #find ave of subpopulation

    #choose new subpop dimension, proportional to ave fitness
    new_dim = new_pop_sizes(ave_list[-1], npop*nkeys)


    t_i = time.time()

    #run the algorithm
    for step in range(ngen):
        new_pop = []

        #compute new dimensions for subpop
        new_dim = new_pop_sizes(ave_list[-1], npop*nkeys)

        #generate new population
        for i in range(nkeys):
            new_pop.extend(ga.generate_new_population(pop[index[i] : index[i+1]], text, int(p_elite*npop),
                                                      p_xover, dim=new_dim[i], fitness=scores[i]))

        #update list with new indexes (last one is the end)
        index = [0]
        for j in range(nkeys):
            index.append(index[-1] + new_dim[j])

        #mutate population an overwrite pop variable (important use _old)
        pop = ga.mutate_population_old(new_pop, mut_prob)

        #new scores
        scores = compute_scores(pop, text, index, cores=cores)

        #evaluate
        max_list.append( [max(scores[i]) for i in range(nkeys)] )
        ave_list.append( [trunc(mean(scores[i])) for i in range(nkeys)] )


    t_f = time.time()


    #prepare printing of best keys
    max1 = max(max_list[-1])
    pos_max1 = max_list[-1].index(max1)
    pos_in_pop = scores[pos_max1].index(max1)
    key1 = pop[index[pos_max1] + pos_in_pop]

    #WARN!!
    max_list[-1][pos_max1] = -1

    max2 = max(max_list[-1])
    pos_max2 = max_list[-1].index(max2)
    pos_in_pop = scores[pos_max2].index(max2)
    key2 = pop[index[pos_max2] + pos_in_pop]


    print("Most probable keys.")
    print("Length:", klmin + pos_max1, "Key:", "".join(key1))
    print("Length:", klmin + pos_max2, "Key:", "".join(key2))
    print("Time: {:.2f} sec ({:.3f} sec per generation)".format(t_f-t_i, (t_f-t_i)/ngen))

    ga.print_to_file(filename, ga.decode_vigenere_cipher, text, "".join(key1))

    print()
    print("Text decrypted using '{}' as key saved in file: {}".format("".join(key1), ga.print_name))



    return 0



#RUN
if __name__ == '__main__':
    main()
