#!/usr/bin/env python3

#Genetic Algorithm to decrypt a message - unknown cryptography (Vigenere/permutation)

import os
import random
import argparse
import time
import numpy as np
from math import trunc
import multiprocessing as mp
from tabulate import tabulate
import genalgo_keyfixed as ga
import genalgo_permutation as gperm
import genalgo_key as gpop


def compute_scores(pop_k, pop_p, string, dict_bigrams, indexes, cores=1):

    """
    Returns two lists with computed fitnessesfor a population of keys ith different
    length and a population of permutations.

    This version accepts number of cores in order to use multiprocessing.
    """

    #metti qualche check

    nkeys = len(indexes) - 1

    #compute new fitness (if monocore, list-comprehension is much better)
    if cores == 1:
        scores_k = [ [ ind.get_fitness(string) for ind in pop_k[indexes[i] : indexes[i+1]] ] for i in range(nkeys) ]
        scores_p = [ ind.get_fitness(string, dict_bigrams) for ind in pop_p ]
    else:
        scores_k = []

        pool = mp.Pool(processes=cores)
        with pool:
            for i in range(nkeys):
                scores_k.append(pool.starmap(ga.individual.get_fitness, [ (ind, string) for ind in pop_k[indexes[i] : indexes[i+1]] ] ))

            scores_p = pool.starmap(gperm.individual.get_fitness, [(ind, string, dict_bigrams) for ind in pop_p])

        pool.join()


    return scores_k, scores_p


def is_vigenere(fitness_key, fitness_perm, lim_key=1500, lim_perm=89.5):

    """"
    Determine if encryption method was Vigenere cipher of permutation cipher.
    Based on maximum values reached by fitnesses.

    """

    if max(fitness_key) > lim_key and fitness_perm < lim_perm:
        return True
    elif max(fitness_key) < lim_key and fitness_perm > lim_perm:
        return False
    else:
        print("WARNING: cannot decide if Vigenere or permutation!")    #dai test non si arriva mai in questo caso con i parametri impostati

    return False



################################################################################
###                                   MAIN                                   ###
################################################################################

def main():

    """
    Main function - Genetic Algorithm, unknown cryptography (Vigenere/permutation).
    """

    #get input values and set variables
    parser = argparse.ArgumentParser(description="Genetic algo for cryptography -> unknown cryptography (Vigenere/permutation)")
    parser.add_argument('filename', help="File with encrypted text", type=str)
    parser.add_argument('popsize', help="Initial single population size", type=ga.positiveint)
    parser.add_argument('ngen', help="Number of generations", type=ga.positiveint)
    parser.add_argument('--popsize_perm', help="Population size", type=ga.positiveint, default=None)
    parser.add_argument('--pelite', help="Fraction of elite", type=ga.floatrange(0,1), default=0.1)
    parser.add_argument('--pxover', help="Crossover probability", type=ga.floatrange(0,1), default=0.75)
    parser.add_argument('--mutprob', help="Mutation probability", type=ga.floatrange(0,1), default=0.05)
    parser.add_argument('--mutprob_perm', help="Mutation probability (permutations)", type=ga.floatrange(0,1), default=None)
    parser.add_argument('--randseed', help="Random seed", type=int, default=None)
    parser.add_argument('--klmin', help="Minimum key length (first in)", type=ga.positiveint, default=4)
    parser.add_argument('--klmax', help="Maximum key length (last in)", type=ga.positiveint, default=15)
    parser.add_argument('--cores', help="Number of cores for multiprocessing",
                        type=ga.positiveint, default=len(os.sched_getaffinity(0)))

    args = parser.parse_args()

    filename = args.filename
    npop = args.popsize
    ngen = args.ngen
    npop_perm = args.popsize_perm
    p_elite = args.pelite
    p_xover = args.pxover
    mut_prob = args.mutprob
    mut_prob_perm = args.mutprob_perm
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

    #if None, set the same values as in subpopulation
    if npop_perm == None:
        npop_perm = npop
    if mut_prob_perm == None:
        mut_prob_perm = mut_prob

    #import bigrams and create a dictionary
    dict_bigrams = gperm.load_bigrams("bigrams.dat")

    #print chosen parameters
    values = [npop, npop_perm, ngen, p_elite, p_xover, mut_prob, mut_prob_perm] + ["None" if seed == None else seed] + [
              klmin, klmax-1, cores, filename]

    names = ["Population size per key", "Population permutations", "Generations", "Fraction of elite",
             "Crossover probability", "Mutation probability", "Mutation probability (perm)",
             "Random seed", "Min key length", "Max key length", "Cores", "Encrypted text in file"]

    appo_tabulate = [[name, val] for name, val in zip(names, values)]

    print("Genetic Algorithm to decrypt a message - unknown cryptography between Vigenere cipher and permutation cipher.")
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

#ELIMINA
    #creo la permutazione da usare come chiave (creo un iteratore e rrivo fino a una permutazione scelta)
    perm = list(range(26))
    perm.reverse()
    random.shuffle(perm)
    perm = tuple(perm)
    #encode text
    text = gperm.code_permutation_cipher(text, perm)
# ELIMINA
    #text = ga.code_vigenere_cipher(text, "straordinario")


    #instanciate lists
    pop_key, pop_perm = [], []
    scores_key, scores_perm = [], []
    max_list, ave_list, max_list_perm = [], [], []
    nkeys = klmax - klmin   #number of keys
    index = [i*npop for i in range(nkeys+1)]  #beginning of a specific population (last one is the end)

    #make both populations
    for keysize in range(klmin, klmax):
        pop_key.extend(ga.generate_population(npop, keysize))

    pop_perm = gperm.generate_population(npop_perm)

    #compute fitness
    scores_key, scores_perm = compute_scores(pop_key, pop_perm, text, dict_bigrams,
                                             index, cores=cores)


    #evaluate
    max_list.append( [max(scores_key[i]) for i in range(nkeys)] )              #find max of subpopulation
    ave_list.append( [trunc(np.mean(scores_key[i])) for i in range(nkeys)] )   #find ave of subpopulation
    max_list_perm.append(max(scores_perm))                                     #find max of population of permutations

    #choose new subpop dimension, proportional to max/ave fitness
    new_dim = gpop.new_pop_sizes(ave_list[-1], npop*nkeys)


    t_i = time.time()

    #run the algorithm
    for step in range(ngen):
        t0 = time.time()                                                                                #TOGLI
        new_pop_key = []

        new_dim = gpop.new_pop_sizes(ave_list[-1], npop*nkeys)

        print("Dim:", new_dim)                                                                                #TOGLI
        print("Max permutation:", max_list_perm[-1])                                                                                #TOGLI

        #generate new populations
        for i in range(nkeys):
            new_pop_key.extend(ga.generate_new_population_elite(pop_key[index[i] : index[i+1]], text, int(p_elite*npop),
                                                                p_xover, dim=new_dim[i], fitness=scores_key[i]))

        pop_perm = gperm.generate_new_population(pop_perm, text, int(p_elite*npop_perm), dict_bigrams,
                                                 p_xover, fitness=scores_perm)


        #update list with new indexes (last one is the end)
        index = [0]
        for j in range(nkeys):
            index.append(index[-1] + new_dim[j])

        #mutate population and overwrite pop variable
        pop_key = ga.mutate_population_old(new_pop_key, mut_prob)
        ga.mutate_population(pop_perm, mut_prob_perm)

        #compute fitness
        scores_key, scores_perm = compute_scores(pop_key, pop_perm, text, dict_bigrams,
                                                 index, cores=cores)

        #evaluate
        max_list.append( [max(scores_key[i]) for i in range(nkeys)] )              #find max of subpopulation
        ave_list.append( [trunc(np.mean(scores_key[i])) for i in range(nkeys)] )   #find ave of subpopulation
        max_list_perm.append(max(scores_perm))

        t1 = time.time()
#ELIMINA
        print("Step:", step, sum(new_dim), len(pop_perm))                                                                               #TOGLI
        print("Time: {:.2f}".format(t1-t0))                                                                               #TOGLI
        print()                                                                               #TOGLI

    t_f = time.time()                                                                               #TOGLI


    #find if polyalphabetic or monoalphabetic and print to file
    if is_vigenere(max_list[-1], max_list_perm[-1]):

        #prepare printing of best keys
        max1 = max(max_list[-1])
        pos_max1 = max_list[-1].index(max1)
        pos_in_pop = scores_key[pos_max1].index(max1)
        key1 = pop_key[index[pos_max1] + pos_in_pop]

        #ATTENZIONE
        max_list[-1][pos_max1] = -1

        max2 = max(max_list[-1])
        pos_max2 = max_list[-1].index(max2)
        pos_in_pop = scores_key[pos_max2].index(max2)
        key2 = pop_key[index[pos_max2] + pos_in_pop]

        print("Text in file", filename, "was encrypted using Vigenere cipher.")
        print()
        print("Most probable keys.")
        print("Length:", klmin + pos_max1, "Key:", "".join(key1))
        print("Length:", klmin + pos_max2, "Key:", "".join(key2))

        ga.print_to_file(filename, ga.decode_vigenere_cipher, text, "".join(key1))

    else:
        pos_max = scores_perm.index(max_list_perm[-1])
        key = pop_perm[pos_max]
        dec_alphabet = [chr(k + 97) for k in key]

        print("Text in file", filename, "was encrypted using permutation cipher.")
        print()
        print("Decrypted alphabet:", dec_alphabet)

        ga.print_to_file(filename, gperm.decode_permutation_cipher, text, key)


    print("Time: {:.2f} sec ({:.3f} sec per generation)".format(t_f-t_i, (t_f-t_i)/ngen))
    print()
    print("Text decrypted saved in file:", ga.print_name)


    return 0




#RUN
if __name__ == '__main__':
    main()
