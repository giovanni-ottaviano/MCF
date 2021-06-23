#!/usr/bin/env python3

#Genetic Algorithm to decrypt a message - monoalphabetic permutation chiper

import os
import time
import random
import argparse
import numpy as np
import multiprocessing as mp
from tabulate import tabulate
from collections import Counter, Iterable
import genalgo_keyfixed as ga


class individual(list):

    """
    Class individual for GA monoalphabetic permutation chiper.

    This class inherits from built-in list whose elements are integer fro 0 to 25
    and represent a permutation of the alphabet.

    """

    def get_fitness(self, string, dict_bigrams):

        """
        Compute and return the fitness function on a given (encrypted) string.

        Here the scoring is based on the fraction of all bigraphs
        in the decrypted text, whose decoding is done with self as key (permutation).

        """

        cost, coeff = 0., 0.1

        #decode text using self as key
        decoded = decode_permutation_cipher(string, self)

        #count bigrams
        count_bigram = Counter(decoded[i:i+2] for i in range(len(decoded)-1))
        sum_bi = sum(count_bigram.values())  #number of bigrams (could be fixed - always the same)

        #count bigram frequencies in decoded text and increase cost with absolute difference
        for bi, freq in dict_bigrams.items():
            freq_decoded = decoded.count(bi)/sum_bi

            cost += abs(freq_decoded - freq)


        fitness = 100*(1. - coeff*cost)


        return fitness


    def mate(self, other, p_xover=1.):

        """
        Returns a list with two individuals after crossover operation, with given probability.
        Otherwise, returns the same elements.

        This method acts on self and another given individual and uses Cycle Crossover.

        """

        if random.random() < p_xover:
            return CX(self, other)

        return [self, other]


    def mutate(self, mut_prob):

        """
        Mutate individual with given probability.

        Implemented mutations:
            1. Letter swap
            2. Inversion

        """

        #letter swap: get 2 different random positions in the list and swap them
        #inversion: invert each element in a given (random) range

        if random.random() < mut_prob:
            pos1 = random.randint(0, len(self)-1)
            pos2 = random.randint(0, len(self)-1)

            while pos1 == pos2:
                pos2 = random.randint(0, len(self)-1)

            #swap
            self[pos1], self[pos2] = self[pos2], self[pos1]

        elif random.random() < mut_prob:
            pos1 = random.randint(0, len(self)-1)
            pos2 = random.randint(0, len(self)-1)

            while pos1 == pos2:
                pos2 = random.randint(0, len(self)-1)

            pos_min = min(pos1, pos2)
            pos2 = max(pos1, pos2)

            #inversion
            self[pos_min:pos2+1] = self[pos_min:pos2+1][::-1]



#another way: use itertools.permutation and choose the integer as permutation number
def code_permutation_cipher(string, key):

    """
    Returns a string with encrypted text using the following encoding method:
    monoalphabetic permutation cipher (with given key/permutation).

    A letter is encoded through substitution with another letter (new alphabet is
    just a permutation of the old one).

    This version can get 2 types of key:
     1. int, seed for random generator that sets the permutation
     2. iterable, with 26 integer that is a permutation of 0-25

    """


    letters = [chr(l) for l in range(97, 123)]
    perm_letters = []

    if isinstance(key, int):
        old_state = random.getstate() #save current state of random generator
        random.seed(key)              #set new state for permutation

        copy_letters = letters[:]
        #creo una permutazione
        for i in range(len(letters)):
            pos = random.randint(0, len(copy_letters) - 1)

            perm_letters.append(copy_letters.pop(pos))

        #restore old state
        random.setstate(old_state)

    else:                           #check if iterable (duck typing)
        try:
            iterator = iter(key)
        except:
            raise AssertionError ("Key must be an integer or an iterable with 26 elements, but {} given".format(type(key)))
        else:
            if len(key) == 26:
                perm_letters = [chr(k + 97) for k in key]  #key is an array with 26 elements
            else:
                raise AssertionError ("Key length and alphabet length must be the same. Given {} expected 26".format(len(key)))



    permuted = dict(zip(letters, perm_letters))

    #encrypt given string
    coded = [permuted[l] for l in string]


    return "".join(coded)


def decode_permutation_cipher(string, key):

    """
    Returns a string with decrypted text using the following decoding method:
    monoalphabetic permutation cipher (with given key/permutation).

    This version can get 2 types of key:
     1. int, seed for random generator that sets the permutation
     2. iterable, with 26 integer that is a permutation of 0-25

    """


    letters = [chr(l) for l in range(97, 123)]
    perm_letters = []

    if isinstance(key, int):
        old_state = random.getstate() #save current state of random generator
        random.seed(key)              #set new state for permutation

        copy_letters = letters[:]
        #creo una permutazione
        for i in range(len(letters)):
            pos = random.randint(0, len(copy_letters) - 1)

            perm_letters.append(copy_letters.pop(pos))

        #restore old state
        random.setstate(old_state)

    else:                           #check if iterable with duck typing
        try:
            iterator = iter(key)
        except:
            raise AssertionError ("Key must be an integer or an iterable with 26 elements, but {} given".format(type(key)))
        else:
            if len(key) == 26:
                perm_letters = [chr(k + 97) for k in key]  #key is an array with 26 elements
            else:
                raise AssertionError ("Key length and alphabet length must be the same. Given {} expected 26".format(len(key)))


    permuted = dict(zip(perm_letters, letters))

    #decode
    decoded = [permuted[s] for s in string]


    return "".join(decoded)



def OBX(parent1, parent2, rndpos=None):

    """
    Order Based Crossover.

    Selects at random several positions in one of the parent and the order of
    the elements in the selected positions of this parent is imposed on the other
    parent to produce one child. The other child is generated in an analogous
    manner from the other parent.

    This version accept an iterable with 2 iterable elements (position whose
    order must be imposed in the other parent). This is useful while testing the
    function.

    """

    #check
    if len(parent1) != len(parent2):
        raise AssertionError ("Parents must have same length. Instead given {} and {}".format(len(parent1), len(parent1)))

    if rndpos != None:
        if isinstance(rndpos, Iterable):
            if len(rndpos) != 2:
                raise IndexError ("Random position must be an iterable with 2 elements. Instead len is {}".format(len(rndpos)))
        else:
            raise AssertionError ("Random position must be an iterable with 2 elements. Instead given {}".format(type(rndpos)))

    #choose number of elements
    if (rndpos == None):
        rndint = random.randint(1, len(parent1) - 1)

    offspring = []
    parents = (parent1, parent2)
    k = 0

    for i in range(2):
        if rndpos == None:
            appo_elem = random.sample(parents[i], rndint)

            #order as in parent
            ord_elem = []

            for elem in parents[i]:
                if elem in appo_elem:
                    ord_elem.append(elem)

        else:
            ord_elem = rndpos[k]

        #do crossover
        son = parents[1-i][:]
        j = 0

        for i, elem in enumerate(son):
            if elem in ord_elem:
                son[i] = ord_elem[j]
                j += 1

        offspring.append(individual(son))
        k += 1


    return offspring


#modified order based crossover
def MOC(parent1, parent2, rndint=None):

    """
    Modified Order Crossover.

    A randomly chosen crossover point split parents in left and right
    substrings. The right substrings of the parent s1 and s2 are selected.
    After selection of elements the process is the same as the order crossover.
    Only difference is that instead of selecting random several positions in a
    parent all the positions to the right of the randomly chosen crossover point
    are selected.

    This version accepts the point in which the parents must be split.

    """

    #check parameters
    if len(parent1) != len(parent2):
        raise AssertionError ("Parents must have same length. Instead given {} and {}".format(len(parent1), len(parent1)))

    if rndint != None:
        if isinstance(rndint, int):
            if rndint <= 0 or rndint >= len(parent1):
                raise ValueError ("rndint must be a positive integer. Instead given {}".format(rndint))
        else:
            raise TypeError ("rndint must be a positive integer. Instead given {}".format(type(rndint)))


    #choose number of elements
    if (rndint == None):
        rndint = random.randint(1, len(parent1) - 1)


    parents = [parent1, parent2] #by reference (no copy)
    offspring = []

    for i in range(2):
        cut_left = parents[i][:rndint]
        cut_right = parents[i][rndint:]

        son = []
        j = 0

        for elem in parents[1-i]:
            if elem in cut_left:
                son.append(elem)
            else:
                son.append(cut_right[j])
                j += 1

        offspring.append(individual(son))


    return offspring


#check if offspring is ready
def is_not_ready(offspring):

    return -1 in offspring


def CX(parent1, parent2):

    """
    Cycle Crossover.

    This operator identifies a number of so-called cycles between two parent chromosomes.
    Then, to form Child 1, each cycle is copied from a random chosen parent,
    until child has the same dimensions of parents.
    The same process is repeated for the other child.

    """

    #check parameters
    if len(parent1) != len(parent2):
        raise AssertionError ("Parents must have same length. Instead given {} and {}".format(len(parent1), len(parent1)))

    parents = (parent1, parent2)
    offspring = []

    for _ in range(2):
        son = [-1 for _ in range(len(parent1))]

        while is_not_ready(son):
            prt_pos = random.randint(0,1)

            #first free position
            for i in range(len(son)):
                if son[i] == -1:
                    pos = i
                    break

            #repeat 'til find a cycle (do-while)
            while True:
                new_elem = parents[prt_pos][pos]
                son[pos] = new_elem
                pos = parents[1-prt_pos].index(new_elem)

                if son[pos] != -1:
                    break


        offspring.append(individual(son))


    return offspring



def generate_population(npop):

    """
    Returns a population of npop individuals with given length.
    Here individuals are lists with integers from 0 to 25.

    """

    l = list(range(26))  #a = 0, z = 25

    pop = []

    for n in range(npop):
        random.shuffle(l)
        pop.append(individual(l[:]))  #need to append a copy of l


    return pop



def generate_new_population(pop, string, n_elite, dict_bigrams, p_xover, fitness=None, cores=1):

    """
    Generate a new population from list of individuals pop, using individual's mate method
    and applying elitism (part of old population is preserved and passes to new population).
    Returned population has the same size as the old one.

    This version can accept an iterable with fitnesses, otherwise requires a
    dictionary with bigrams and their fraction to calculate fitness and number of cores
    to run multiprocessing.

    """

    if n_elite >= len(pop):
        raise AssertionError (("Number of elite members must be less than individuals "
                               "in population. Given {} as elite with {} in population.".format(n_elite, len(pop))))

    if fitness == None:
        fitness = compute_scores(pop, string, dict_bigrams, cores=cores)
    else:
        if len(fitness) != len(pop):
            raise AssertionError ("Sequence of fitnesses and population must have equal size.")


    fitness_appo = fitness[:]
    new_pop = []

    nxover = (len(pop) - n_elite)//2  #number of crossovers

    #ELITISM: preserve best member of this population and save the unchanged
    for _ in range(n_elite):
        pos_max = fitness_appo.index(max(fitness_appo))
        new_pop.append(pop[pos_max])
        fitness_appo[pos_max] = -1  #work only for positive fitnesses

    #crossover
    for _ in range(nxover):
        parent1, parent2 = ga.weighted_sample(pop, fitness, 2)
        new_pop.extend(parent1.mate(parent2, p_xover))

    #if is odd, add another element
    if (len(pop) - n_elite) % 2 == 1:
        parent1, parent2 = ga.weighted_sample(pop, fitness, 2)
        new_pop.append(parent1.mate(parent2, p_xover)[0])


    return new_pop


def compute_scores(pop, string, dict_bigrams, cores=1):

    """
    Returns a list with computed fitnesses for a population of keys.

    This version accepts number of cores in order to use multiprocessing.

    """

    if cores == 1:
        scores = [ind.get_fitness(string, dict_bigrams) for ind in pop]

    else:
        pool = mp.Pool(processes=cores)
        with pool:
            scores = pool.starmap(individual.get_fitness, [(ind, string, dict_bigrams) for ind in pop])

        pool.join()


    return scores


def load_bigrams(file):

    """
    Return a dictionary with bigrams as keys and fraction of appearance as values.

    File must have a specif structure: 26 columns with bigrams on even and values on
    odd (percentage).

    """

    char = np.genfromtxt(file, dtype=str, usecols=(range(0,26,2)))
    char = char.flatten() #676 elements
    char = [bi.lower() for bi in char]

    perc = np.loadtxt(file, dtype=float, usecols=(range(1,26,2)))/100.
    perc = perc.flatten() #676 elements

    bi_dict = dict(zip(char, perc))


    return bi_dict


################################################################################
###                                   MAIN                                   ###
################################################################################

def main():

    """
    Main function - Genetic Algorithm, monoalphabetic permutation chiper.
    """

    #get input values and set variables
    parser = argparse.ArgumentParser(description="Genetic algo for cryptography -> monoalph. permutation cipher")
    parser.add_argument('filename', help="File with encrypted text", type=str)
    parser.add_argument('popsize', help="Population size", type=ga.positiveint)
    parser.add_argument('ngen', help="Number of generations", type=ga.positiveint)
    parser.add_argument('--pelite', help="Fraction of elite", type=ga.floatrange(0,1), default=0.1)
    parser.add_argument('--pxover', help="Crossover probability", type=ga.floatrange(0,1), default=0.75)
    parser.add_argument('--mutprob', help="Mutation probability", type=ga.floatrange(0,1), default=0.08)
    parser.add_argument('--randseed', help="Random seed", type=int, default=None)
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
    cores = args.cores

    #check values
    if not os.path.isfile(filename):
        raise FileNotFoundError ("File {} does not exists.".format(filename))

    #check number of CPUs
    if cores > len(os.sched_getaffinity(0)):
        print("WARNING: Invalid number of cores. {} will be changed to {}".format(cores, len(os.sched_getaffinity(0))))
        print()
        cores = len(os.sched_getaffinity(0)) #number of cores assigned to this process

    #import bigrams and create a dictionary
    dict_bigrams = load_bigrams("bigrams.dat")


    #print chosen parameters
    values = [npop, ngen, p_elite, p_xover, mut_prob
             ] + ["None" if seed == None else seed] + [cores, filename]

    names = ["Population size", "Generations", "Fraction of elite", "Crossover probability",
             "Mutation probability", "Random seed", "Cores", "Encrypted text in file"]

    appo_tabulate = [[name, val] for name, val in zip(names, values)]

    print("Genetic Algorithm to decrypt a message - monoalphabetic permutation chiper.")
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


    #make new population
    pop = generate_population(npop)

    max_list = []  #max fitness for each iteration
    max_key = []   #key with maximum fitness

    #compute fitness using multiprocessing
    scores = compute_scores(pop, text, dict_bigrams, cores=cores)

    max_score = max(scores)
    max_list.append(max_score)
    pos_max = scores.index(max_score)
    max_key.append(pop[pos_max])


    t_i = time.time()
    
    #run the algorithm
    for step in range(ngen):
        pop = generate_new_population(pop, text, int(p_elite*npop), dict_bigrams,
                                      p_xover, scores)

        ga.mutate_population(pop, mut_prob)

        #compute fitness using multiprocessing
        scores = compute_scores(pop, text, dict_bigrams, cores=cores)

        #update list with best value
        max_score = max(scores)
        max_list.append(max_score)
        pos_max = scores.index(max_score)
        max_key.append(pop[pos_max])



    t_f = time.time()

    #decrypted alphabet
    dec_alphabet = [chr(k + 97) for k in max_key[-1]]

    print()
    print("Time: {:.2f} sec ({:.3f} sec per generation)".format(t_f-t_i, (t_f-t_i)/ngen))
    print("Decrypted alphabet:", dec_alphabet)
    print()

    ga.print_to_file(filename, decode_permutation_cipher, text, max_key[-1])

    print()
    print("Decrypted text saved in file:", ga.print_name)



    return 0



#RUN
if __name__ == '__main__':
    main()
