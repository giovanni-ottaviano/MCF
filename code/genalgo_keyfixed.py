#!/usr/bin/env python3

#Genetic Algorithm to decrypt a message - polyalphabetic Vigenere chiper with given key length

import os
import re
import time
import random
import argparse
import multiprocessing as mp
from tabulate import tabulate


class individual(list):

    """
    Class individual for GA polyalphabetic Vigenere chiper.

    This class inherits from built-in list and represent the key to encode/decode
    a given text with given cipher.
    """


    def get_fitness(self, string):

        """
        Compute and return the fitness function on a given (encrypted) string.

        Here the scoring is based on the number of most common bigrams and trigrams
        in the decrypted text, whose decoding is done with self as key.

        """

        #most common digraphs and trigraphs
        MC_digraph = ("th", "er", "on", "an", "re", "he", "in", "ed", "nd", "ha",
                      "at", "en", "es", "of", "or", "nt", "ea", "ti", "to", "it",
                      "st", "io", "le", "is", "ou", "ar", "as", "de", "rt", "ve")

        MC_trigraph = ("the", "and", "tha", "ent", "ion", "tio", "for", "nde",
                       "has", "nce", "edt", "tis", "oft", "sth", "men")


        fitness = 0

        #decrypt string with self as key
        decoded =  decode_vigenere_cipher(string, "".join(self))

        #different scores for most common trigraph and bigraph (quadratic)
        for tri in MC_trigraph:
            fitness += 9*decoded.count(tri)

        for bi in MC_digraph:
            fitness += 4*decoded.count(bi)


        return fitness


    def mate(self, other, p_xover=1.):

        """
        Returns a list with two individuals after crossover operation, with given probability.
        Otherwise, returns the same elements.

        This method acts on self and another given individual.

        """

        if random.random() < p_xover:
            xpos = random.randint(1, len(self) - 1)
        else:
            xpos = 0

        return [ individual(self[:xpos] + other[xpos:]),
                 individual(other[:xpos] + self[xpos:]) ]


    def mutate(self, mut_prob):

        """
        Mutate individual with given probability.

        Implemented mutations:
            1. Letter swap

        """

        if random.random() < mut_prob:
            pos = random.randint(0, len(self) - 1)
            n_char = random.randint(97, 122)      #ASCII coding a = 97, z = 122

            #change letter
            while ord(self[pos]) == n_char:
                n_char = random.randint(97, 122)

            self[pos] = chr(n_char)



#N.B: this method gives +1 shift to letters, compared to other versions of Vigenere cipher
def code_vigenere_cipher(text, key):

    """
    Returns a string with encrypted text using the following encoding method:
    Vigenere polyalphabetic cipher (with given key).

    Letters are encoded adding their position in the alphabet to the position
    of the corresponding letter in the (repeated) key.

    """

    if not isinstance(key, str):
        raise AssertionError ("Key value must be a string, instead {} was given.".format(type(key)))


    nletters = 26

    #prepare encoding (replicate key 'til have the same dimension of text)
    times = len(text)//len(key)
    res = len(text)%len(key)

    code_word = times*key
    for i in range(res):
        code_word += key[i]


    #make a shift from initial position (periodic boundary conditions) ---> ASCII coding: a = 97, z = 122
    encoded = [ chr( (ord(s) - 96 + ord(r) - 97) % nletters + 97) for s, r in zip(text, code_word) ]


    return "".join(encoded)



def decode_vigenere_cipher(text, key):

    """
    Returns a string with decrypted text using the following dencoding method:
    Vigenere polyalphabetic cipher (with given key).

    """

    if not isinstance(key, str):
        raise AssertionError ("Key value must be a string, instead {} was given.".format(type(key)))


    nletters = 26

    #prepare encoding (replicate key 'til have the same dimension of text)
    times = len(text)//len(key)
    res = len(text)%len(key)

    code_word = times*key
    for i in range(res):
        code_word += key[i]

    #make a shift from initial position (periodic boundary conditions) ---> ASCII coding: a = 97, z = 122
    #in python -1 % n = n-1
    decoded = [ chr( (ord(s) - ord(r) - 1) % nletters + 97) for s, r in zip(text, code_word) ]


    return "".join(decoded)



def generate_population(npop, length):

    """
    Returns a population of npop individuals with given length.

    Here individuals are lists with lower case letters.

    """

    population = []

    for i in range(npop):
        population.append( individual([chr(random.randint(97, 122)) for _ in range(length)]) )


    return population


#same behavior as: numpy.random.choice(a, size=None, replace=False, p=None)
def weighted_sample(v, scores, n):

    """
    Generates a random sample without replacement from a given iterable.

    """

    if n < 0 or n > len(v) or not isinstance(n, int):
        raise ValueError ("{} not valid. It must be integer in (0, {})".format(n, len(v)))

    #copy initial population and scores
    chosen = []
    v_c = v[:]
    scores_c = scores[:]

    for i in range(n):
        totscore = sum(scores_c)
        r = random.random()
        tot = 0

        for j, s in enumerate(scores_c):
            tot += s/float(totscore)

            if tot > r:
                break

        chosen.append(v_c.pop(j))
        scores_c.pop(j)

    return chosen



def generate_new_population(pop, string, n_elite, p_xover, dim=None, fitness=None, cores=1):

    """
    Generate a new population from list of individuals pop, using individual's mate method
    and applying elitism (part of old population is preserved and passes to new population).
    Returned population has the same size as the old one.

    This version can accept an iterable with fitnesses, otherwise it requires number of
    cores to calculate it.
    This version allow to select a different dimension for new population.

    """

    #check parameters
    if dim != None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError ("Dimension of new population must be a positive integer. Instead given {}".format(dim))

    else:
        dim = len(pop) #if no dim, use old population dim

    if n_elite >= dim:
        raise AssertionError (("Number of elite members must be less than individuals "
                               "in population. Given {} as elite with {} in population.".format(n_elite, dim)))

    #if necessary compute fitness. Otherwise check if good
    if fitness == None:
        fitness = compute_scores(pop, string, cores=cores)
    else:
        if len(fitness) != len(pop):
            raise AssertionError ("Sequence of fitnesses and population must have equal size.")



    fitness_appo = fitness[:]
    new_pop = []

    nxover = (dim - n_elite)//2  #number of crossovers

    #ELITISM: preserve best member of this population and save the unchanged
    for _ in range(n_elite):
        pos_max = fitness_appo.index(max(fitness_appo))
        new_pop.append(pop[pos_max])
        fitness_appo[pos_max] = -1  #work only for positive fitnesses

    #crossover
    for _ in range(nxover):
        parent1, parent2 = weighted_sample(pop, fitness, 2)
        new_pop.extend(parent1.mate(parent2, p_xover))


    #if is odd, add another element
    if (dim - n_elite) % 2 == 1:
        parent1, parent2 = weighted_sample(pop, fitness, 2)
        new_pop.append(parent1.mate(parent2, p_xover)[0])


    return new_pop


def mutate_population_old(pop, mut_prob):

    """
    Returns a new population, after the mutation process.

    """

    new_pop = []

    for i in pop:
        i.mutate(mut_prob)
        new_pop.append(i)

    return new_pop


def mutate_population(pop, mut_prob):

    """
    Apply mutations to each individual.

    """

    for ind in pop:
        ind.mutate(mut_prob)



def compute_scores(pop, string, cores=1):

    """
    Returns a list with computed fitnesses for a population of keys.

    This version accepts number of cores in order to use multiprocessing.
    """

    if cores == 1:
        scores = [ind.get_fitness(string) for ind in pop]

    else:
        pool = mp.Pool(processes=cores)
        with pool:
            scores = pool.starmap(individual.get_fitness, [(ind, string) for ind in pop])

        pool.join()


    return scores



def get_text(filename):

    """
    Read input file into a string.

    This version trims newlines '\n'
    """

    with open(filename, 'r') as file:
        input = file.read().replace('\n', '')

    return input


def clean_text(text):

    """
    Clean input text, removing all non-alphabet chars and returning a lower case
    string.

    This version leaves input text unchanged.
    """

    try:
        new_text = text.lower()
    except:
        raise TypeError ("clean_text argument must be a string. Instead given {}".format(type(text)))


    p = re.compile(r'[a-z]')

    new_text = p.findall(new_text)
    #new_text = [elem for elem in new_text if 97 <= ord(elem) <= 122] #same result


    return "".join(new_text)


def print_to_file(inputfile, f_decode, text, key):

    """
    Print the decrypted text into a file, whose name is composed unsing input name.
    The argument 'f_decode' must be the function needed to decode the given string.
    """

    #compose name for new file
    global print_name   #can use it in main
    print_name = os.path.basename(inputfile)
    print_name = "decrypted_" + print_name.split(".")[0] + ".txt"

    #print decrypted text in file
    with open(print_name, 'w') as file:
        file.write(f_decode(text, key))


#new type for argparse
def positiveint(value):

    """
    Type function for argparse - Positive int
    """

    #check if numeric
    try:
        intvalue = int(value)
    except:
        raise argparse.ArgumentTypeError ("{} is not a valid positive int".format(value))

    #check if positive
    if intvalue <= 0:
        raise argparse.ArgumentTypeError ("{} is not a valid positive int".format(value))


    return intvalue


#new type for argparse
def floatrange(min, max):

    """
    Type function for argparse - Float with bounds

     min - minimum acceptable parameter
     max - maximum acceptable parameter
    """

    # Define the function with default arguments
    def float_range_checker(value):

        try:
            f = float(value)
        except:
            raise argparse.ArgumentTypeError("{} is not a valid float".format(value))

        if f < min or f > max:
            raise argparse.ArgumentTypeError("Argument must be in range [{},{}]".format(min,max))


        return f

    # Return function handle to checking function
    return float_range_checker

################################################################################
###                                   MAIN                                   ###
################################################################################

def main():

    """
    Main function - Genetic Algorithm, polyalphabetic Vigenere chiper (fixed key length).
    """

    #get input values and set variables
    parser = argparse.ArgumentParser(description="Genetic algo for cryptography -> polyalph. with given key length")
    parser.add_argument('filename', help="File with encrypted text", type=str)
    parser.add_argument('popsize', help="Population size", type=positiveint)
    parser.add_argument('keysize', help="Number of letters in key", type=positiveint)
    parser.add_argument('ngen', help="Number of generations", type=positiveint)
    parser.add_argument('--pelite', help="Fraction of elite", type=floatrange(0,1), default=0.1)
    parser.add_argument('--pxover', help="Crossover probability", type=floatrange(0,1), default=0.75)
    parser.add_argument('--mutprob', help="Mutation probability", type=floatrange(0,1), default=0.05)
    parser.add_argument('--randseed', help="Random seed", type=int, default=None)
    parser.add_argument('--cores', help="Number of cores for multiprocessing",
                        type=positiveint, default=len(os.sched_getaffinity(0)))

    args = parser.parse_args()

    filename = args.filename
    npop = args.popsize
    keysize = args.keysize
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


    #print chosen parameters
    values = [npop, keysize, ngen, p_elite, p_xover, mut_prob
             ] + ["None" if seed == None else seed] + [cores, filename]

    names = ["Population size", "Key size", "Generations", "Fraction of elite", "Crossover probability",
             "Mutation probability", "Random seed", "Cores", "Encrypted text in file"]

    appo_tabulate = [[name, val] for name, val in zip(names, values)]

    print("Genetic Algorithm to decrypt a message - polyalphabetic Vigenere chiper with given key length.")
    print("...")
    print(tabulate(appo_tabulate, headers=["Parameter", "Value"]))
    print("...")
    print("...")

    #set random seed (for reproducibility)
    random.seed(seed)

    #set method for multiprocessing
    mp.set_start_method('fork')

    #load encrypted message and clean it
    text = clean_text(get_text(filename))


    #make new population
    pop = generate_population(npop, keysize)

    max_list = []  #max fitness for each iteration
    max_key = []   #key with maximum fitness

    #fitnesses
    scores = compute_scores(pop, text, cores=cores)

    max_score = max(scores)
    max_list.append(max_score)
    pos_max = scores.index(max_score)
    max_key.append("".join(pop[pos_max]))


    t_i = time.time()

    #run the algorithm
    for step in range(ngen):
        pop = generate_new_population(pop, text, int(p_elite*len(pop)), p_xover, fitness=scores)
        mutate_population(pop, mut_prob)

        #compute fitnesses
        scores = compute_scores(pop, text, cores=cores)

        #evaluate
        max_score = max(scores)
        max_list.append(max_score)
        pos_max = scores.index(max_score)
        max_key.append("".join(pop[pos_max]))


    t_f = time.time()

    #final printing and save to file
    print()
    print("Best key:", max_key[-1])
    print("Time: {:.2f} sec ({:.3f} sec per generation)".format(t_f-t_i, (t_f-t_i)/ngen))

    print_to_file(filename, decode_vigenere_cipher, text, "".join(max_key[-1]))

    print()
    print("Decrypted text saved in file:", print_name)



    return 0



#RUN
if __name__ == '__main__':
    main()
