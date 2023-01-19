# Code for Flexible job shop problem using Genetic algorithm


import copy
import math
import random
import specs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np


# Helper function for creating map of all operations
def generate_jobs_dictionary(all_tasks):

    jobs = {}
    for task in all_tasks:
        if task // 10 not in jobs.keys():
            jobs[task // 10] = [task]
        else:
            jobs[task // 10].append(task)
    return jobs


# Function for generating valid combination of chromosomes
def generate_initial(all_tasks, jobsORG, number_of_machines, num_of_parents=10):

    initial_generation = []
    machines = list(specs.spec1.keys())
    no_machine = len(machines)

    while len(initial_generation) < num_of_parents:

        jobs = copy.deepcopy(jobsORG)
        init_gen_child = []

        while len(init_gen_child) < len(all_tasks):
            job_key = random.choice(list(jobs.keys()))
            init_gen_child.append(jobs[job_key].pop(0))
            if jobs[job_key] == []:
                del jobs[job_key]

        if init_gen_child not in initial_generation:
            half_length = len(init_gen_child)

            for i in range(half_length):
                mach = random.randint(1, number_of_machines)
                duration = specs.spec1[mach][init_gen_child[i]]

                while duration >= 150:
                    mach = random.randint(1, number_of_machines)
                    duration = specs.spec1[mach][init_gen_child[i]]

                init_gen_child.append(mach)
            initial_generation.append(init_gen_child)

    return initial_generation


# Helper function for generating operations of one job
def one_job_operations(job):

    operations = specs.spec1[1].keys()
    return_operations= []
    job = int(job)

    for operation in operations:
        if operation // 10 == job:
            return_operations.append(operation)
        elif return_operations != []:
            break

    return return_operations


# Function that will kill all invalid chromosomes
def chromosome_check(chromosome):

    length = len(chromosome) // 2
    for i in range(length):
        for j in range(i + 1, length):
            if chromosome[i] > chromosome[j] > (chromosome[i] // 10) * 10:
                return False
        if specs.spec1[chromosome[length + i]][chromosome[i]] >= 150:
            return False

    return True


# Helper function for generating operations
def generate_all_operations():

    map_of_operations = {}
    keys = list(specs.spec1[1].keys())
    for j in keys:
        map_of_operations[j] = 0

    return map_of_operations


# Helper function for generating machines
def generate_all_machines():

    map_of_machines = {}
    keys = list(specs.spec1.keys())
    for j in keys:
        map_of_machines[j] = 0

    return map_of_machines


# Function that calculates the fitness of chromosomes (quality)
def fitness(chromosome):

    total = 0 # F2
    element_op = chromosome[:len(chromosome)//2]
    element_machine = chromosome[len(chromosome)//2:]

    map_operation = generate_all_operations()
    map_machine = generate_all_machines()
    map_machine_load = generate_all_machines()

    for i in range(len(element_op)):

        duration = specs.spec1[element_machine[i]][element_op[i]]
        total += duration
        map_machine_load[element_machine[i]] += duration
        on_machine = map_machine[element_machine[i]]

        if element_op[i] % 10 == 1:
            max_val = on_machine

        else:
            previous = map_operation[element_op[i] - 1]
            if previous > on_machine:
                max_val = previous
                map_machine[element_machine[i]] = previous
            else:
                max_val = on_machine

        map_operation[element_op[i]] = max_val + duration
        map_machine[element_machine[i]] = max_val + duration

    maximum = 0  # F1
    for i in map_machine.keys():
        if map_machine[i] > maximum:
            maximum = map_machine[i]

    overloaded = 0 # F3
    for i in map_machine_load.keys():
        if map_machine_load[i] > overloaded:
            overloaded = map_machine_load[i]

    # return 0.5 * maximum + 0.2 * total + 0.3 * overloaded
    # This here is one of possible heuristics
    return maximum + overloaded


# Combined sort = roulette + natural
def selection(chromosomes):

    result_list = []
    map_of_chromosome = {}
    list_of_fitness = []
    for i in range(len(chromosomes)):
        map_of_chromosome[i] = fitness(chromosomes[i])

    sorted_map = {key: val for key, val in sorted(map_of_chromosome.items(), key=lambda ele: ele[1])} # by values
    val = 1
    map_temp = {}

    for i in sorted_map.keys():
        map_temp[i] = val
        val += 1

    for i in range(len(chromosomes) // 2):

        map1 = copy.deepcopy(map_temp)
        for j in map1:
            map1[j] *= random.uniform(0, 1)

        mini = math.inf
        imini = -1
        for j in map1:
            if map1[j] < mini:
                mini = map1[j]
                imini = j

        del map1[imini]
        mini1 = math.inf
        imini1 = -1
        for j in map1:
            if map1[j] < mini1:
                mini1 = map1[j]
                imini1 = j

        tmp_list = []
        tmp_list.append(chromosomes[imini])
        tmp_list.append(chromosomes[imini1])
        result_list.append(tmp_list)

    return result_list


# Helper function for crossover where the first half of chromosome is checked
def half_chromosome_check(chromosome):
    length = len(chromosome)
    for i in range(length):
        for j in range(i + 1, length):
            if chromosome[i] > chromosome[j] > (chromosome[i] // 10) * 10:
                return False
    return True


# Ordered crossover
def crossover(chrom1, chrom2, number_of_machines):

    half_length = len(chrom1) // 2
    new_chrom1 = []
    new_chrom2 = []

    while True:

        j = 0
        border = random.randint(0, half_length - 1)
        genome_length = random.randint(border, half_length - 1)
        new_chrom1 = []

        for i in range(half_length):

            if border <= i < genome_length:
                new_chrom1.append(chrom2[border:genome_length][j])
                j = j + 1
            if not chrom1[i] in chrom2[border:genome_length]:
                new_chrom1.append(chrom1[i])

        if half_chromosome_check(new_chrom1):
            break

    while True:

        j = 0
        border = random.randint(0, half_length - 1)
        genome_length = random.randint(border, half_length - 1)
        new_chrom2 = []

        for i in range(half_length):

            if border <= i < genome_length:
                new_chrom2.append(chrom1[border:genome_length][j])
                j = j + 1
            if not chrom2[i] in chrom1[border:genome_length]:
                new_chrom2.append(chrom2[i])

        if half_chromosome_check(new_chrom2):
            break

    for i in range(half_length):

        machine_key = random.randint(1, number_of_machines)
        machine_spec = specs.spec1[machine_key]
        time_spec = machine_spec[new_chrom1[i]]

        while time_spec >= 150:

            machine_key = random.randint(1, number_of_machines)
            machine_spec = specs.spec1[machine_key]
            time_spec = machine_spec[new_chrom1[i]]

        new_chrom1.append(machine_key)
        machine_key = random.randint(1, number_of_machines)
        machine_spec = specs.spec1[machine_key]
        time_spec = machine_spec[new_chrom2[i]]

        while time_spec >= 150:

            machine_key = random.randint(1, number_of_machines)
            machine_spec = specs.spec1[machine_key]
            time_spec = machine_spec[new_chrom2[i]]

        new_chrom2.append(machine_key)

    return new_chrom1, new_chrom2


# Function for mutating elements
def mutation(chrom, number_of_machines):

    half_length = len(chrom) // 2
    mutate = True

    while mutate:

        gene1 = random.randint(0, half_length - 1)
        gene2 = random.randint(0, half_length - 1)

        while gene1 == gene2:
            gene1 = random.randint(0, half_length-1)
            gene2 = random.randint(0, half_length-1)

        temp1 = chrom[gene1]
        temp2 = chrom[gene2]
        chrom[gene1] = temp2
        chrom[gene2] = temp1

        if half_chromosome_check(chrom[:half_length]):
            break

        else:
            chrom[gene1] = temp1
            chrom[gene2] = temp2

    for i in range(half_length):
        machine = random.randint(1, number_of_machines)

        while specs.spec1[machine][chrom[i]] >= 150:
            machine = random.randint(1, number_of_machines)

        chrom[half_length + i] = machine

    return chrom


# Letting max 5 parents and min 5 children pass to another generation
def elitism(children, parents):

    temp_child = []
    for i in children:
        if i in parents:
            temp_child.append(i)
            children.remove(i)

    all_el = {}
    for i in range(len(parents)):
        all_el[i] = fitness(parents[i])

    for i in range(len(children)):
        all_el[i + 10] = fitness(children[i])

    sorted_map = {key: val for key, val in sorted(all_el.items(), key=lambda ele: ele[1])}
    counter = 0
    result_chromosomes = []

    for i in sorted_map.keys():
        if counter == 10:
            break
        if counter >= 5:
            if i > 9:
                result_chromosomes.append(children[i - 10])
            else:
                continue
        else:
            if i > 9:
                result_chromosomes.append(children[i - 10])
            else:
                result_chromosomes.append(parents[i])

        counter += 1

    return result_chromosomes


# Helper function that returns a dict with job ids as keys and their colors
def job_color_dictionary():

    operations = specs.spec1[1].keys()
    jobs_color_dict = {}
    colors = list(mcolors.TABLEAU_COLORS.values())

    for op in operations:
        job = op // 10
        if job not in jobs_color_dict:
            jobs_color_dict[job] = colors.pop(0)

    return jobs_color_dict


# Function for visualizing the results
def visualize(chromosome):

    element_op = chromosome[:len(chromosome) // 2]
    element_machine = chromosome[len(chromosome) // 2:]
    map_operation = generate_all_operations()
    map_machine = generate_all_machines()
    machine_names = []
    map_color_op = job_color_dictionary()

    for machine_key in map_machine.keys():
        machine_names.append("Machine " + str(machine_key))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(element_op)):
        machine_log = []
        pause_log = []
        left = []

        for name in machine_names:
            machine_log.append(0)
            pause_log.append(0)

        duration = specs.spec1[element_machine[i]][element_op[i]]
        on_machine = map_machine[element_machine[i]]

        if element_op[i] % 10 == 1:
            max_val = on_machine
        else:
            previous = map_operation[element_op[i] - 1]
            if previous > on_machine:
                max_val = previous
                pause_log[element_machine[i] - 1] = previous-on_machine
                map_machine[element_machine[i]] = previous

                for val in map_machine.values():
                    left.append(val)

                plt.barh(machine_names, pause_log, left=left, color="white")
            else:
                max_val = on_machine

        map_operation[element_op[i]] = max_val + duration
        machine_log[element_machine[i] - 1] = duration

        left1 = []
        sorted_map = {key: val for key, val in sorted(map_machine.items(), key=lambda ele: ele[0])}
        for val in sorted_map.values():
            left1.append(val)

        plt.barh(machine_names, machine_log, left=left1, color=map_color_op[element_op[i] // 10], edgecolor="black")
        map_machine[element_machine[i]] = max_val + duration

    rect = []
    color_names = []

    for posaoID in map_color_op.keys():
        color_names.append("Posao " + str(posaoID))
        ax.add_patch(patches.Rectangle((0, 0), 0, 0, fc=map_color_op[posaoID]))
        rect.append(patches.Rectangle((0, 0), 0, 0, fc=map_color_op[posaoID]))

    plt.legend(rect, color_names, loc='lower right')
    plt.show()


def main():

    machines = list(specs.spec1.keys())
    no_machine = len(machines)
    all_tasks = list(specs.spec1[random.choice(machines)].keys())
    jobs = generate_jobs_dictionary(all_tasks)
    print("Program se izvrsava...")
    # Generate initial
    chromosomes = generate_initial(all_tasks, jobs, no_machine, 10)
    previous_generation = chromosomes

    for i in range(1000):
        next_generation = []

        # Selection
        selected = selection(previous_generation)
        for j in selected:

            # Crossover
            first, second = crossover(j[0], j[1], len(machines))

            # Mutation
            if random.uniform(0, 1) < 0.1:
                first = mutation(first, len(machines))

            next_generation.append(first)
            next_generation.append(second)

        temp_gen = copy.deepcopy(next_generation)

        # Elitism
        next_generation = elitism(next_generation, previous_generation)
        previous_generation = temp_gen

    result = next_generation[0]
    visualize(result)
    print("Program je zavrsen. ")


if __name__ == "__main__":
    main()

