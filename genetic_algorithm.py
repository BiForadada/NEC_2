import numpy as np
import random
import matplotlib.pyplot as plt
import sys

MUTATION_PROB = 0.4
CROSSOVER_PROB = 0.2
POPULATION_SIZE = 200
NUMBER_OF_GENERATIONS = 600
STATIONARY_THRESHOLD = 100
FILE_PATH = 'file.txt'

def parse_input_data(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Read the header of the file, which specifies the number of jobs and machines
    header = lines[0].split()
    num_jobs, num_machines = int(header[0]), int(header[1])
    
    job_data = []
    for i in range(1, num_jobs + 1):
        # Read each of the lines that has the format: [machine time machine time ...]
        line_values = list(map(int, lines[i].split()))
        # Convert flat list [m, t, m, t...] to list of tuples [(m, t), (m, t)...]
        job_ops = [(line_values[j], line_values[j+1]) for j in range(0, len(line_values), 2)]
        job_data.append(job_ops)
    return num_jobs, num_machines, job_data

def compute_total_time(chromosome, num_jobs, num_machines, job_data):
    m_free = np.zeros(num_machines)             # Tracks when each machine becomes available for a new task (in time)
    j_free = np.zeros(num_jobs)                 # Tracks when each job finishes its current operation and can move to the next (in time)
    j_op_idx = np.zeros(num_jobs, dtype=int)    # Keeps track of which operation each job is currently on

    # Iterate through the chromosome
    for job_id in chromosome:
        # Look up the time of the nth operation of the job
        op_idx = j_op_idx[job_id]
        machine_id, duration = job_data[job_id][op_idx]
        
        # Get the start time by comparing the maximum value of when the machine or the job will be available
        start_time = max(m_free[machine_id], j_free[job_id])
        end_time = start_time + duration
        
        # Update the machine and job availability time lists
        m_free[machine_id] = end_time
        j_free[job_id] = end_time
        # Increase the job's operation index
        j_op_idx[job_id] += 1
        
    # The total time is the maximum across all the machines
    return max(m_free)

def get_schedule(chromosome, num_jobs, num_machines, job_data):
    m_free = np.zeros(num_machines)             # Tracks when each machine becomes available for a new task (in time)
    j_free = np.zeros(num_jobs)                 # Tracks when each job finishes its current operation and can move to the next (in time)
    j_op_idx = np.zeros(num_jobs, dtype=int)    # Keeps track of which operation each job is currently on
    schedule = [] # Stores the list of tuples (machine_id, job_id, start_time, duration) to plot the Gantt chart

    # Iterate through the
    for job_id in chromosome:
        # Look up the time of the nth operation of the job
        op_idx = j_op_idx[job_id]
        machine_id, duration = job_data[job_id][op_idx]
        
        # Get the start time by comparing the maximum value of when the machine or the job will be available
        start_time = max(m_free[machine_id], j_free[job_id])
        end_time = start_time + duration

        # Update the machine and job availability time lists
        m_free[machine_id] = end_time
        j_free[job_id] = end_time
        # Increase the job's operation index
        j_op_idx[job_id] += 1

        # Record the task details for visualization before updating the trackers
        schedule.append((machine_id, job_id, start_time, duration))  
        
    # Return the list of tuples with all the information to plot the result
    return schedule

def position_based_crossover(p1, p2, num_machines):
    size = len(p1)
    child = [None] * size
    # Pick random positions from parent 1
    indices = random.sample(range(size-num_machines), size // 2)
    for idx in indices:
        child[idx] = p1[idx]
    
    # Fill the remaining spots with genes from parent 2 in order
    p2_pointer = 0
    for i in range(size):
        if child[i] is None:
            while True: # Find the next gene in p2 to fill the child
                gene = p2[p2_pointer]
                # Count how many of this gene we already have in child so that it does not surpases the number of jobs
                if child.count(gene) < p1.count(gene):
                    child[i] = gene
                    p2_pointer += 1
                    break
                p2_pointer += 1
    return child

def main():
    # load the data
    num_jobs, num_m, job_data = parse_input_data(FILE_PATH)
    
    # History of the best times for the final plot
    history = []

    # Initialize the population with random values
    base = []
    for j in range(num_jobs):
        base.extend([j] * num_m)
    population = [random.sample(base, len(base)) for _ in range(POPULATION_SIZE)]

    # Variable to track a stationary stage in the training
    generations_without_improv = 0
    best_score = float('inf')

    # Start training
    for gen in range(NUMBER_OF_GENERATIONS):
        # Evaluation of all the chromosomes of the population
        scores = [(compute_total_time(c, num_jobs, num_m, job_data), c) for c in population]
        scores.sort(key=lambda x: x[0])
        
        # Get the current best score
        current_best_score = scores[0][0]
        history.append(current_best_score)

        # Check for non-improvement over STATIONARY_THRESHOLD times
        if current_best_score < best_score:
            best_score = current_best_score
            generations_without_improv = 0
        else:
            generations_without_improv += 1
        if generations_without_improv >= STATIONARY_THRESHOLD: # If the limit has been reached exit the 'training' loop
            print(f'Stationary limit reached at generation {gen}')
            break
        
        # Keep the two best scores for the next generation
        new_pop = [scores[0][1], scores[1][1]]
        
        while len(new_pop) < POPULATION_SIZE:
            # Get three random chromosomes of the population and choose the best one
            p1 = min(random.sample(scores, 3), key=lambda x: x[0])[1]
            p2 = min(random.sample(scores, 3), key=lambda x: x[0])[1]

            # Crossover algorithms
            if random.random() < CROSSOVER_PROB:
                child = position_based_crossover(p1, p2, num_m)
            else:
                child = list(p1)

            # child = list(p1)
            # Mutation algorithms
            # Change the position of two of the jobs of the item (apply the change to the probablity defined by MUTATION_PROB)
            if random.random() < MUTATION_PROB:
                idx1, idx2 = random.sample(range(len(child)), 2)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            new_pop.append(child)
        # print('the scores are:', len(new_pop))
        population = new_pop
        if gen % 20 == 0:
            print(f"Generation {gen}: Best Total Time = {scores[0][0]}")


    # Visualization the evolution of the total cost of the jobs
    plt.figure(figsize=(10, 6))
    plt.plot(history, color='blue', linewidth=2)
    plt.title("Evolution of Minimum Total Time")
    plt.xlabel("Algorithm Step (Generation)")
    plt.ylabel("Total Execution Time")
    plt.grid(True)
    print("\nDisplaying Evolution Plot. Close the window to see the Gantt Chart.")
    plt.show()  # Close the evolution plot to see the gantt diagram of the jobs

    # This visualizes the final task distribution on machines
    best_total_time, best_chromo = scores[0]
    final_sched = get_schedule(best_chromo, num_jobs, num_m, job_data)
    plt.figure(figsize=(12, 6))
    colors = plt.get_cmap('tab20', num_jobs)
    for m_id, j_id, start, dur in final_sched:
        plt.broken_barh([(start, dur)], (m_id - 0.4, 0.8), facecolors=colors(j_id), edgecolor='black')
        plt.text(start + dur/2, m_id, f"J{j_id}", va='center', ha='center', color='black', fontsize=8)
    plt.title(f"Final Job Shop Schedule (Best Total Time: {best_total_time})")
    plt.yticks(range(num_m), [f"Machine {i}" for i in range(num_m)])
    plt.xlabel("Time Units")
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    print("Displaying Gantt Chart.")
    plt.show()

if __name__ == "__main__":
    main()