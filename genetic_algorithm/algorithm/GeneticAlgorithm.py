import numpy as np
import re
import subprocess
import socket
import os
import math
from pathlib import Path
from utils import project_root
from collections import defaultdict, deque


UDP_IP = "127.0.0.1"
UDP_PORT_READ_SIM = 5431
UDP_PORT_READ_BASH = 5432

class GeneticAlgorithm():
    def __init__(self):
        self.project_root = project_root()
        self.kmc_path = os.path.join(self.project_root, '../Config/KickEngine/lobKick.kmc')
        self.ballOffset_path = os.path.join(self.project_root, '../Config/Robots/Nao/Nao/kickInfo.cfg')
        self.sock_read_sim = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  # UDP
        self.sock_read_sim.bind((UDP_IP, UDP_PORT_READ_SIM))
        
        self.sock_read_bash = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)  # UDP
        self.sock_read_bash.bind((UDP_IP, UDP_PORT_READ_BASH)) 

        self.low_obs = np.array([-170, -240, -1.57, -170, -240, -1.57, -170, -240, -1.57, -185])
        self.high_obs = np.array([0, -150, 1.57, 170, -150, 1.57, 170, -150, 1.57, -140])

        # Population is an ordered list (ordered according to the fitness value)
        self.population = []
        self.num_initial_population = 8

        # Dictionary individual-fitness is a dictionary to save the fitness value 
        # of each individual
        self.individual_fitness_dict = defaultdict()

        # New_population is a queue of the new generated individuals, of which I don't 
        # know the fitness value yet. After been simulated, they must be added to population
        self.new_population = deque()

        # Probability to apply a random mutation on a specific gene of an indivudual
        self.mutation_rate = 0.2

        # Threshold to reach
        self.fitness_threshold = 800

        # Mutation factor (5%)
        self.k = 0.2

    # Receive message from SimRobot
    def receive_message(self):
        data, addr = self.sock_read_sim.recvfrom(1024)  # buffer size is 1024 bytes
        message = data.decode('utf-8')
        print("ENV  ->   received message:  ", message)
        return message
    

    # Close all connections
    def close(self):
        self.sock_read_sim.close()
        self.sock_read_bash.close()
        result = subprocess.run(os.path.join(self.project_root, "genetic_algorithm/scripts/kill_pid.sh"))

    # Modify KMC
    def modify_kmc_single_phase(self, state):
        line_to_modify = 106                     # linea di rightFootTra1 della quarta fase
        with open(self.kmc_path, 'r+') as file:
            lines = file.readlines()
        
        # UPDATE
        x1, z1, b1, x2, z2, b2, _ = state        # REAL
        # x2, z2, b2 = state        # REAL
        
        line1_tra1 = lines[line_to_modify]
        line1_tra1 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x1}', line1_tra1)
        line1_tra1 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z1}', line1_tra1)

        line1_tra2 = lines[line_to_modify+1]
        line1_tra2 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x2}', line1_tra2)
        line1_tra2 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z2}', line1_tra2)
    
        line1_rot1 = lines[line_to_modify + 2]
        line1_rot1 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b1}', line1_rot1)
        
        line1_rot2 = lines[line_to_modify + 3]
        line1_rot2 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b2}', line1_rot2)

        # Le due righe di ogni specifica dell'e-e le considero uguali
        lines[line_to_modify] = line1_tra1
        lines[line_to_modify + 1] = line1_tra2
        lines[line_to_modify + 2] = line1_rot1
        lines[line_to_modify + 3] = line1_rot2
        
        with open(self.kmc_path, 'w') as file:
            file.writelines(lines)

        print("ENV  ->   kmc written")

    def modify_kmc_two_phases(self, state):
        line_to_modify_1 = 82                      # linea di rightFootTra1 della terza fase
        line_to_modify_2 = 106                     # linea di rightFootTra1 della quarta fase
        with open(self.kmc_path, 'r+') as file:
            lines = file.readlines()
        
        # UPDATE
        x1, z1, b1, x2, z2, b2, x3, z3, b3, _ = state

        # Phase 1         
        line1_tra1 = lines[line_to_modify_1]
        line1_tra1 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x1}', line1_tra1)
        line1_tra1 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z1}', line1_tra1)

        line1_tra2 = lines[line_to_modify_1+1]
        line1_tra2 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x1}', line1_tra2)
        line1_tra2 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z1}', line1_tra2)
    
        line1_rot1 = lines[line_to_modify_1 + 2]
        line1_rot1 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b1}', line1_rot1)
        
        line1_rot2 = lines[line_to_modify_1 + 3]
        line1_rot2 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b1}', line1_rot2)

        # Phase 2
        line2_tra1 = lines[line_to_modify_2]
        line2_tra1 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x2}', line2_tra1)
        line2_tra1 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z2}', line2_tra1)
    
        line2_tra2 = lines[line_to_modify_2 + 1]
        line2_tra2 = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {x3}', line2_tra2)
        line2_tra2 = re.sub(r'z\s*=\s*-?\d+(\.\d+)?', f'z = {z3}', line2_tra2)
    
        line2_rot1 = lines[line_to_modify_2 + 2]
        line2_rot1 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b2}', line2_rot1)

        line2_rot2 = lines[line_to_modify_2 + 3]
        line2_rot2 = re.sub(r'y\s*=\s*-?\d+(\.\d+)?', f'y = {b3}', line2_rot2)

        # Le due righe di ogni soecifica dell'e-e le considero uguali
        lines[line_to_modify_1] = line1_tra1
        lines[line_to_modify_1 + 1] = line1_tra2
        lines[line_to_modify_1 + 2] = line1_rot1
        lines[line_to_modify_1 + 3] = line1_rot2
        
        lines[line_to_modify_2] = line2_tra1
        lines[line_to_modify_2 + 1] = line2_tra2
        lines[line_to_modify_2 + 2] = line2_rot1
        lines[line_to_modify_2 + 3] = line2_rot2

        with open(self.kmc_path, 'w') as file:
            file.writelines(lines)

        print("ENV  ->   kmc written")

    def modify_ballOffset(self, state):
        line_to_modify = 304
        _, _, _, _, _, _, _, _, _, ballOffset = state
        with open(self.ballOffset_path, 'r+') as file:
            lines = file.readlines()

        line = lines[line_to_modify]
        line = re.sub(r'x\s*=\s*-?\d+(\.\d+)?', f'x = {ballOffset}', line)
        lines[line_to_modify] = line

        with open(self.ballOffset_path, 'w') as file:
            file.writelines(lines)

        print("ENV  ->   ballOffset written")

    # Find index where to insert the individual in the population list, according to the fitness value
    def find_index(self, individual, fitness):
        # Middle transition of the buffer
        left = 0
        right = len(self.population)
        mid = 0
            
        while left < right:
            mid = (left + right) // 2
            if fitness > self.individual_fitness_dict[tuple(self.population[mid][:9])][0]:
                right = mid
            else:
                left = mid + 1
        return left

    # Create initial population (gen.0)
    def create_initial_population(self) -> None:
        while len(self.new_population) < self.num_initial_population:
            new = np.round(np.random.uniform(low=self.low_obs, high=self.high_obs), decimals=3 )
            if(self.check_constraints(new)):
                self.new_population.append(new)


    # Simulation function
    def simulate(self, individual):
        self.modify_kmc_two_phases(individual)
        self.modify_ballOffset(individual)
        result = subprocess.Popen(os.path.join(self.project_root, "genetic_algorithm/scripts/run.sh"))
        message = self.receive_message()
        return message

    # Compute fitness value of an individual
    def compute_fitness(self, individual):
        message = self.simulate(individual=individual)
        ballVX, ballVY, ballVZ, ballHeight, isFallen = message.split("|")
        # Check if individual satisfies constraints
        compliant_factor = 1 if self.check_constraints(individual) else 0.5
        fitness_value = compliant_factor * float(ballHeight)

        data, addr = self.sock_read_bash.recvfrom(1024)  # buffer size is 1024 bytes
        print(f"ENV  ->   received message: {data.decode('utf-8')}")
        print(f"ENV  ->  compliant factor:  {compliant_factor}")
        return fitness_value     

    # Compute the pairing between the individuals, which will be mate together
    def pair(self):
        indices_pairing = [(0,1), (2,3), (4,5), (6,7)]
        return indices_pairing

    # Mate together the selected population, according to the pairing indices.
    # This step generate the new generation of individuals
    def mate(self, selected_population, indices_pairing):
        new_individuals = []
        for pair in indices_pairing:
            mate1 = selected_population[pair[0]]
            mate2 = selected_population[pair[1]]

            # These new individuals come from the crossover between the two mates
            new1 = np.concatenate([mate1[:3], mate2[-7:]])
            new2 = np.concatenate([mate2[:3], mate1[-7:]])
            new3 = np.concatenate([mate1[:3], mate2[3:6], mate1[6:9], mate2[-1:]])
            new4 = np.concatenate([mate2[:3], mate1[3:6], mate2[6:9], mate1[-1:]])
            
            p = self.compute_mutation_prob(mate1, mate2)

            # 6) Apply mutation if random number is lower than mutation_rate
            self.apply_mutations(new1,p)
            self.apply_mutations(new2,p)
            self.apply_mutations(new3,p)
            self.apply_mutations(new4,p)
            # This new individual is an average of the two mates
            alpha = np.random.random()
            new5 = alpha * mate1 + (1-alpha) * mate2
            self.apply_mutations(new5, p)

            # DEBUG
            print(f"mate1:  {mate1}")
            print(f"mate2:  {mate2}")
            print(f"new1:  {new1}")
            print(f"new2:  {new2}")
            print(f"new3:  {new3}")
            print(f"new4:  {new4}")
            print(f"new5:  {new5}")
            print("------------------")

            new_individuals.extend([new1, new2, new3, new4, new5])
        return new_individuals

    # Apply a random mutation to the individual (according to the mutation rate)
    def apply_mutations(self, individual, prob):
        p = [1-prob, prob] if self.check_constraints(individual) else [self.mutation_rate, 1-self.mutation_rate]
        mutations_rates = np.random.choice([0, 1], size=len(self.high_obs), p=p)
        means = np.zeros_like(self.high_obs)
        std_devs = self.k * (self.high_obs-self.low_obs) / 6
        mutations = np.random.normal(loc=means, scale=std_devs)
        individual = individual + (mutations_rates * mutations)
        individual = np.clip(a=individual, a_min=self.low_obs, a_max=self.high_obs)

    # Dynamic probability of applying mutations, depending on how many
    # elements the two parents have
    def compute_mutation_prob(self, mate1, mate2):
        num_same_elems = np.sum(mate1==mate2)
        print("ENV  -> number same elements:  ", num_same_elems)
        if num_same_elems==0:
            return 0
        else:
            return math.log(num_same_elems) / math.log(len(mate1) + num_same_elems)


    # Check if constraints are satisfied. If they are not, 
    # penalize fitness value of a factor 0.5
    def check_constraints(self, individual):
        # Constraints to be satisfied:
        #   1) x1 <= x2
        return True if (individual[0] <= individual[3] <= individual[6]) else False  
        

    # In this function all the steps of the algorithm are executed, 
    # then the simulation of the new created population is launched
    def learn(self):
        # Steps of the algorithm:
        #   1) Create initial population
        #   2) Compute fitness function for each individual of the population
        #   3) Select candidates of the population, according to their fitness values
        #   4) Pair (decidi l'accoppiamento) candidates between them according to some policy
        #   5) Mate (effettua accoppiamento) candidates between them according to some policy
        #   6) Random mutations on the new population
        # Once the new individuals are created, they are integrated inside the population.
        # At this point we can restart the algorithm with the updated population.
        
        # Links:
        #   https://www.cs.us.es/~fsancho/ficheros/IA2019/TheContinuousGeneticAlgorithm.pdf
        #   https://towardsdatascience.com/continuous-genetic-algorithm-from-scratch-with-python-ff29deedd099


        # 1)
        self.create_initial_population()
        # Generation index
        gen = 0
        while len(self.population)==0 or self.individual_fitness_dict[tuple(self.population[0][:9])][0] < self.fitness_threshold:
            print(f"-------->   GENERATION  {gen}")
            # 2)
            while len(self.new_population) > 0:
                individual = self.new_population.popleft()
                fitness = self.compute_fitness(individual=individual) 
                print(f"ENV  ->  individual:    {individual}")
                print(f"ENV  ->  fitness value:    {fitness}")
                print(" --------------------")

                # Save individual fitness value in the dictionary
                # Must be converted in tuple because list is not hashable  
                if tuple(individual[:9]) not in self.individual_fitness_dict or \
                    fitness > self.individual_fitness_dict[tuple(individual[:9])][0]:
                    self.individual_fitness_dict[tuple(individual[:9])] = (fitness, individual[-1], gen)
                
                # Insert the individual in the i-th position of the population list
                i = self.find_index(individual=individual, fitness=fitness)
                self.population.insert(i, individual)

            # 3)
            selected_population = self.population[:8]
            
            # Print the elite population
            print(f"....... ELITE POPULATION .........")
            for i in range(len(selected_population)):
                print(f"{i+1})   {selected_population[i]}")
                print(f"            fitness: {self.individual_fitness_dict[tuple(selected_population[i][:9])][0]}")
                print(f"            distance: {self.individual_fitness_dict[tuple(selected_population[i][:9])][1]}")
                print(f"            gen: {self.individual_fitness_dict[tuple(selected_population[i][:9])][2]}")

            # 4)
            indices_pairing = self.pair()

            # 5-6)
            new_individuals = self.mate(selected_population, indices_pairing)
            # Add new individuals to the new population queue
            self.new_population.extend(new_individuals)

            gen += 1            

        print("######       FINAL SOLUTION      #######")
        print(f"individual: {self.population[0]}")
        print(f"fitness: {self.individual_fitness_dict[tuple(self.population[0])][0]}")
        print(f"gen: {self.individual_fitness_dict[tuple(self.population[0])][1]}")
        
        


            


