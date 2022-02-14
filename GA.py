import numpy as np
import matplotlib.pyplot as plt

def get_individual():
    if type_random == 'int':
        return np.random.randint(range_of_gen[0], range_of_gen[1] + 1, n_gen)
    return np.random.uniform(range_of_gen[0], range_of_gen[1], n_gen)

# Khởi tạo quần thể BỘ THAM SỐ
def get_population():
    return np.array([get_individual() for _ in range(n_individual)])

# Chọn 1 BỘ THAM SỐ
def selection(sort_population):
    index1 = np.random.randint(0, n_individual-1)
    index2 = np.random.randint(0, n_individual-1)
    
    while index1 == index2:
        index2 = np.random.randint(0, n_individual-1)
    
    return sort_population[max(index1, index2)]

# Chéo 2 BỘ THAM SỐ
def cross_individual(individual1, individual2):
    
    individual1, individual2 = individual1.copy(), individual2.copy()
    
    prob = np.random.random(size = n_gen) < rate_cross
    individual1[prob], individual2[prob] = individual2[prob], individual1[prob]

    return individual1, individual2

# Đột biến 1 BỘ THAM SỐ
def mutate_individual(individual):
    prob= np.random.random(size = n_gen) < rate_mutate
    individual[prob] = get_individual()[prob]
    return individual

def heuristic_funtion(population):
    return np.sum(population, axis=-1)

n_gen = 200            
range_of_gen = (0,1)   
type_random = 'int'    
rate_cross = 0.9       # tỷ lệ cross over
rate_mutate = 0.05     # tỷ lệ Mutation

n_epoch = 500           
elitism = 10          # giữ lại elitism 
n_individual = 500    # số lượng cá thể trong 1 quần thể
individual_optimal = None # lưu lại cá thể tốt nhất
losses = []               # loss

population = get_population()
for _ in range(n_epoch):
    
    population = population[np.argsort(heuristic_funtion(population))]

    for i in range((n_individual - elitism)//2):

        # selection
        in1 = selection(population)
        in2 = selection(population)
        
        # cross individual
        in1, in2 = cross_individual(in1, in2)
        
        # mutate individual
        in1 = mutate_individual(in1)
        in2 = mutate_individual(in2)

        population[i*2] = in1
        population[i*2+1] = in2

    losses.append(1/(heuristic_funtion(population[-1:])+1e-7))

individual_optimal = population[-1]

print(individual_optimal)