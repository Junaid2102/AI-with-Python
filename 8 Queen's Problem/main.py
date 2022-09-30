import numpy as np
import random as rd
import heapq

# Initializing the boards
def initialize(board):
    for i in range(8):
        board[rd.randint(0,7)][i]=1
    return board

# Converting boards to list
def makelist(board):
    l = list()
    for i in range(8):
        for j in range(8):
            if(board[j][i] == 1):
                l.append(j)
            else:
                continue
    return l

# Horizontal Check
def check_horizontal(board,x,y):
    sum = 0
    for i in range(8):
        if i != y:
            if board[x][i] == 1:
                sum += 1
    return sum

# Check top left diagonal:
def top_left(board,x,y):
    sum = 0
    i = x-1
    j = y-1
    while i>=0 and j>=0:
        if board[i][j] == 1:
            sum += 1
        i -= 1
        j -= 1
    return sum

# Check bottom left diagonal
def bottom_left(board,x,y):
    sum = 0
    i = x + 1
    j = y - 1
    while i<8 and j>=0:
        if board[i][j] == 1:
            sum += 1
        i += 1
        j -= 1
    return sum

# Check top right diagonal
def top_right(board,x,y):
    sum = 0
    i = x - 1
    j = y + 1
    while i >= 0 and j <= 7:
        if board[i][j] == 1:
            sum += 1
        i -= 1
        j += 1
    return sum

# Check bottom right diagonal
def bottom_right(board,x,y):
    sum = 0
    i = x + 1
    j = y + 1
    while i < 8 and j < 8:
        if board[i][j] == 1:
            sum += 1
        i += 1
        j += 1
    return sum

# Fitness function
def fitness(board,l):
    f = 0
    for i in range(8):
        if(check_horizontal(board,l[i],i) == 0 and top_left(board,l[i],i) == 0 and bottom_left(board,l[i],i) == 0 and top_right(board,l[i],i) == 0 and bottom_right(board,l[i],i) == 0):
            f += 1
    return f

# getting Best 2 Boards
def get(Fitness, boards):
    best = heapq.nlargest(2,Fitness)
    i = 0
    get = list()
    for j in best:
        if((j == Fitness[0] and i == 0) or (j == Fitness[0] and i == 1 and makelist(get[0]) != makelist(boards[0]))):
            get.append(boards[0])
        elif ((j == Fitness[1] and i == 0) or (j == Fitness[1] and i == 1 and makelist(get[0]) != makelist(boards[1]))):
            get.append(boards[1])
        elif ((j == Fitness[2] and i == 0) or (j == Fitness[2] and i == 1 and makelist(get[0]) != makelist(boards[2]))):
            get.append(boards[1])
        elif (j == Fitness[3]):
            get.append(boards[3])
        j += 1
    if(len(get) == 1):
        get.append(get[0])
    return get

# Crossover
def crossover(B1,B2):
    fc = np.arange(8)
    sc = np.arange(8)
    p1 = B1
    p2 = B2
    for i in range(0,4):
        fc[i] = p1[i]
    for i in range(4,8):
        fc[i] = p2[i]
    for i in range(0,4):
        sc[i] = p2[i]
    for i in range(4,8):
        sc[i] = p1[i]
    new = np.array((p1,p2,fc,sc))
    return new

# Board from List
def baordcreation(pop):
    Board = np.zeros((8,8))
    for i in range(8):
        Board[pop[i]][i] = 1
    return Board

# Mutation
def mutation(total):
    in1 = rd.randint(0,7)
    in2 = rd.randint(0, 7)
    in3 = rd.randint(0, 7)
    in4 = rd.randint(0, 7)
    total[2][in1] = rd.randint(0,7)
    total[3][in2] = rd.randint(0, 7)
    total[2][in3] = rd.randint(0, 7)
    total[3][in4] = rd.randint(0, 7)
    return total

# Make 4 Chess Boards
board1 = np.zeros((8,8))
board2 = np.zeros((8,8))
board3 = np.zeros((8,8))
board4 = np.zeros((8,8))

# Place queen Randomly
board1 = initialize(board1)
board2 = initialize(board2)
board3 = initialize(board3)
board4 = initialize(board4)
Boards = [board1,board2,board3,board4]
#print(board4)

# Boards into list conversion
l1 = makelist(board1)
l2 = makelist(board2)
l3 = makelist(board3)
l4 = makelist(board4)
#print(l2)

# Computing Fitness
f1 = fitness(board1, l1)
f2 = fitness(board1, l2)
f3 = fitness(board1, l3)
f4 = fitness(board1, l4)
fit = [f1,f2,f3,f4]
#print(fit)

index = 0
population = np.random.randint(0, 7, size=(4, 8))
while (max(fit) < 8 and index < 5000):
    # Keep 2 Best
    best2 = get(fit, Boards)
    # Convert 2 best to list for crossover
    Bl1 = makelist(best2[0])
    Bl2 = makelist(best2[1])
    # Crossover & Mutation
    population = crossover(Bl1,Bl2)
    population = mutation(population)
    # convert list to boards
    b1 = baordcreation(population[0])
    b2 = baordcreation(population[1])
    b3 = baordcreation(population[2])
    b4 = baordcreation(population[3])
    Boards = [b1,b2,b3,b4]
    # Compute fitness for 4 boards
    for i in range(4):
        fit[i] = fitness(Boards[i], population[i])
    index += 1

# Print board with max fitness
m = max(fit)
print("Max Fitness is \n", m)
for k in range(4):
    if(fit[k] == m):
        print("Final Board is: \n", population[k])
        print(Boards[k])
        print("Fitness is: \n", m)
        print("Number of iterations is: ", index)
        break

#print(check_horizontal(board1, l1[2],2))
#print(top_left(board1,l1[2],2))
#print(bottom_left(board1,l1[2],2))
#print(top_right(board1,l1[2],2))
#print(bottom_right(board1,l1[2],2))