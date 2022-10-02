import numpy as np
from matplotlib import pyplot as plt

# Reading from Testing Data
testx = np.genfromtxt("testX.txt")
testy = np.genfromtxt("testY.txt")

# Reading from Training Data
trainx = np.genfromtxt("trainX.txt")
trainy = np.genfromtxt("trainY.txt")
# print(len(trainx))
# print(len(trainy))

# Separating Values 2 and 4 from trainx.txt
train2 = trainx[:250, :]
train4 = trainx[250:, :]
# print(len(train2))
# print(len(train4))

# Printing image of 2
plt.imshow(np.reshape(trainx[200],(16,16),order='F'))
plt.title('2 figure')
plt.show()

# Printing image of 4
plt.imshow(np.reshape(trainx[256],(16,16),order='F'))
plt.title('4 figure')
plt.show()

# Calculating probability of P(x= 1 given 2) & P(x= 1 given 4)
trainprob1_2 = (train2.sum(axis=0)+1)/(train2.shape[0]+2)
trainprob1_4 = (train4.sum(axis=0)+1)/(train4.shape[0]+2)
# Added 1 and 2 for navier bayes smothering

# Calculating probability of P(x= 0 given 2) & P(x= 0 given 4)
trainprob0_2 = 1 - trainprob1_2
trainprob0_4 = 1 - trainprob1_4
#print(trainprob1_2)

# As half are 2 and half are 4 so P(2) or P(4) is equal to 0.5
Pfor2_4 = 0.5

# Testing Data
test_out = []
for a in range(0, testx.shape[0]):
    P2ts = 1
    P4ts = 1
    for b in range(0, testx.shape[1]):
        if testx[a,b] == 1:
            P2ts = P2ts * trainprob1_2[b]
            P4ts = P4ts * trainprob1_4[b]
        elif testx[a,b] == 0:
            P2ts = P2ts * trainprob0_2[b]
            P4ts = P4ts * trainprob0_4[b]
    P2ts = P2ts*Pfor2_4
    P4ts = P4ts*Pfor2_4
    if(P2ts > P4ts):
        test_out.append(2)
    else:
        test_out.append(4)
#print(len(test_out))

# Accuracy of Testing Data
acc = 0
two_acc = 0
four_acc = 0
for x in range(0,testx.shape[0]):
    if testy[x] == test_out[x]:
        acc = acc + 1
        if test_out[x] == 2:
            two_acc = two_acc + 1
        elif test_out[x] == 4:
            four_acc = four_acc + 1

#print(len(testy[testy == 4]))
all_acc = (acc/testx.shape[0])*100
all_two_acc = ((two_acc/len(testy[testy == 2]))*100)
all_four_acc = ((four_acc/len(testy[testy == 4]))*100)
print("~~~~~~~~ Testing Data Accuracy ~~~~~~~~")
print("Overall Accuracy =", all_acc, "%")
print("Class 2 Accuracy =", all_two_acc,"%")
print("Class 4 Accuracy =", all_four_acc,"%")

# Training Data
train_out = []
for a in range(0, trainx.shape[0]):
    P2tn = 1
    P4tn = 1
    for b in range(0, trainx.shape[1]):
        if trainx[a,b] == 1:
            P2tn = P2tn * trainprob1_2[b]
            P4tn = P4tn * trainprob1_4[b]
        elif trainx[a,b] == 0:
            P2tn = P2tn * trainprob0_2[b]
            P4tn = P4tn * trainprob0_4[b]
    P2tn = P2tn * Pfor2_4
    P4tn = P4tn * Pfor2_4
    if(P2tn > P4tn):
        train_out.append(2)
    else:
        train_out.append(4)
#print(len(train_out))

# Accuracy of Training Data
acc = 0
two_acc = 0
four_acc = 0
for x in range(0,len(train_out)):
    if train_out[x] == trainy[x]:
        acc = acc + 1
        if train_out[x] == 2:
            two_acc = two_acc + 1
        elif train_out[x] == 4:
            four_acc = four_acc + 1

#print(len(testy[testy == 4]))

#print(len(trainy[trainy == 2]))
all_acc = (acc/trainx.shape[0])*100
all_two_acc = ((two_acc/len(trainy[trainy == 2]))*100)
all_four_acc = ((four_acc/len(trainy[trainy == 4]))*100)
print("\n~~~~~~~~ Training Data Accuracy ~~~~~~~~")
print("Overall Accuracy =", all_acc, "%")
print("Class 2 Accuracy =", all_two_acc, "%")
print("Class 4 Accuracy =", all_four_acc, "%")