import os
import csv
import matplotlib.pyplot as plt
import numpy as np

pwd = os.getcwd()


def readFile(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

times = readFile(pwd + '/multiprocess_canny/times.csv')
bestFitnesses = readFile(pwd + '/multiprocess_canny/bestFitnesses.csv')
averageFitnesses = readFile(pwd + '/multiprocess_canny/averageFitnesses.csv')

generations = len(times)



x = np.arange(generations)
# print(x.shape)
y1 = np.squeeze(np.array(bestFitnesses))
# print(y1.shape)
y2 = np.squeeze(np.array(averageFitnesses))
# print(y2.shape)

# # x = np.arange(1, 11)
# # print(x.shape)
# # y = np.array([100, 10, 300, 20, 500, 60, 700, 80, 900, 100])
# # y = np.array([[100], [10], [300], [20], [500], [60], [700], [80], [900], [100]])
# # print(y.shape)

# # plot a line graph between the best and average fitnesses

plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('Best and Average Fitnesses')
plt.plot(x, y1, label='Best Fitness')
plt.plot(x, y2, label='Average Fitness')
plt.legend()
plt.show()


# # data to be plotted
# x = np.arange(1, 11)
# y = np.array([100, 10, 300, 20, 500, 60, 700, 80, 900, 100])
 
# # plotting
# plt.title("Line graph")
# plt.xlabel("X axis")
# plt.ylabel("Y axis")
# plt.plot(x, y, color ="green")
# plt.show()