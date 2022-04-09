import csv

def logTime(time):
    with open('times.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([time])


def logBestFitness(fitness):
    with open('bestFitnesses.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([fitness])


def logAverageFitness(fitness):
    with open('averageFitnesses.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([fitness])