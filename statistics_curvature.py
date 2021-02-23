import math

import numpy as np
import csv
from scipy.special import comb

'''
These parameters need to be changed according to your study. 
filepath = name of the CSV
term = This handles multiple conditions for the same list of objects. The CSV should say Term_Object1_Object2 for a column header. ex. Flat_Sphere_Cylinder
s = number of participants
shapes = dictionary of your objects being compared from 0 to n. Make sure they are spelled exactly like they are on the CSV
'''
filepath = r'Perception of 3D Curvature_February 22, 2021_15.21.csv'
term = "F"
s = 15
shapes = {'Cube': 0, 'Boat': 1, 'Cylinder': 2, 'Cow': 3, 'Rocker': 4, 'Sphere': 5}

# some global variables needed in equations, don't change
t = len(shapes)
cSum = 0

'''
input: CSV row of a participant
output: zeta (coefficient of consistency), preference matrix of participant
'''
def get_zeta(row):
    # needs a global cSum to calculate average zeta, this could probably be returned instead
    global cSum

    # sets the shape_matrix to 0
    shape_matrix = np.zeros((t, t))
    for item in row:
        item_array = item.split("_")

        # if we are looking at the right term (C, F, S)
        if item_array[0] == term:
            # if they chose the first object in the pair, assign the [FirstItemIndex, SecondItemIndex] of the pair to 1
            if int(row[item]) == 1:
                shape_matrix[shapes[item_array[1]]][shapes[item_array[2]]] = 1
            # if they chose the second object in the pair, assign the [SecondItemIndex, FirstItemIndex] of the pair to 1
            else:
                shape_matrix[shapes[item_array[2]]][shapes[item_array[1]]] = 1

    # sums the score of each row (p_i) to calculate number of circular triads
    # T is the summation that uses the sum of each row score according to the formula
    p_i = sum(shape_matrix.T)
    T = sum((p_i - ((t - 1) / 2)) ** 2)

    # calculates number of circular triads and zeta
    c = ((t / 24.0) * ((t ** 2) - 1)) - (T / 2)
    zeta = 1 - ((24 * c) / (t ** 3 - 4 * t))

    # adds to global number of circular triads
    cSum = cSum + c
    return zeta, shape_matrix

'''
input: p_i - sum of scores in the overall preference matrix, pref_matrix - overall preference matrix of all participants
output: prints coefficient of agreement and chisquared
'''
def print_coefficient_agreement(p_i, pref_matrix):
    asigma = 0
    for num in p_i:
        asigma = asigma + (num ** 2)
    Sigma = 0
    for i in range(t):
        for j in range(t):
            if i != j:
                Sigma += comb((pref_matrix[i, j]).astype(int), 2)
    print('Sigma = ', Sigma)
    mu = -1 + (2 * Sigma / (comb(s, 2) * comb(t, 2)))
    print('mu (coefficient of agreement) = ', mu)
    chisquared = (t * (t - 1) * (1 + (mu * (s - 1)))) / 2
    print('Chisquared of coefficient of agreement = ', chisquared)


'''
input: none
output: Durbin's number for a Chi-squared test to see if there are significant differences between the scores.
You should check to see that this number is greater than the chi-squared with t-1 degrees of freedom 
If it is, you can proceed to Least Significant Difference to group items
'''
def print_durbin():
    asigma = 0
    for num in p_i:
        asigma = asigma + (num ** 2)
    D = (4 * (asigma - (0.25 * t * s * s * (t - 1) * (t - 1)))) / (t * s)
    print("Durbin's Number for 5% significance", D)


'''
Main function 
Prints out total preference matrix, total scores, average zeta (coefficient of consistency), and coefficient of agreement
'''
if __name__ == '__main__':
    # fills the overall preference matrix with 0s
    total_pref_matrix = np.zeros((t, t))
    avgZeta = 0
    count = 0

    # open file and read each line (1 line = 1 participant)
    with open(filepath) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        # get coefficient of consistency and preference matrix for each participant to find the average
        for row in csv_reader:
            zeta, participant_pref_matrix = get_zeta(row)
            # add participant preference matrix to total preference matrix
            total_pref_matrix = np.add(total_pref_matrix, participant_pref_matrix)
            # print("Coefficient of consistency: ", zeta)
            avgZeta += zeta
            count += 1

    # print overall preference matrix and rankings
    print('Overall preference matrix:')
    print(total_pref_matrix.astype(int))
    p_i = sum(total_pref_matrix.T)
    print("\nTotal Scores:\n", p_i, end="\n\n")
    print(shapes)

    print("\n=========================")
    # print stats
    print_coefficient_agreement(p_i, total_pref_matrix)
    avgZeta /= count
    print("Average zeta: ", avgZeta)
    print_durbin()
    lsd = 1.96 * math.sqrt((0.5 * s * t)) + 0.5
    print("Least significant difference", lsd)
