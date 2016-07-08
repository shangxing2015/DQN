from config_2 import *
import math

P_test = [(0.4, 0.6), (0.55, 0.45)]
GOOD = False
P = P_test

#P = P_MATRIX
N = N_CHANNELS

w_0 = P[0][1]/(P[0][1]+P[1][0])

if GOOD:

    C = w_0*(1 - math.pow(P[1][1]-P[0][1], N))

    D = w_0*(1-((math.pow(P[1][1]-P[0][1], N+1))*(1-P[1][1]))/(1-math.pow(P[1][1], 2) + P[1][1]*P[0][1]))

    thru_avg_low = C / (C + (1-D+C)*(1-P[1][1]))

    thru_avg_up = w_0/(1-P[1][1]+w_0)

    print 'average througput lower bound is %f, and upper bound is %f' %(thru_avg_low, thru_avg_up)

else:

    p_temp = P[1][0]*P[0][0] + P[1][1]*P[1][0]

    F = (1-P[0][1])*(1-w_0)*(1/(2-P[0][1]) - (P[0][1]*math.pow(P[1][1]-P[0][1], 4))/(1-math.pow(P[1][1]-P[0][1], 2)*math.pow(1-P[0][1], 2)))

    E = p_temp*(1+P[0][1]) + P[0][1]*(1-F)

    G = (1-w_0)*(1/(2-P[0][1])-P[0][1]*math.pow(P[1][1]-P[0][1],6)/(1-math.pow(P[1][1]-P[0][1], 2)*math.pow(1-P[0][1], 2)))

    H = (1-w_0)*(1/(2-P[0][1]) - P[0][1]*math.pow(P[1][1]-P[0][1], 2*N-1)/(1-math.pow(P[1][1]-P[0][1], 2)*math.pow(1-P[0][1], 2)))

    thru_avg_low = 1 - p_temp/(E - P[0][1]*H)
    thru_avg_up = 1- p_temp/(E-P[0][1]*G)

    print 'average througput lower bound is %f, and upper bound is %f' % (thru_avg_low, thru_avg_up)

print w_0