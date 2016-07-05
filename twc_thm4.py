from config_2 import *
import math

P = P_MATRIX
N = N_CHANNELS

w_0 = P[0][1]/(P[0][1]+P[1][0])

C = w_0*(1 - math.pow(P[1][1]-P[0][1], N))

D = w_0*(1-((math.pow(P[1][1]-P[0][1], N+1))*(1-P[1][1]))/(1-math.pow(P[1][1], 2) + P[1][1]*P[0][1]))

thru_avg_low = C / (C + (1-D+C)*(1-P[1][1]))

thru_avg_up = w_0/(1-P[1][1]+w_0)

print 'average througput lower bound is %f, and upper bound is %f' %(thru_avg_low, thru_avg_up)

print w_0