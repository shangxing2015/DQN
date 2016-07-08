from config_2 import *
import math
import heapq


class WhittleIndex:
    
    def __init__(self, B, gamma):
        
        self.p_matrix = [P_MATRIX for i in range(N_CHANNELS)]
        self.B = B
        self.gamma = gamma
        self.init_belief = [p[0][1]/(p[0][1]+p[1][0]) for p in self.p_matrix]
        self.belief = self.init_belief
        self.whittle_idx = [0 for i in range(N_CHANNELS)]
        if DISCOUNT:
            self._whittle_index_discounted()
        else:
            self._whittle_index_average()

        
        
    def _update_belief(self, action, observation):
        
        for i in range(N_CHANNELS):
            if i in action:
                if observation[i] == 1:
                    self.belief[i] = self.p_matrix[i][1][1]
                else:
                    self.belief[i] = self.p_matrix[i][0][1]
            else:
                self.belief[i] = self.belief[i]*self.p_matrix[i][1][1] + (1-self.belief[i])*self.p_matrix[i][0][1]
                
                
    
    def _k_step_belief_update(self, k, omega, chan_idx):

        p = self.p_matrix[chan_idx]
        
        Tk_omega = (p[0][1] - math.pow(p[1][1]-p[0][1], k)
                    *(p[0][1] - (1+p[0][1]-p[1][1])*omega)) / (1+p[0][1]-p[1][1])
        
        return Tk_omega
    
    
    
    def _L_pos_func(self, x, y, chan_idx):
        
        if x > y:
            return 0
        elif x <= y and y >= self.belief[chan_idx]:
            return float('infinity')
        else:
            p = self.p_matrix[chan_idx]
            temp = (p[0][1] - y*(1-p[1][1]+p[0][1]))/(p[0][1] - x*(1-p[1][1]+p[0][1]))
            return math.floor(math.log(temp, p[1][1]-p[0][1])) + 1

    def _L_neg_func(self, x, y, chan_idx):

        p = self.p_matrix[chan_idx]

        T = x*p[1][1] + (1-x)*p[0][1]

        if x > y:
            return 0
        elif x <= y and T > y:
            return 1
        else:
            return float('infinity')


    def _whittle_index_discounted(self):

        for i in range(N_CHANNELS):

            p = self.p_matrix[i]

            if p[1][1] >= p[0][1]:

                c_1_num = (1-self.gamma*p[1][1]) * (1 - math.pow(self.gamma, self._L_pos_func(p[0][1], self.belief[i], i)))
                c_1_denum_part_1 = (1 - self.gamma*p[1][1])*(1-math.pow(self.gamma, self._L_pos_func(p[01], self.belief[i], i)+1))
                c_1_denum_part_2 = (1-self.gamma)*math.pow(self.gamma, self._L_pos_func(p[01], self.belief[i], i)+1)*self._k_step_belief_update(self._L_pos_func(p[0][1], self.belief[i], i), p[0][1], i)
                c_1 = c_1_num / (c_1_denum_part_1+c_1_denum_part_2)

                c_2_num = math.pow(self._L_pos_func(p[0][1], self.belief[i], i))*self._k_step_belief_update(self._L_pos_func(p[0][1], self.belief[i], i), p[0][1], i)
                c_2 = c_2_num / (c_1_denum_part_1+c_1_denum_part_2)
        
                if self.belief[i] <= p[0][1] or self.belief[i] >= p[1][1]:
                    self.whittle_idx[i] = self.belief[i]*self.B[i]
                elif self.belief[i] >= self.init_belief[i] and self.belief[i] < p[1][1]:
                    self.whittle_idx[i] = self.belief[i]*self.B[i] / (1-self.gamma*p[1][1]+self.gamma*self.belief[i])
                else:
                    temp_num = self.belief[i] - self.gamma*self._k_step_belief_update(1, self.belief[i], i) + c_2*(1-self.gamma)*(self.gamma*(1-self.gamma*p[1][1])-self.gamma*(self.belief[i]-self.gamma*self._k_step_belief_update(1, self.belief[i], i)))

                    temp_denum = 1 - self.gamma*p[1][1] - c_1*(self.gamma*(1-self.gamma*p[1][1])-self.gamma*(self.belief[i] - self.gamma*self._k_step_belief_update(1, self.belief[i], i)))
                    self.whittle_idx[i] = temp_num*self.B[i]/temp_denum


            else:
                c_3_num = 1-self.gamma*(1-p[0][1])
                c_3_denum = 1 + (1+self.gamma)*self.gamma*p[0][1] - math.pow(self.gamma, 2)*self._k_step_belief_update(1, p[1][1], i)
                c_3 = c_3_num / c_3_denum
                c_4_num = self.gamma*self._k_step_belief_update(1, p[1][1], i)*(1-self.gamma) + math.pow(self.gamma, 2)*p[0][1]
                c_4_denum = 1 + (1+self.gamma)*self.gamma*p[0][1] - math.pow(self.gamma, 2)*self._k_step_belief_update(1, p[1][1], i)
                c_4 = c_4_num/c_4_denum

                if self.belief[i] <= p[1][1] or self.belief[i] >= p[0][1]:
                    self.whittle_idx[i] = self.belief[i] * self.B[i]
                elif self.belief[i] >= self._k_step_belief_update(1, p[1][1], i) and self.belief[i] < p[0][1]:
                    self.whittle_idx[i] = (self.gamma*p[0][1] + self.belief[i]*(1-self.gamma))*self.B[i] / (1+self.gamma*(p[0][1]-self.belief[i]))
                elif self.belief[i] >= self.init_belief[i] and self.belief[i] < self._k_step_belief_update(1, p[1][1], i):
                    temp_num = (1-self.gamma+self.gamma*c_4)*(self.gamma*p[0][1]+self.belief[i]*(1-self.gamma))
                    temp_denum = 1 - self.gamma*(1-p[0][1]) - c_3*(math.pow(self.gamma,2)*p[0][1]+self.gamma*self.belief[i]-math.pow(self.gamma,2)*self.belief[i])
                    self.whittle_idx[i] = temp_num*self.B[i]/temp_denum
                else:
                    temp_num = (1-self.gamma)*(self.gamma*p[0][1]+self.belief[i]-self.gamma*self._k_step_belief_update(1, self.belief[i], i)) - c_4*self.gamma*(self.gamma*self._k_step_belief_update(1, self.belief[i], i)-self.gamma*p[0][1]-self.belief[i])
                    temp_denum = 1 - self.gamma*(1-p[0][1]) + c_3*self.gamma*(self.gamma*self._k_step_belief_update(1, self.belief[i], i)-self.gamma*p[0][1]-self.belief[i])
                    self.whittle_idx[i] = temp_num*self.B[i] / temp_denum

    def _whittle_index_average(self):

        for i in range(N_CHANNELS):

            p = self.p_matrix[i]



            if p[1][1] >= p[0][1]:

                if self.belief[i] <= p[0][1] or self.belief[i] >= p[1][1]:
                    self.whittle_idx[i] = self.belief[i]*self.B[i]
                elif self.belief[i] > p[0][1] and self.belief[i] < self.init_belief[i]:
                    temp_num = (self.belief[i]-self._k_step_belief_update(1, self.belief[i], i))*(self._L_pos_func(p[0][1], self.belief[i], i)+1) + self._k_step_belief_update(self._L_pos_func(p[0][1], self.belief[i], i), p[0][1], i)
                    temp_denum = 1-p[1][1]+(self.belief[i]-self._k_step_belief_update(1,self.belief[i], i)*self._L_pos_func(p[0][1], self.belief[i], i) + self._k_step_belief_update(self._L_pos_func(p[0][1], self.belief[i], i))*p[0][1])
                    self.whittle_idx[i] = temp_num*self.B[i]/temp_denum
                else:
                    self.whittle_idx[i] = self.belief[i]*self.B[i]/(1-p[1][1]-self.belief[i])
            else:

                if self.belief[i] <= p[1][1] or self.belief[i] > p[0][1]:
                    self.whittle_idx[i] = self.belief[i]*self.B[i]
                elif self.belief[i] > p[1][1] and self.belief[i] < self.init_belief[i]:
                    temp_num = self.belief[i] + p[0][1] - self._k_step_belief_update(1, self.belief[i], i)
                    temp_denum = 1 + p[0][1] - self._k_step_belief_update(1, p[1][1], i) +self._k_step_belief_update(1, self.belief[i], i) - self.belief[i]
                    self.whittle_idx[i] = temp_num*self.B[i]/temp_denum
                elif self.belief[i] >= self.init_belief[i] and self.belief[i] < self._k_step_belief_update(1, p[1][1], i):
                    temp_num = p[0][1]
                    temp_denum = 1+p[0][1]-self._k_step_belief_update(1, p[1][1], i)
                    self.whittle_idx[i] = temp_num*self.B[i]/temp_denum
                else: #condition?
                    self.whittle_idx[i] = p[0][1]*self.B[i]/(1+p[0][1]-self.belief[i])


    def getAction(self, prev_action, observation):

        print observation

        self._update_belief(prev_action, observation)

        if DISCOUNT:
            self._whittle_index_discounted()
        else:
            self._whittle_index_average()

        print self.whittle_idx

        action_temp = heapq.nlargest(N_SENSING, self.whittle_idx)
        action = [action_temp.index(i) for i in action_temp]

        print action


        return action





        