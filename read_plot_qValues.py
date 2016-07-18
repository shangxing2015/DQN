import matplotlib.pyplot as plt

index_list=[]
reward_list=[]
action_list=[]
time_list=[]


with open('log_async_qlearning_identical_1_3.txt') as input_file:
    for line in input_file:
        index, index_num, reward, reward_num, action, action_num, time, time_num = (item.strip() for item in line.split())
        index_list.append(int(index_num))
        reward_list.append(float(reward_num))
        action_list.append(int(action_num))
        time_list.append(float(time_num)/60)


ideal_list = [0.767448] * len(time_list)
plt.plot(time_list, reward_list, time_list, ideal_list, 'r-')

plt.ylabel('average reward')
plt.xlabel('time in min')
plt.title('2 identical markov channel')
plt.legend(['async_q', 'whittle_idx'])

plt.show()

