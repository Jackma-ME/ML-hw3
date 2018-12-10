import numpy as np
import random
import matplotlib.pyplot as plt

alpha = 0.5
epsilon = 0.1
gamma = 0.95
x_length = 12
y_length = 4

def observe(x,y,a):
	goal = 0
	if x == x_length - 1 and y == 0:
		goal = 1
	if a == 0:
		y += 1
	elif a == 1:
		x += 1
	elif a == 2:
		y -= 1
	elif a == 3:
		x -= 1
	x = max(0,x)
	x = min(x_length - 1, x)
	y = max(0,y)
	y = min(y_length - 1, y)
	
	if goal == 1:
		return x,y,1
	elif x > 0 and x < x_length-1 and y == 0:
		return 0,0,-100
	return x,y,-1

def epsilon_policy(x,y,q,eps):
	t = random.randint(0,3)
	if random.random() < eps:
		a = t
	else:
		q_max = q[x][y][0]
		a_max = 0
		for i in range(4):
			if q[x][y][i] >= q_max:
				q_max = q[x][y][i]
				a_max = i
		a = a_max
	return a

def max_q(x,y,q):
	q_max = q[x][y][0]
	a_max = 0
	for i in range(4):
		if q[x][y][i] >= q_max:
			q_max = q[x][y][i]
			a_max = i 
	a = a_max
	return a

## sarsa on-policy
def sarsa_on_policy(q):
	runs = 20
	rewards = np.zeros([500])
	for j in range(runs):
		for i in range(500):
			reward_sum = 0
			x = 0
			y = 0
			a = epsilon_policy(x,y,q,epsilon)
			while True:
				x_next, y_next, reward = observe(x,y,a)
				reward_sum += reward
				a_next = epsilon_policy(x_next,y_next,q,epsilon)
				q[x][y][a] += alpha*(reward + gamma*q[x_next][y_next][a_next] - q[x][y][a])
				if x == x_length - 1 and y == 0:
					break
				x = x_next
				y = y_next
				a = a_next
			rewards[i] += reward_sum
	rewards /= runs
## need to think
	avg_rewards = []
	for i in range(9):
		avg_rewards.append(np.mean(rewards[:i+1]))
	for i in range(10, len(rewards)+1):
		avg_rewards.append(np.mean(rewards[i-10:i]))
	return avg_rewards

## Q-learning
def q_learning(q):
	runs = 20
	rewards = np.zeros([500])
	for j in range(runs):
		for i in range(500):
			reward_sum = 0
			x = 0
			y = 0
			while True:
				a = epsilon_policy(x,y,q,epsilon)
				x_next, y_next, reward = observe(x,y,a)
				a_next = max_q(x_next,y_next,q)
				reward_sum += reward
				q[x][y][a] += alpha*(reward + gamma*q[x_next][y_next][a_next] - q[x][y][a])
				if x == x_length - 1 and y == 0:
					break
				x = x_next
				y = y_next
			rewards[i] += reward_sum
	rewards /= runs
## need to think
	avg_rewards = []
	for i in range(9):
		avg_rewards.append(np.mean(rewards[:i+1]))
	for i in range(10, len(rewards)+1):
		avg_rewards.append(np.mean(rewards[i-10:i]))
	return avg_rewards

q = np.zeros([12,4,4])
sarsa_rewards = sarsa_on_policy(q)
qq = np.zeros([12,4,4])
q_learning_rewards = q_learning(qq)

plt.plot(range(len(sarsa_rewards)), sarsa_rewards, label="sarsa")
plt.plot(range(len(q_learning_rewards)), q_learning_rewards, label="q-learning")
plt.ylim(-100,0)
plt.legend(loc="lower right")
plt.show()




