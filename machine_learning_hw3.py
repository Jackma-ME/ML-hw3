import numpy as np
import random
import matplotlib.pyplot as plt

alpha = 0.5
epsilon = 0.1
gamma = 0.95
x_length = 12
y_length = 4

def observe(x,y,a):
'''
	ran = random.random()
	if ran < 0.2:
		ran2 = random.randint(1,3)
		a = (a + ran2) % 4
'''
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

def valid_action(x,y,action,r):
	inv = np.zeros([4])
	a = 0
	n = 3
	inv[action] = 1
	if x == 0:
		n -= 1
		inv[3] = 1
	if y == 0:
		n -= 1
		inv[2] = 1
	if x == x_length - 1:
		n -= 1
		inv[1] = 1
	if y == y_length - 1:
		n -= 1
		inv[0] = 1
	if n == 1:
		for i in range(4):
			if inv[i] == 0:
				a = i
				return a
	elif n == 2:
		if r < 0.2/2:
			for i in range(4):
				if inv[i] == 0:
					a = i
					return a
		elif r > 0.2/2:
			count = 0
			for i in range(4):
				if inv[i] == 0:
					if count == 1:
						a = i
						return a
					count += 1
	elif n == 3:
		if r < 0.2/3:
			for i in range(4):
				if inv[i] == 0:
					a = i
					return a
		elif r > 0.2/3 and r < 2*0.2/3:
			count = 0
			for i in range(4):
				if inv[i] == 0:
					if count == 1:
						a = i
						return a
					count += 1
		elif r > 2*0.2/3:
			count = 0
			for i in range(4):
				if inv[i] == 0:
					if count == 2:
						a = i
						return a
					count += 1

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
'''
	ran = random.random()
	if ran < 0.2:
		a = valid_action(x,y,a,ran)
		return a
	else:
		return a
''' 

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
	avg_rewards = []
	for i in range(9):
		avg_rewards.append(np.mean(rewards[:i+1]))
	for i in range(10, len(rewards)+1):
		avg_rewards.append(np.mean(rewards[i-10:i]))
	return avg_rewards

def OptimalPath(q):
	x = 0
	y = 0
	path = np.zeros([x_length, y_length]) - 1
	end = 0
	exist = np.zeros([x_length, y_length])
	while (x != x_length -1 or y != 0) and end == 0:
		a = max_q(x,y,q)
		path[x][y] = a
		if exist[x][y] == 1:
			end = 1
		exist[x][y] = 1
		x,y,r = observe(x,y,a)
	for j in range(y_length-1, -1, -1):
		for i in range(x_length):
			if i == x_length-1 and j == 0:
				print("G ", end = "")
				continue
			a = path[i,j]
			if a == -1:
				print("0 ", end = "")
			elif a == 0:
				print("↑ ", end = "")
			elif a == 1:
				print("→ ", end = "")
			elif a == 2:
				print("↓ ", end = "")
			elif a == 3:
				print("← ", end = "")
		print("")

q = np.zeros([12,4,4])
sarsa_rewards = sarsa_on_policy(q)
qq = np.zeros([12,4,4])
q_learning_rewards = q_learning(qq)

plt.plot(range(len(sarsa_rewards)), sarsa_rewards, label="sarsa")
plt.plot(range(len(q_learning_rewards)), q_learning_rewards, label="q-learning")
plt.legend(loc="lower right")

print("Sarsa method")
OptimalPath(q)
print("")
print("Q-learning")
OptimalPath(qq)

plt.show()



