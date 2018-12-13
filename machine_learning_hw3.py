import numpy as np
import random
import matplotlib.pyplot as plt

alpha = 0.05  # learning rate
epsilon = 0.1 # exploring rate
gamma = 0.95  # discount factor
x_length = 12 # environment of x length
y_length = 4  # environment of y length

# observe next motion
def observe(x,y,a,mod = 1):
	if mod == 1:
		ran = random.random()
		# possibility of taking another action
		if ran < 0.2:
			a = valid_action(x,y,a,ran)
	goal = 0
	# arrive the goal
	if x == x_length - 1 and y == 0:
		goal = 1
	# turn up 
	if a == 0:
		y += 1
	# turn right
	elif a == 1:
		x += 1
	# turn down
	elif a == 2:
		y -= 1
	# turn left
	elif a == 3:
		x -= 1
	
	x = max(0,x)
	x = min(x_length - 1, x)
	y = max(0,y)
	y = min(y_length - 1, y)
	
	# reward 500 when arrives the goal
	if goal == 1:
		return x,y,500
	# reward -100 when go into cliff
	elif x > 0 and x < x_length-1 and y == 0:
		return 0,0,-100
	# reward -1 when go one step
	return x,y,-1

# choose the valid action
def valid_action(x,y,action,r):
	inv = np.zeros([4]) # invalid action list
	n = 3 # three chooses
	a = 0
	inv[action] = 1 
	
	# ensure the invalid action
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
	
	# choose the another valid action
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

# epsilon policy
def epsilon_policy(x,y,q,eps):
	t = random.randint(0,3) # random walk
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

# maximun reward
def max_q(x,y,q):
	q_max = q[x][y][0]
	a_max = 0
	for i in range(4):
		if q[x][y][i] >= q_max:
			q_max = q[x][y][i] # maximum reward
			a_max = i # action of maximum reward
	a = a_max
	return a

## sarsa on-policy
def sarsa_on_policy(q):
	runs = 200 # times of learning
	rewards = np.zeros([500]) # 500 episodes
	for j in range(runs):
		for i in range(500):
			# initialize parameter
			reward_sum = 0
			x = 0
			y = 0
			a = epsilon_policy(x,y,q,epsilon) # take action
			while True:
				x_next, y_next, reward = observe(x,y,a) # observe reward and next state
				reward_sum += reward
				a_next = epsilon_policy(x_next,y_next,q,epsilon) # take next action
				# sarsa update rule
				q[x][y][a] += alpha*(reward + gamma*q[x_next][y_next][a_next] - q[x][y][a])
				# stop when arrive the goal
				if x == x_length - 1 and y == 0:
					break
				# update parameter
				x = x_next
				y = y_next
				a = a_next
			# store the sum of rewards
			rewards[i] += reward_sum
	# average the reward
	rewards /= runs
	avg_rewards = []
	for i in range(9):
		avg_rewards.append(np.mean(rewards[:i+1]))
	for i in range(10, len(rewards)+1):
		avg_rewards.append(np.mean(rewards[i-10:i]))
	return avg_rewards

## Q-learning
def q_learning(q):
	runs = 200 # times of learning
	rewards = np.zeros([500]) # 500 episodes
	for j in range(runs):
		for i in range(500):
			# initialize parameter
			reward_sum = 0
			x = 0
			y = 0
			while True:
				a = epsilon_policy(x,y,q,epsilon) # take action
				x_next, y_next, reward = observe(x,y,a) # observe reward and next state
				a_next = max_q(x_next,y_next,q) # take next action
				reward_sum += reward
				# q-learning update rule
				q[x][y][a] += alpha*(reward + gamma*q[x_next][y_next][a_next] - q[x][y][a])
				# stop when arrive the goal
				if x == x_length - 1 and y == 0:
					break
				# update parameter
				x = x_next
				y = y_next
			# # store the sum of rewards
			rewards[i] += reward_sum
	# average the reward
	rewards /= runs
	avg_rewards = []
	for i in range(9):
		avg_rewards.append(np.mean(rewards[:i+1]))
	for i in range(10, len(rewards)+1):
		avg_rewards.append(np.mean(rewards[i-10:i]))
	return avg_rewards

# print the path
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
		x,y,r = observe(x,y,a,2)
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

# testing 
def testing(q):
	rewards = np.zeros([500]) # 500 episodes
	for i in range(500):
		# initialize parameter
		reward_sum = 0
		x = 0
		y = 0
		while True:
			a = max_q(x,y,q) # action of maximum reward
			x_next,y_next,r = observe(x,y,a)
			reward_sum += r
			if x == x_length - 1 and y == 0:
				break
			# update parameter
			x = x_next
			y = y_next
		rewards[i] += reward_sum
	return rewards

q = np.zeros([12,4,4])
sarsa_rewards = sarsa_on_policy(q)
qq = np.zeros([12,4,4])
q_learning_rewards = q_learning(qq)

print("Sarsa method")
OptimalPath(q)
print("")
print("Q-learning")
OptimalPath(qq)

plt.figure(1)
plt.plot(range(len(sarsa_rewards)), sarsa_rewards, label="sarsa")
plt.plot(range(len(q_learning_rewards)), q_learning_rewards, label="q-learning")
plt.legend(loc="lower right")

plt.figure(2)
re1 = testing(q)
plt.plot(range(len(re1)), re1, label="sarsa")
re2 = testing(qq)
plt.plot(range(len(re2)), re2, label="q-learning")
plt.legend(loc="lower right")
plt.show()
