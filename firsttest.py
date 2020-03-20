import gym
import numpy as np
import matplotlib.pyplot as plt

def value_iteration(env, max_iterations=100000, lmbda=0.9):
    stateValue = [0 for i in range(env.nS)]
    newStateValue = stateValue.copy()
    for i in range(max_iterations):
        for state in range(env.nS):
            action_values = []
            for action in range(env.nA):
                state_value = 0
                for i in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][i]
                    state_action_value = prob * (reward + lmbda * stateValue[next_state])
                    state_value += state_action_value
                action_values.append(state_value)  # the value of each action
                best_action = np.argmax(np.asarray(action_values))  # choose the action which gives the maximum value
                newStateValue[state] = action_values[best_action]  # update the value of the state
        if i > 1000:
            if sum(stateValue) - sum(newStateValue) < 1e-04:  # if there is negligible difference break the loop
                break
                print(i)
        else:
            stateValue = newStateValue.copy()
    return stateValue


def get_policy(env, stateValue, lmbda=0.9):
    policy = [0 for i in range(env.nS)]
    for state in range(env.nS):
        action_values = []
        for action in range(env.nA):
            action_value = 0
            for i in range(len(env.P[state][action])):
                prob, next_state, r, _ = env.P[state][action][i]
                action_value += prob * (r + lmbda * stateValue[next_state])
            action_values.append(action_value)
        best_action = np.argmax(np.asarray(action_values))
        policy[state] = best_action
    return policy


def get_score(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:

            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                # print('You have got the fucking Frisbee after {} steps'.format(steps))
                steps_list.append(steps)
                break
            elif done and reward == 0:
                # print("You fell in a hole!")
                misses += 1
                break
    average_number_of_steps=np.mean(steps_list)
    failure_chance=(misses / episodes) * 100
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(average_number_of_steps))
    print('And you fell in the hole {:.2f} % of the times'.format(failure_chance))
    print('----------------------------------------------')
    return average_number_of_steps,failure_chance


env = gym.make('FrozenLake8x8-v0')

average_number_of_steps_dict={}
failure_chance_dict={}
training_iterations = [i*100 for i in range(1,50)]
for training_iteration in training_iterations:
    env.reset()
    print(format(f" traing interations number={training_iteration}"))
    stateValues = value_iteration(env, max_iterations=training_iteration)
    policy = get_policy(env, stateValues)
    average_number_of_steps,failure_chance=get_score(env, policy, episodes=1000)
    average_number_of_steps_dict[training_iteration]=average_number_of_steps
    failure_chance_dict[training_iteration] = failure_chance

plt.figure(1)
plt.subplot(211)
plt.plot(*zip(*sorted(average_number_of_steps_dict.items())))
plt.subplot(212)
plt.plot(*zip(*sorted(failure_chance_dict.items())))

plt.show()
