import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")

state,info = env.reset()

for step in range(10000000):
    action = env.action_space.sample()

    next_state,reward,terminate,truncated,info = env.step(action)
    print("State:", state)
    print("Action:", action)
    print("Reward:", reward)
    print("Next State:", next_state)
    print("----------------")

    time.sleep(0.01)
    state = next_state
    if terminate or truncated:
        break

env.close()