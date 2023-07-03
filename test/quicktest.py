import gymnasium as gym
import upkie.envs

upkie.envs.register()

with gym.make("UpkieWheelsEnv-v3", frequency=200.0) as env:
    observation = env.reset()
    action = 0.0 * env.action_space.sample()
    for step in range(1_000_000):
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation = env.reset()
        pitch = observation[0]
        action[0] = 10.0 * pitch
