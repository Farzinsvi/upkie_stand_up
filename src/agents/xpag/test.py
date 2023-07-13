import os

from upkie_stand_up.src.envs import register

import xpag
from xpag.agents import SAC, TQC

import gymnasium as gym

from upkie_stand_up.tools.utils import print_arguments_as_table

agent_dict = {
    'SAC': SAC,
    'TQC': TQC
}

register()

def test_model(
            agent_name='SAC',
            env_name='UpkieWheelsEnv-v3',
            exp_name='unnamed_exp',
            steps=1_000_000
            ):

    
    # Print arguments
    print_arguments_as_table(locals())

    # Create environment
    # TODO: test frequency=None
    env = gym.make(env_name, frequency=200.0)

    # Create agent
    agent = agent_dict[agent_name](
        env.observation_space.shape[0],
        env.action_space.shape[0],
        {
            'seed': 0
        }
    )

    # Load agent
    agent.load(os.path.join(os.getcwd(), 'results', agent_name, exp_name, 'agent'))

    # Test agent
    # TODO: solve the way observations are handled
    observation = (env.reset()[0])
    action = agent.select_action(observation, eval_mode=True)
    for step in range(steps):
        observation, reward, done, _, _ = env.step(action)
        if done:
            observation = (env.reset()[0])
        action = agent.select_action(observation, eval_mode=True)
    
if __name__ == "__main__":
    test_model()
