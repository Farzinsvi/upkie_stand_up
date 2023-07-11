import os

import xpag
from xpag.wrappers import gym_vec_env
from xpag.buffers import DefaultBuffer
from xpag.samplers import DefaultSampler
from xpag.setters import DefaultSetter
from xpag.agents import SAC, TQC
from xpag.tools import learn

import upkie
from upkie.envs import register

from upkie_stand_up.tools.utils import print_arguments_as_table

agent_dict = {
    'SAC': SAC,
    'TQC': TQC
}

register()

def train_model(
                agent_name='SAC',
                env_name='UpkieWheelsEnv-v3',
                exp_name='unnamed_exp',
                # Agent arguments
                actor_lr=3e-3,
                critic_lr=3e-3,
                temp_lr=3e-4,
                tau=5e-2,
                # Buffer arguments
                buffer_size=200_000,
                # Learning arguments
                batch_size=256,
                gradient_steps=1,
                start_training_after=0,
                max_steps=1_000_000_000,
                evaluate_every=5_000,
                save_every=50_000,
                seed=0
                ):
    
    # Print arguments
    print_arguments_as_table(locals())

    # Create the environment
    num_envs = 1
    env, eval_env, env_info = gym_vec_env(env_name, num_envs)

    # Create the agent
    agent = agent_dict[agent_name](
        env_info['observation_dim'],
        env_info['action_dim'],
        {
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "temp_lr": temp_lr,
            "tau": tau,
            "seed": seed
        }
    )

    # Create the replay buffer
    sampler = DefaultSampler()
    buffer = DefaultBuffer(
        buffer_size=buffer_size,
        sampler=sampler
    )
    setter = DefaultSetter()

    save_dir = os.path.join(os.getcwd(), 'results', agent_name, exp_name)
    save_episode = True
    plot_projection = None

    # Start training
    learn(
        env,
        eval_env,
        env_info,
        agent,
        buffer,
        setter,
        batch_size=batch_size,
        gd_steps_per_step=gradient_steps,
        start_training_after_x_steps=start_training_after,
        max_steps=max_steps,
        evaluate_every_x_steps=evaluate_every,
        save_agent_every_x_steps=save_every,
        save_dir=save_dir,
        save_episode=save_episode,
        plot_projection=plot_projection,
        custom_eval_function=None,
        additional_step_keys=None,
        seed=seed
    )

if __name__ == "__main__":
    train_model()

