import argparse

from upkie_stand_up.src.agents.xpag.train import train_model

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Upkie wheel balancer with SAC')

    parser.add_argument('--agent', type=str, choices=['SAC', 'TQC'], required=True,
                        help='SAC or TQC')
    
    parser.add_argument('--env', type=str, default='UpkieWheelsEnv-v3',
                        help='Training environment')
    
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of the experiment')

    parser.add_argument('--max_steps', type=int, default=4_000_000,
                        help='Maximum number of steps for training (default: 1_000_000_000)')
    
    parser.add_argument('--evaluate_every', type=int, default=16_000,
                        help='Evaluate the agent every x steps (default: 5_000)')
    
    parser.add_argument('--save_every', type=int, default=50_000,
                        help='Save the agent every x steps (default: 50_000)')

    return parser.parse_args()


args = parse_arguments()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Train model
    train_model(
        agent_name=args.agent,
        env_name=args.env,
        exp_name=args.exp_name,
        max_steps=args.max_steps,
        evaluate_every=args.evaluate_every,
        save_every=args.save_every
    )

