import argparse

from upkie_stand_up.src.agents.xpag.test import test_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Upkie wheel balancer with SAC')

    parser.add_argument('--agent', type=str, choices=['SAC', 'TQC'], required=True,
                        help='SAC or TQC')
    
    parser.add_argument('--env', type=str, default='UpkieWheelsEnv-v3',
                        help='Training environment')
    
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Name of the experiment')

    parser.add_argument('--steps', type=int, default=1_000,
                        help='Number of steps of testing (default: 1_000_000)')

    return parser.parse_args()


args = parse_arguments()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Train model
    test_model(
        agent_name=args.agent,
        env_name=args.env,
        exp_name=args.exp_name,
        steps=args.steps
    )

