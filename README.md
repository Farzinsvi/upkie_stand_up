# upkie-stand-up
Learning an stand up motion with Upkie wheeled biped robot.

## Workflow

Hi all!

Below is a description of my workflow.

Right now we are using a locally modify version of the [upkie](https://github.com/tasts-robots/upkie) repo, which you can find [here](https://github.com/mariogpascual). My set up is:
```
├── envupkie
├── others
├── upkie
└── upkie-stand-up
```
1. Go to `upkie` folder and start a simulation using `./tools/bazelisk run -c opt //spines:bullet -- --show`. The `--show` parameter is optional and enables visualization
2. In a different terminal and once the simulation has started, you can run the training script `python upkie-stand-up/src/agents/xpag/train_sac.py --exp_name exp_name` to train a SAC agent in simulation, for example. The results are saved in `src/agents/xpag/results/exp_name` automatically by `xpag`
3. To visualize the plot of the evaluations done during training you can use `python upkie-stand-up/tools/plot.py upkie-stand-up/src/agent/xpag/results/exp_name`
4. To try the agent trained in the experiment you can use `python upkie-stand-up/tools/try_agent.py --exp_path upkie-stand-up/src/agent/xpag/results/exp_name --agent SAC`, making sure there is a simulation running to conect to

### To train an agent

```
train_sac.py [-h] [--max_steps MAX_STEPS]
                    [--evaluate_every_x_steps EVALUATE_EVERY_X_STEPS]
                    [--save_agent_every_x_steps SAVE_AGENT_EVERY_X_STEPS]
                    --exp_name EXP_NAME
```

### To plot experiment results

```
plot.py [-h] exp_path
```

### To try a trained agent

```
try_agent.py [-h] --exp_path EXP_PATH --agent {SAC,TQC} [--steps STEPS]
```
