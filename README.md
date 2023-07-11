# Upkie Stand Up

Learning an stand up motion with Upkie wheeled biped robot.

## Workflow

Below is a description of my workflow.

Right now we are using a locally modified version of the [upkie](https://github.com/tasts-robots/upkie) repo, which you can find [here](https://github.com/mariogpascual). My set up is:
```
├── envupkie
├── others
├── upkie
└── upkie_stand_up
```
1. Go to __upkie__ folder and start a simulation using `./tools/bazelisk run -c opt //spines:bullet -- --show`. The `--show` parameter is optional and enables visualization
2. In a different terminal and once the simulation has started, you can run the training script `python tools/train_agent.py --agent SAC --exp_name exp_name` to train a SAC agent in simulation, for example. The results are saved in `results/SAC/exp_name` automatically by __xpag__
3. To visualize the plot of the evaluations done during training you can use `python tools/plot.py --agent SAC --exp_name exp_name`
4. To try the agent trained in the experiment you can use `python tools/test_agent.py --agent SAC --exp_name exp_name`, making sure there is a simulation running to conect to

### Train an agent

```
python tools/train_agent.py [-h] --agent {SAC,TQC} [--env ENV] --exp_name EXP_NAME
                      [--max_steps MAX_STEPS] [--evaluate_every EVALUATE_EVERY]
                      [--save_every SAVE_EVERY]
```

### Plot experiment results

```
python tools/plot.py [-h] --agent {SAC,TQC} --exp_name EXP_NAME
```

### Test a trained agent

```
python tools/test_agent.py [-h] --agent {SAC,TQC} [--env ENV] --exp_name EXP_NAME [--steps STEPS]
```
