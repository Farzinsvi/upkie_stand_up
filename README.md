# upkie-stand-up
Learning an stand up motion with Upkie wheeled biped robot

## Train

```
train_sac.py [-h] [--max_steps MAX_STEPS]
                    [--evaluate_every_x_steps EVALUATE_EVERY_X_STEPS]
                    [--save_agent_every_x_steps SAVE_AGENT_EVERY_X_STEPS]
                    --exp_name EXP_NAME
```

## Plot experiment results

```
plot.py [-h] exp_path
```

## Try trained agent

```
try_agent.py [-h] --exp_path EXP_PATH --agent {SAC,TQC} [--steps STEPS]
```
