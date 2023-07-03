import xpag
import upkie.envs

upkie.envs.register()
env, eval_env, env_info = xpag.wrappers.gym_vec_env('UpkieWheelsEnv-v3', 1)
env.reset()
__import__("IPython").embed()