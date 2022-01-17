"""Example of running StarCraft2 with RLlib PPO.
In this setup, each agent will be controlled by an independent PPO policy.
However the policies share weights.
Increase the level of parallelism by changing --num-workers.
"""
import sys
from absl import flags
import ray
from ray.tune import run_experiments, register_env, Experiment
from ray.rllib.models import ModelCatalog
from DefeatRoaches import DREnv
from CollectMineralAndGas import CMGEnv

def main():
    ray.init()
    register_env("cmg", lambda config: CMGEnv())
    experiment_spec = Experiment(
            "dqn_sc2", #experiment name 
            "DQN", #run function 
            checkpoint_freq=100, #save model at 100th iteration
            num_samples=1, #Number of times to sample from the hyperparameter space.
            stop={  #stop condition 
                "training_iteration": 10000,
            },
            config={ #Algorithm-specific configuration for Tune variant generation
                "env": "cmg",
                "framework": "torch",
                "buffer_size": 50000, # Size of the replay buffer.
                "timesteps_per_iteration": 1000, # Minimum env steps to optimize for per train call. Not affect learning
                "n_step": 3, # N-step Q learning
                "prioritized_replay": True, # If True prioritized replay buffer will be used.
                "grad_clip": None,
                "num_workers": 1, #Number of environments to evaluate vectorwise per worker.
                
                "exploration_config": {
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 1000,# Timesteps over which to anneal epsilon.)
                }
            }
            )

    run_experiments(experiment_spec)

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS([''])
    main()

