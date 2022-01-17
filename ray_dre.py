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

def main():
    ray.init()
    register_env("dre", lambda config: DREnv())
    experiment_spec = Experiment(
            "dqn_sc2", #experiment name 
            "DQN", #run function 
            checkpoint_freq=100, #save model at 100th iteration
            num_samples=1, #run how many game
            stop={  #stop condition 
                "training_iteration": 10000,
            },
            config={ 
                "env": "dre",
                "framework": "torch",
                "buffer_size": 50000,
                "timesteps_per_iteration": 1000, 
                "n_step": 3, 
                "prioritized_replay": True,
                "grad_clip": None,
                "num_workers": 1,
                
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

