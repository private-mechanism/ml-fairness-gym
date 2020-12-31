"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
import gym
from gym.spaces import Discrete, Box, Dict,Tuple
from gym import Wrapper
import numpy as np
import os
from environments.lending import selection_rate_based_lending_env, thr_rate_based_lending_env,State,dp_selection_rate_based_lending_env
from environments import lending_params
from experiments import lending
import spaces
import attr

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved




tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="A3C")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=500)
parser.add_argument("--stop-timesteps", type=int, default=800)
parser.add_argument("--stop-reward", type=float, default=10000000000)


class selection_DelayedImpactEnv(Wrapper):
  def __init__(self,  dp_selection_rate_based_lending_env):
    super(selection_DelayedImpactEnv, self).__init__(dp_selection_rate_based_lending_env)
    self.observation_space = Dict(self.observable_state_vars)

  def _get_observable_state(self):
    return {
        'applicant_distribution':
          np.array([self.state.params.applicant_distribution.components[0].weights,
                  self.state.params.applicant_distribution.components[1].weights])
    }

def env_creator(env_config):
    env= dp_selection_rate_based_lending_env()
    env = selection_DelayedImpactEnv(env)
    return env

class CustomModel(TFModelV2,selection_DelayedImpactEnv):
    """Example of a keras custom model that just delegates to an fc-net."""
    obs_space=selection_DelayedImpactEnv.observation_space
    action_space=selection_DelayedImpactEnv.action_space
    # num_outputs=169
    model_config={}
    name='My_model'
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class A3CModel(TFModelV2,dp_selection_rate_based_lending_env):
    obs_space = selection_DelayedImpactEnv.observation_space
    action_space = selection_DelayedImpactEnv.action_space
    num_outputs = 441
    model_config = {}
    name = 'My_A3C_model'
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(A3CModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        input_layer = tf.keras.layers.Input(shape=(int(np.product(obs_space.shape)), ), name="observations")
        hidden_layer = tf.keras.layers.Dense(1000, name="hidden")
        output_layer = tf.keras.layers.Dense(num_outputs,name="fc_out")
        value_layer = tf.keras.layers.Dense(1, name="value")
        self.model = tf.keras.Model(input_layer, hidden_layer,[output_layer, value_layer])
        self.register_variables(self.model.variables)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    tune.registry.register_env("selection_DelayedImpactEnv", env_creator)
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.torch else CustomModel)
    config = {
        "env": "selection_DelayedImpactEnv",  # or "corridor" if registered above
        "env_config": {},
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
        },
        "lr": grid_search([1e-2]),  # try different lrs
        "num_workers": 30,  # parallelism
        "framework": "torch" if args.torch else "tf",
    }
    stop = {
        "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    results = tune.run(args.run, config=config, stop=stop)
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()