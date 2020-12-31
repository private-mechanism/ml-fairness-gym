import numpy as np
A=[1,2,3,4]
A[-1]=5
print(A)
# from gym.spaces import Discrete, Box, Dict,Tuple
# from gym import Wrapper
# import numpy as np
#
# import ray
# from environments.lending import selection_rate_based_lending_env, State
# from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
# from ray.rllib import agents
# from ray import tune
# from ray.tune import grid_search
#
# ray.init()
#
# class selection_DelayedImpactEnv(Wrapper):
#   def __init__(self, selection_rate_based_lending_env):
#     super(selection_DelayedImpactEnv, self).__init__(selection_rate_based_lending_env)
#     self.observation_space = Dict(self.observable_state_vars)
#
#   def _get_observable_state(self):
#     return {
#         'applicant_distribution':
#           np.array([self.state.params.applicant_distribution.components[0].weights,
#                   self.state.params.applicant_distribution.components[1].weights])
#     }
#
# def env_creator(env_config):
#     env=selection_rate_based_lending_env()
#     env = selection_DelayedImpactEnv(env)
#     return env
#
# tune.registry.register_env("selection_DelayedImpactEnv", env_creator)
#
# # def env_creator(env_name):
# # #     if env_name == 'MyEnv-v0':
# # #         from custom_gym.envs.custom_env import CustomEnv0 as env
# # #     elif env_name == 'MyEnv-v1':
# # #         from custom_gym.envs.custom_env import CustomEnv1 as env
# # #     else:
# # #         raise NotImplementedError
# # #     return env
#
# # env_name = 'MyEnv-v0'
# config = {
#     # Should use a critic as a baseline (otherwise don't use value baseline;
#     # required for using GAE).
#     "use_critic": True,
#     # If true, use the Generalized Advantage Estimator (GAE)
#     # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
#     "use_gae": True,
#     # Size of rollout batch
#     "rollout_fragment_length": 10,
#     # GAE(gamma) parameter
#     "lambda": 1.0,
#     # Max global norm for each gradient calculated by worker
#     "grad_clip": 40.0,
#     # Learning rate
#     "lr": 0.0001,
#     # Learning rate schedule
#     "lr_schedule": None,
#     # Value Function Loss coefficient
#     "vf_loss_coeff": 0.5,
#     # Entropy coefficient
#     "entropy_coeff": 0.01,
#     # Min time per iteration
#     "min_iter_time_s": 5,
#     # Workers sample async. Note that this increases the effective
#     # rollout_fragment_length by up to 5x due to async buffering of batches.
#     "sample_async": True,
# }
#
# trainer = agents.a3c.A3CTrainer(
#     env="selection_DelayedImpactEnv",
#     config=config)
# max_training_episodes = 1000
# while True:
#     results = trainer.train()
#     # Enter whatever stopping criterion you like
#     if results['episodes_total'] >= max_training_episodes:
#         break
# print('Mean Rewards:\t{:.1f}'.format(results['episode_reward_mean']))
#
# # class RunnerThread():
# #     def __init__(self, env, policy):
# #
# #         # starts simulation environment, policy, and thread.
# #         # Thread will continuously interact with the simulation environment
# #         # self.id = actor_id
# #         # self.policy = A3CTFPolicy()
# #         # self.runner = RunnerThread(env, self.policy, 20)
# #         # self.start()
# #
# #     def start_runner(self):
#
#
#
#
# #
# # @ray.remote
# # class Runner(object):
# #     """Actor object to start running simulation on workers.
# #         Gradient computation is also executed on this /object."""
# #     def __init__(self, env_name, actor_id):
# #         # starts simulation environment, policy, and thread.
# #         # Thread will continuously interact with the simulation environment
# #         self.env = env = env_creator(env_name)
# #         self.id = actor_id
# #         self.policy = A3CTFPolicy()
# #         self.runner = RunnerThread(env, self.policy, 20)
# #         self.start()
# #
# #     def start(self):
# #         # starts the simulation thread
# #         self.runner.start_runner()
# #
# #     def pull_batch_from_queue(self):
# #         # Implementation details removed - gets partial rollout from queue
# #         return rollout
# #
# #     def compute_gradient(self, params):
# #         self.policy.set_weights(params)
# #         rollout = self.pull_batch_from_queue()
# #         batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)
# #         gradient = self.policy.compute_gradients(batch)
# #         info = {"id": self.id,
# #                 "size": len(batch.a)}
# #         return gradient, info
# #
# # def train(num_workers, env_name="lendingEnv"):
# #     # Setup a copy of the environment
# #     # Instantiate a copy of the policy - mainly used as a placeholder
# #     env = env_creator("lendingEnv")
# #     policy = A3CTFPolicy(env.observation_space.shape, env.action_space.n, 0)
# #     obs = 0
# #
# #     # Start simulations on actors
# #     agents = [Runner(env_name, i) for i in range(num_workers)]
# #
# #     # Start gradient calculation tasks on each actor
# #     parameters = policy.get_weights()
# #     gradient_list = [agent.compute_gradient.remote(parameters) for agent in agents]
# #
# #     while True: # Replace with your termination condition
# #         # wait for some gradient to be computed - unblock as soon as the earliest arrives
# #         done_id, gradient_list = ray.wait(gradient_list)
# #
# #         # get the results of the task from the object store
# #         gradient, info = ray.get(done_id)[0]
# #         obs += info["size"]
# #
# #         # apply update, get the weights from the model, start a new task on the same actor object
# #         policy.apply_gradients(gradient)
# #         parameters = policy.get_weights()
# #         gradient_list.extend([agents[info["id"]].compute_gradient(parameters)])
# #         return policy
# #
# # if __name__ == "__main__":
# #     num_workers=20
# #     policy=train(num_workers, env_name="lendingEnv")