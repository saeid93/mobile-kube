import numpy as np
from typing import Dict
from tabulate import tabulate
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy


class CloudCallback(DefaultCallbacks):

    """
    callbakc for saving my own metrics
    functions to add necessary metrics either to the
    tensorboard or print it in the output during the
    training
    points (functions) to add metrics (see parent class
    description for each):
        1. on_episode_start
        2. on_episode_step
        3. on_episode_end
        4. on_sample_end
        5. on_postprocess_trajectory
        6. on_sample_end
        7. on_learn_on_batch
        8. on_train_result
    variables to add/store custom metrics:
        1. episode.user_data: to pass data between episodes
        2. episode.custom_metrics: what is being printed to the tensorboard
        3. episode.hist_data: histogram data saved to the json
    my metrics:
        1. num_consolidated
        2. num_overloaded
        3. greedy_num_consolidated
        4. num_moves
    """
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        self.workers_total_episodes = [0]*1000
        self.count = 0
        self.total = 0

    def on_episode_start(self, *, worker: RolloutWorker,
                         base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode,
                         env_index: int, **kwargs):

        # timestep of the dataset
        episode.user_data["timestep"] = []
        episode.hist_data["timestep"] = []
        # num_consolidated placeholder lists
        episode.user_data["num_consolidated"] = []
        episode.hist_data["num_consolidated"] = []
        # num_overloaded placeholder lists
        episode.user_data["num_overloaded"] = []
        episode.hist_data["num_overloaded"] = []
        # num_moves
        episode.user_data["num_moves"] = []
        episode.hist_data["num_moves"] = []
        # rewards
        episode.user_data["rewards"] = []
        episode.hist_data["rewards"] = []
        # greedy_consolidation
        episode.user_data["greedy_num_consolidated"] = []
        episode.hist_data["greedy_num_consolidated"] = []

        # # TODO TEMP
        # episode.user_data["step_reward"] = []
        # episode.hist_data["step_reward"] = []

    def on_episode_step(self, *, worker: RolloutWorker,
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: int, **kwargs):

        if type(episode.last_info_for()) == dict:
            # extract the timestep of the current step from the dict
            timestep = episode.last_info_for()['timestep']
            episode.user_data["timestep"].append(timestep)
            # extract the number of conolidated from the info dict
            num_consolidated = episode.last_info_for()['num_consolidated']
            episode.user_data["num_consolidated"].append(num_consolidated)
            # extract the number of overloaded from the info dict
            num_overloaded = episode.last_info_for()['num_overloaded']
            episode.user_data["num_overloaded"].append(num_overloaded)
            # extract of the greedy_num_consolidated from the dict
            num_moves = episode.last_info_for()['num_moves']
            episode.user_data["num_moves"].append(num_moves)
            # extract of the greedy_num_consolidated from the dict
            greedy_num_consolidated = episode.last_info_for()[
                'greedy_num_consolidated']
            episode.user_data["greedy_num_consolidated"].append(
                greedy_num_consolidated)

            # rewards
            rewards = episode.last_info_for()['rewards']
            episode.user_data["rewards"].append(rewards)

            # # TODO TEMP
            # step_reward = episode.prev_reward_for()
            # episode.user_data["step_reward"].append(step_reward)

            # print(f"env: <{episode.env_id}>")
            # print(f"worker: <{worker.worker_index}>")
            # print(f"episode: <{episode.episode_id}>")
            # print(f"timestep: <{timestep}>")
            # print(f"last state unprocessed:\n<{episode.last_raw_obs_for()}>")
            # print(f"last state process:\n<{episode.last_observation_for()}>")
            # print(f"rllib action:\n<{episode.last_action_for()}>")
            # print(f"env action: <{episode.last_info_for()['action']}>")

    def on_episode_end(self, *, worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        # extract the episode information
        num_moves_avg = np.mean(episode.user_data["num_moves"])
        num_consolidated_avg = np.mean(episode.user_data["num_consolidated"])
        num_overloaded_avg = np.mean(episode.user_data["num_overloaded"])
        last_timestep = np.max(episode.user_data["timestep"])
        if episode.user_data["greedy_num_consolidated"][0] is not None:
            greedy_num_consolidated_avg = np.mean(
                episode.user_data["greedy_num_consolidated"])
        else:
            greedy_num_consolidated_avg = None
        episode_total_reward = episode.total_reward
        action_logit_max = round(max(episode.last_action_for()).item(), 2)
        action_logit_min = round(min(episode.last_action_for()).item(), 2)
        action_logit_avg = round(np.mean(episode.last_action_for()).item(), 2)

        # extract episodes rewards info
        reward_consolidation_max = max([a['reward_consolidation']
                                        for a in episode.user_data[
                                            "rewards"]])
        reward_illegal_max = max([a['reward_illegal']
                                  for a in episode.user_data[
                                      "rewards"]])
        reward_move_max = max([a['reward_move']
                               for a in episode.user_data[
                                   "rewards"]])
        reward_variance_max = max([a['reward_variance']
                                   for a in episode.user_data[
                                       "rewards"]])

        # print episode information in the ouput
        # print('-'*50)
        print(f"<----- workder <{worker.worker_index}>,"
              f" episode <{episode.episode_id}>,"
              f" env <{env_index}> ----->")
        # headers = ['last_timestep', 'episode_length', 'num_consolidated_avg',
        #            'greedy_num_consolidated_avg', 'num_overloaded_avg',
        #            'action_logits_min', 'action_logits_max',
        #            'action_logits_avg']
        # table = [[last_timestep, episode.length, num_consolidated_avg,
        #           greedy_num_consolidated_avg, num_overloaded_avg,
        #           action_logit_max, action_logit_min, action_logit_avg]]
        # print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
        print("[-- episode info --]")
        print(f"last_timestep <{last_timestep}>"
              f", episode_length <{episode.length}>\n"
              f"num_consolidated_avg <{num_consolidated_avg}>"
              f", num_overloaded_avg <{num_overloaded_avg}>"
              f", greedy_num_consolidated_avg <{greedy_num_consolidated_avg}>\n"
              f"episode_total_reward <{episode_total_reward}>")
        print("[-- reward info --]")
        print(f"reward_consolidation_max <{reward_consolidation_max}>"
              f", reward_illegal_max <{reward_illegal_max}>"
              f", reward_move_max <{reward_move_max}>"
              f", reward_variance_max <{reward_variance_max}>")
        print("[-- last episode state/action info --]")
        print(f"action_logits_min <{action_logit_max}>"
              f", action_logits_max <{action_logit_min}>"
              f", action_logits_avg <{action_logit_avg}>\n")

        # add custom metrics to tensorboard
        episode.custom_metrics['num_moves'] = num_moves_avg
        episode.custom_metrics['num_consolidated'] = num_consolidated_avg
        episode.custom_metrics['num_overloaded'] = num_overloaded_avg
        episode.custom_metrics['action_logit_max'] = action_logit_max
        episode.custom_metrics['reward_consolidation_max'] =\
            reward_consolidation_max
        episode.custom_metrics['reward_illegal_max'] = reward_illegal_max
        episode.custom_metrics['reward_move_max'] = reward_move_max
        episode.custom_metrics['reward_variance_max'] = reward_variance_max
        if episode.user_data["greedy_num_consolidated"][0] is not None:
            episode.custom_metrics['greedy_num_consolidated_avg'] =\
                greedy_num_consolidated_avg

        # # add histogram data
        # episode.hist_data["timestep"] = \
        #     episode.user_data["timestep"]
        # episode.hist_data["num_consolidated"] = \
        #     episode.user_data["num_consolidated"]
        # episode.hist_data["greedy_num_consolidated"] = \
        #     episode.user_data["greedy_num_consolidated"]
        # episode.hist_data["num_overloaded"] = \
        #     episode.user_data["num_overloaded"]
        # episode.hist_data["num_moves"] = \
        #     episode.user_data["num_moves"]

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        num_episodes = np.unique(samples['eps_id']).size
        self.workers_total_episodes[worker.worker_index] += num_episodes
        # headers = ['rollout_fragment_length', 'num_episodes',
        #  'total_episodes']
        # table = [[samples.count, num_episodes,
        #           self.workers_total_episodes[worker.worker_index]]]
        print('*'*50)
        print("<--- one sample batch of worker"
              f" <{worker.worker_index}> ended --->")
        print(f"rollout_fragment_length <{samples.count}>"
              f", num_episodes <{num_episodes}>"
              ", total_episodes "
              f"<{self.workers_total_episodes[worker.worker_index]}>")
        # print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
        print('*'*50, '\n')

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          **kwargs) -> None:
        print(f"train batch <{self.count}> of"
              f" size <{train_batch.count}>"
              f" total <{self.total}>")
        self.total += train_batch.count
        self.count += 1

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        print("trainer.train() result: <{}> -> <{}> episodes".format(
            trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True
