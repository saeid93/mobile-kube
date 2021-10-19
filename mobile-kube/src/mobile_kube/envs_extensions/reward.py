import numpy as np
from copy import deepcopy
from typing import Tuple, Dict, Any

def _reward(
    self, *, num_moves: int,
    num_overloaded: int,
    users_distances: np.array = None) -> Tuple[
        float, Dict[str, Any]]:

    if num_overloaded > 0:
        return _reward_illegal(self, num_overloaded), {
            "reward_move": 0,
            "reward_illegal": 0,
            "reward_consolidation": 0,
            "reward_variance": 0,
            "reward_latency": 0
            }

    reward_total_edge, rewards_edge = _reward_latency(self, users_distances)
    reward_total_cloud, rewards_cloud = _reward_cloud(
        self,
        num_moves=num_moves,
        num_overloaded=num_overloaded)
    reward_total = reward_total_edge + reward_total_cloud
    rewards_edge.update(rewards_cloud)
    rewards = deepcopy(rewards_edge)
    return reward_total, rewards

def _reward_cloud(self, *, num_moves: int,
                num_overloaded: int = 0) -> Tuple[float, Dict[str, Any]]:
    """absolute reward function based-on the absolute number
    of consolidated servers
    steps:
        1. normalise
        2. times penalties
        3. only negative rewards in overloaded states
    """
    reward_move = 0
    reward_illegal = 0
    reward_consolidation = 0
    reward_variance = 0
    reward_move = _reward_move(self, num_moves)
    reward_variance = _reward_variance(self)
    reward_consolidation = _reward_consolidation(self)

    reward_total = reward_move + reward_illegal + \
        reward_consolidation + reward_variance

    rewards = {
        "reward_move": reward_move,
        "reward_illegal": reward_illegal,
        "reward_consolidation": reward_consolidation,
        "reward_variance": reward_variance
    }
    return reward_total, rewards


def _reward_latency(self, users_distances: np.array) -> Tuple[float, Dict[str, Any]]:
    """
    calcuate the edge reward
    """
    # normalise distances with the largest distance
    if self.normalise_latency:
        users_distances = users_distances/self.normalise_factor
    reward = np.sum(users_distances)
    reward *= self.penalty_latency
    reward = 1/reward
    # clip the reward if it's greater 10 
    if reward > 10:
        reward = 10
    rewards = {'reward_latency': reward}
    return reward, rewards


def _reward_move(self, num_moves: int):
    """reward for the number of moves
    """
    movement_factor = num_moves/self.num_services
    reward_move = self.penalty_move * movement_factor
    return reward_move


def _reward_variance(self):
    """compute the variance reward
    """
    reward_factor = np.sum(np.var(
        self.nodes_resources_request_frac, axis=1))
    reward_variance = reward_factor * self.penalty_variance
    return reward_variance


def _reward_illegal(self, prev_num_overloaded: int):
    """reward for the number of illegal factors
    """
    nodes_overloaded_factor = prev_num_overloaded/self.num_nodes
    reward_illegal = self.penalty_illegal * nodes_overloaded_factor
    return reward_illegal


def _reward_consolidation(self):
    """reward for the num_consolidated
    """
    consolidation_factor = self.num_consolidated/self.num_nodes
    reward_consolidation = self.penalty_consolidated *\
        consolidation_factor
    return reward_consolidation
