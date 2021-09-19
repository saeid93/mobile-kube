"""
scripts to check a learned agent
based-on https://github.com/ray-project/ray/issues/9123
"""
import json
import os
import sys
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.pg as pg

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    RESULTS_PATH,
    CONFIGS_PATH
)
from experiments.utils import (
    # config_reader,
    action_pretty_print
)

class CheckLearned:
    def __init__(self, config_path) -> None:
        """
        It initiates the environment and load the saved
        checkpoint by reading three config files

        1. config_check_learned: the config of the chekcing
           learned, it might be a differnt workload and
           episode lenght but type_env and observation must
           be the same
           saved in:
           <experiment_configs_folder>/config_run.json
        2. trial_config: content of params.json which are the
           hyperparameter of a single trial in the experiment
           saved in:
           data/results/env/<env_id>/datasets/<dataset_id>/
           workloads/<workload_id>/experiments/<experiment_id>/
           <name_of_algorithm>/<trial>/params.json
        """

        type_env, dataset_id, workload_id_train, config_check_learned,\
            experiment_id, trial_string = self.load_test_info(config_path)

        self.checkpoint_path, config_trial, self.algorithm =\
            self.load_trial_config(
                type_env, dataset_id, workload_id_train, experiment_id,
                trial_string)

        env_config, self.env = self.generate_env(
            type_env, config_check_learned)

        self.ray_config = self.generate_ray_config(
            self.env, config_trial, env_config)

    def load_test_info(self, config_path):
        config_check_learned_path = os.path.join(
            CONFIGS_PATH,
            'check',
            config_path)
        with open(config_check_learned_path, 'rb') as in_file:
            config_check_learned = json.load(in_file)
        metadata = config_check_learned['metadata']
        type_env = metadata['type_env']
        dataset_id = metadata['dataset_id']
        workload_id_train = metadata['workload_id_train']
        experiment_id = metadata['experiment_id']
        trial_string = metadata['trial_string']
        return type_env, dataset_id, workload_id_train,\
            config_check_learned, experiment_id, trial_string

    def load_trial_config(self, type_env, dataset_id, workload_id_train,
                          experiment_id, trial_string):
        experiment_folder_path = os.path.join(
            RESULTS_PATH,
            "envs",        str(type_env),
            "datasets",    str(dataset_id),
            "workloads",   str(workload_id_train),
            "experiments", str(experiment_id))
        # find algorithm
        experiment_folder_content = os.listdir(experiment_folder_path)
        experiment_folder_content.remove('experiment_config.json')
        algorithm = experiment_folder_content[0]
        # load the config_trial
        trial_folder_path = os.path.join(
            experiment_folder_path,
            algorithm,
            trial_string)
        config_trial_path = os.path.join(trial_folder_path, 'params.json')
        with open(config_trial_path, 'rb') as json_file:
            config_trial = json.load(json_file)
        to_delete = ['callbacks', 'env', 'env_config']
        for item in to_delete:
            if item in to_delete:
                del config_trial[item]
        # generate checkpoint path
        files = os.listdir(trial_folder_path)
        checkpoint = ''
        for item in files:
            if 'checkpoint' in item:
                checkpoint = item
                break
        assert checkpoint != '', ("the experiment does not "
                                  "have a saved checkpoint")
        checkpoint_path = os.path.join(experiment_folder_path,
                                       algorithm, trial_string,
                                       checkpoint,
                                       checkpoint.replace('_', '-'))
        return checkpoint_path, config_trial, algorithm

    def generate_env(self, type_env, config_check_learned):
        env_config = load_object(
            config_check_learned['metadata'],
            config_check_learned['env_config_base'])
        env = make_env(type_env, env_config)
        return env_config, env

    def check_learned(self):
        """
        check the learned agent
        """
        ray.init(local_mode=True)
        if self.algorithm == 'PPO':
            agent = ppo.PPOTrainer(
                config=self.ray_config, env=self.env.__class__)
        elif self.algorithm == 'A3C':
            agent = a3c.A3CTrainer(
                config=self.ray_config, env=self.env.__class__)
        elif self.algorithm == 'PG':
            agent = pg.PGTrainer(
                config=self.ray_config, env=self.env.__class__)
        agent.restore(self.checkpoint_path)
        # run until episode ends
        episode_reward = 0
        done = False
        obs = self.env.reset()
        while True:
            self.env.render()
            action = agent.compute_action(obs)
            obs, reward, done, info = self.env.step(action)
            # print(f"obs:\n{obs}")
            print(f"reward:\n<{reward}>")
            print(f"info:\n<{info}>")
            episode_reward += reward

    def generate_ray_config(self, env, config_trial, env_config):
        # generate the ray input config
        ray_config = {}
        ray_config.update(config_trial)
        ray_config.update({"env": env.__class__,
                           "env_config": env_config})
        return ray_config

# @click.command()
# @click.argument('config_file')
# def main(config_file):
def main():
    # TODO make it click
    # TODO load the trained agent (the states are necessarilly consistent)
    # TODO only the episode lenght and workload could be different rest should
    #      be the same so load everything from the trained agent env_config
    # TODO so have reward episode length and workload in your input
    experiment_config_folder = "env_0_dataset_3_workload_0_PPO"
    config_path = os.path.join(experiment_config_folder,
                               "config_check_learned.json")

    ins = CheckLearned(config_path)
    ins.check_learned()

if __name__ == "__main__":
    main()
