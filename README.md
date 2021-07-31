# 1. Introduction
## 1.1. Repo contetns
This simulator contains many scenarios for both centralised and edge cloud

TODO introduce paper(s), theory and background
1. cloud:
   1. learned environment
   2. auxiliary node
   3. greedy mitigator
   4. absolute action map
   5. No action for illegal states
2. edge:
   1. edge users and network
   2. latency reduction
   3. energy comsomption
## 1.2. RL Agent Characteristics
**Observations** TODO a [sample observation](docs/learning/raw_observation.txt) with the [preprocessed observation](docs/learning/preprocessed_observation.txt)

**Actions** TODO

**Rewards** TODO

# 2. Environment Setup
## Option 1: Setup the development environment with Docker (recommended)
1. Install [docker](https://docs.docker.com/engine/install/ubuntu/)
2. Setup [vscode development](docs/vscode-development.md)

## Option 2: Setup the development for reproducing paper results (recommended)
1. install [docker](https://docs.docker.com/engine/install/ubuntu/)
2. TODO

## Option 3: Setup the environment in your machine
1. Download source code from GitHub
   ```
    git clone https://github.com/saeid93/CCGrid-paper
   ```
2. Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    conda create --name pycloudsim python=3
   ```
4. Activate conda environment
   ```
    conda activate pycloudsim
   ```
5. if you want to use GPUs make sure that you have the correct version of CUDA and cuDNN installed from [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
6. Use [PyTorch](https://pytorch.org/) or [Tensorflow](https://www.tensorflow.org/install/pip#virtual-environment-install) isntallation manual to install one of them based-on your preference

7. Install the followings
   ```
    sudo apt install cmake libz-dev
   ```
8. Install requirements
   ```
    pip install -r requirements.txt
   ```
9. setup [tensorboard monitoring](docs/tensorboard-monitoring.md)

# 3. Kubernetes Cluster Setup
There are several options for setting up a Kuberentes cluster. As long as you have access to the kube config address e.g. `~/.kube/config` in your config files specified.

We have used [Google Cloud Platform](https://cloud.google.com/) for our experiments.
You can find the toturial for creating the cluster on google cloud and locally in the followings:

* [Google cloud installation](docs/kubernetes/installation-gcp.md)
* [Local cluster setup installation](docs/kubernetes/installation-local.md)

# 4. Project Structure

1. [data](/data)
2. [docs](/docs)
3. [experiments](/experiments)
4. [gym_cloudsim](/gym_cloudsim)
5. [mobility dataset preprossing](/mobility-dataset-preprossing)

The code is separated into three modules
   1. [data/](/data): This is the folder containing all the configs and results of the project. Could be anywhere in the project.
   2. [mobility-dataset-preprossing/](/mobility-dataset-preprossing): These scripts are used for preprocessing the [Cabspotting](https://privamov.github.io/accio/docs/datasets.html) and [California towers stations](antennasearch.com) dataset used in dataset-driven datasets.
   3. [gym_cloudsim/](/gym_cloudsim): the core simulation library with [Open-AI gym](https://gym.openai.com/) interface
   4. [/experiments](/experiments): experiments of the paper and the reinforcement learning side of codes.

## 4.1. [gym_cloudsim](/gym_cloudsim)
### Structure
* [src/](./gym_cloudsim/src): The folder containing the gym_cloudsim simulators. This should be installed for using.
  * [gym_cloudsim/](./gym_cloudsim/src/gym_cloudsim)
    * [dataset/](./gym_cloudsim/src/gym_cloudsim/dataset)
      * [__init__.py](./gym_cloudsim/src/gym_cloudsim/dataset/__init__.py)
      * [dataset_generator.py](./gym_cloudsim/src/gym_cloudsim/dataset/dataset_generator.py): Generate the datasets, nodes and services.
      * [network_generator.py](./gym_cloudsim/src/gym_cloudsim/dataset/network_generator.py): Generate the edge network with stations.
      * [trace_generator.py](./gym_cloudsim/src/gym_cloudsim/dataset/trace_generator.py): Generate the user movement traces.
      * [workload_generator.py](./gym_cloudsim/src/gym_cloudsim/dataset/workload_generator.py): Generates resource usages of the services.
    * [envs/](./gym_cloudsim/src/gym_cloudsim/envs)
      * [__init__.py](./gym_cloudsim/src/gym_cloudsim/envs/__init__.py)
      * [base_env.py](./gym_cloudsim/src/gym_cloudsim/envs/base_env.py): Basic functionalities of environments are implemented here. All other enviornments are inherited from the base_env.
      * [cloud_env.py](./gym_cloudsim/src/gym_cloudsim/envs/cloud_env.py): Base cloud environments for generating different versions of the loud environment.
      * [edge_env.py](./gym_cloudsim/src/gym_cloudsim/envs/edge_env.py): Base edge environments for generating different versions of the loud environment.
      * [greedy_env.py](./gym_cloudsim/src/gym_cloudsim/envs/greedy_env.py): Base greedy environments for generating different versions of the cloud environment.
    * [envs_extensions/](./gym_cloudsim/src/gym_cloudsim/envs_extensions): Functionalities to be added to the environments.
      * [__init__.py](./gym_cloudsim/src/gym_cloudsim/envs_extensions/__init__.py)
      * [mitigators.py](./gym_cloudsim/src/gym_cloudsim/envs_extensions/mitigators.py)
      * [render.py](./gym_cloudsim/src/gym_cloudsim/envs_extensions/render.py)
      * [reward.py](./gym_cloudsim/src/gym_cloudsim/envs_extensions/reward.py)
      * [step.py](./gym_cloudsim/src/gym_cloudsim/envs_extensions/step.py)
      * [take_action.py](./gym_cloudsim/src/gym_cloudsim/envs_extensions/take_action.py)
      * [__init__.py](./gym_cloudsim/src/gym_cloudsim/network/__init__.py)
      * [network_base.py](./gym_cloudsim/src/gym_cloudsim/network/network_base.py)
      * [network_builder_dataset.py](./gym_cloudsim/src/gym_cloudsim/network/network_builder_dataset.py)
      * [network_builder_random.py](./gym_cloudsim/src/gym_cloudsim/network/network_builder_random.py)
      * [network_simulator.py](./gym_cloudsim/src/gym_cloudsim/network/network_simulator.py)
    * [util/](./gym_cloudsim/src/gym_cloudsim/util)
      * [kubernetes_utils/](./gym_cloudsim/src/gym_cloudsim/util/kubernetes_utils)
        * [__init__.py](./gym_cloudsim/src/kubernetes_utils/__init__.py)
        * [aux.py](./gym_cloudsim/src/kubernetes_utils/aux.py)
        * [cluster.py](./gym_cloudsim/src/kubernetes_utils/cluster.py)
        * [utils.py](./gym_cloudsim/src/kubernetes_utils/utils.py)
      * [__init__.py](./gym_cloudsim/src/gym_cloudsim/util/__init__.py)
      * [annotations.py](./gym_cloudsim/src/gym_cloudsim/util/annotations.py)
      * [check_config.py](./gym_cloudsim/src/gym_cloudsim/util/check_config.py)
      * [constants.py](./gym_cloudsim/src/gym_cloudsim/util/constants.py)
      * [data_loader.py](./gym_cloudsim/src/gym_cloudsim/util/data_loader.py)
      * [logger.py](./gym_cloudsim/src/gym_cloudsim/util/logger.py)
      * [parser.py](./gym_cloudsim/src/gym_cloudsim/util/parser.py)
      * [plot.py](./gym_cloudsim/src/gym_cloudsim/util/plot.py)
      * [plot_workload.py](./gym_cloudsim/src/gym_cloudsim/util/plot_workload.py)
      * [preprocessors.py](./gym_cloudsim/src/gym_cloudsim/util/preprocessors.py)
    * [__init__.py](./gym_cloudsim/src/gym_cloudsim/__init__.py)
* [setup.py](./gym_cloudsim/setup.py)
* [unit_tests.py](./gym_cloudsim/unit_tests.py)

### Usage
Go to the [gym_cloudsim/](./CCGrid-paper/gym_cloudsim) and install the library in the editable mode with
   ```
   pip install -e .
   ```

## 4.2. [data](/data)
### Structure
Link the data folder (could be placed anywhere in your harddisk) to the project. A sample of the data folder is available at [data/](./CCGrid-paper/data). The data folder has the follwing structure:

* [backup/](./data/backup): Folder to save backed up files of the result folder generated by [experiments/utils/data_backup.py](experiments/utils/data_backup.py).
* [configs/](./data/configs): Configs that are used in different part of the code.
  * [experimental/](./data/configs/experimental): Configs that are used for checks during the developments, not the real experiments but have the exact same structure.
    * [config_check_env.json](./data/configs/experimental/config_check_env.json): Configs to check a environment functionalities. Used in [experiments/check_scripts/check_env.py](experiments/check_scripts/check_env.py)
    * [config_check_learned.json](./data/configs/experimental/config_check_learned.json): Configs that are used to check a learned environment. Used in [experiments/check_scripts/check_learned.py](experiments/check_scripts/check_learned.py)
    * [config_run.json](./data/configs/experimental/config_run.json): Configs used for learning and setting ray rllib hyperparameters. Used in [experiments/learning/learner.py](experiments/learning/learner.py).
  * [generation-configs/](./data/configs/generation-configs): Used in generating datasets, network, movement traces and workload generations.
    * [dataset-generation/](./data/configs/generation-configs/dataset-generation): Generating datasets configs used in [experiments/dataset/generate_dataset.py](experiments/dataset/generate_dataset.py)
      * [my-dataset/](./data/configs/generation-configs/dataset-generation/my-dataset): We could have several folder with the same structure (with a config.json inside it)
        * [config.json](./data/configs/generation-configs/dataset-generation/my-dataset/config.json)
    * [network-generation/](./data/configs/generation-configs/network-generation): Generating networks configs used in [experiments/dataset/generate_network.py](experiments/dataset/generate_network.py)
      * [my-network/](./data/configs/generation-configs/network-generation/my-network): We could have several folder with the same structure (with a config.json inside it)
        * [config.json](./data/configs/generation-configs/network-generation/my-network/config.json)
    * [trace-generation/](./data/configs/generation-configs/trace-generation): Generating networks configs used in [experiments/dataset/generate_trace.py](experiments/dataset/generate_trace.py)
      * [my-trace/](./data/configs/generation-configs/trace-generation/my-trace): We could have several folder with the same structure (with a config.json inside it)
        * [config.json](./data/configs/generation-configs/trace-generation/my-trace/config.json)
    * [workload-generation/](./data/configs/generation-configs/workload-generation): Generating workload configs used in [experiments/dataset/generate_workload.py](experiments/dataset/generate_workload.py)
      * [my-workload/](./data/configs/generation-configs/workload-generation/my-workload): We could have several folder with the same structure (with a config.json inside it)
        * [config.json](./data/configs/generation-configs/workload-generation/my-workload/config.json)
  * [real/](./data/configs/real): configs for the real-world experiments.
    * [0-PG/](./data/configs/real/0-PG): Each folder is saved in a folder with the experiment's name.
      * [config_run.json](./data/configs/real/0-PG/config_run.json)
* [dataset_metadata/](./data/dataset_metadata): location of real-world users and stations. Generated by the [mobility-dataset-preprossing/main.py](mobility-dataset-preprossing/main.py). Currently manual. TODO make it automatic.
  * [0/](./data/dataset_metadata/0)
    * [stations.txt](./data/dataset_metadata/0/stations.txt)
    * [users.txt](./data/dataset_metadata/0/users.txt)
* [datasets/](./data/datasets): TODO complete after the example
* [results/](./data/results):: TODO complete after the example
* [results-experimental/](./data/results-experimental): TODO complete after the example


### Usage
Go to [/experiments/utils/constants.py](/experiments/utils/constants.py) and set the path to your data and project folders in the file. For example:
   ```
   DATA_PATH = "/Users/saeid/Codes/CCGrid-paper/data"
   ```
## 4.3. [mobility dataset preprossing](/mobility-dataset-preprossing)
TODO change after integration
### Structure

* [ETL.py](./mobility-dataset-preprossing/ETL.py)
* [README.md](./mobility-dataset-preprossing/README.md)
* [Utils.py](./mobility-dataset-preprossing/Utils.py)
* [main.py](./mobility-dataset-preprossing/main.py)
* [requirements.txt](./mobility-dataset-preprossing/requirements.txt)

### Usage
1. Install the requirements:
```bash
pip install -r requirements.txt
```
2. Run the code:
```bash
python main.py
```

3. The options are as follows:
```
Usage: main.py [OPTIONS]

Options:
  -d, --dataset TEXT      Directory of Cabspotting data set  [default:
                          data/*.txt]

  -g, --get BOOLEAN       Get data set from the internet  [default: False]
  -u, --url TEXT          The url of Cabspotting data set  [default: ]
  -i, --interval INTEGER  Enter the intervals between two points in seconds
                          [default: 100]

  --help                  Show this message and exit.

```
4. Enjoy!


formant of the parsed users mobility dataset.
Users data attributes:
1. Creating a CSV file with the size of: 2 * #NUM_OF_USERS + 1, each 3 column contains
the information of one user.
2. For instance, in case when there are 3 users in the network, we have a 7-column table as
follows:

| timesteps | lan1 | lat1 | lan2 | lat2 |
|-----------|------|------|------|------|
|0|35.76727803828804|51.35991084161459|35.76031345070641|1.39458643788907|
|1|35.76727803828856|51.35991084161443|35.76031345070765|1.39458643788633|
|2|35.76727803828856|51.35991084161443|35.76031345070765|1.39458643788633|


## 4.4. [experiments](/experiments)

### 4.4.1. [Data Generation](experiments/dataset)
The dataset, workloads, networks and traces are generated in the following order:
1. **Datasets**: Nodes, services, their capacities and their initial placements. 
2. **Workloads**: The workload for each dataset that determines the resource usage at each time step. This is built on top of the datasets built on step 1. Each dataset can have several workloads.
3. **Networks**: The edge network for each dataset that generates the network with users and stations for the simulation. This contains the network object containing the nodes and stations and intitial location of the users. The network can be built both randomly of based-on the [Cabspotting](https://privamov.github.io/accio/docs/datasets.html) and [California towers stations](antennasearch.com) dataset. This is built on top of the nodes-services datasets built on step 1. Each dataset can have several networks.
4. **Traces**: The movement traces for each network. This is the location of each user at each timestep. This can be a random or based-on [Cabspotting](https://privamov.github.io/accio/docs/) dataset. This is built on top of the networks built in step 3.


To generate the datasets, workloads, networks and traces, first go to your data folder (remember data could be anywhere in your disk just point the data folder as  [/experiments/utils/constants.py](/experiments/utils/constants.py)).

### 4.4.1.1. [Generating the Datasets](experiments/dataset/generate_dataset.py)

Go to the your dataset generation config [/data/configs/generation-configs/dataset-generation](/data/configs/generation-configs/dataset-generation) make a folder named after your config and make the `config.json` in the folder e.g. see the `my-dataset` in the sample [/data](/data) folder [/data/configs/generation-configs/dataset-generation/my-dataset/config.json](/data/configs/generation-configs/dataset-generation/my-dataset/config.json). Then run the [experiments/dataset/generate_dataset.py](experiments/dataset/generate_dataset.py) with the following script:
```
python generate_dataset.py [OPTIONS]

Options:
  --dataset-config-folder TEXT      config-folder
  [default:                         my-dataset] 
```
For a full list of `config.json` parameters options see [dataset-configs-options](/docs/configs-parameters/dataset-generation.md). The results will be saved in [/data/datasets/<dataset_id>](/data/datasets).

### 4.4.1.2. [Generating the Workloads](experiments/dataset/generate_workload.py)

Go to the your workload generation config [/data/configs/generation-configs/workload-generation](/data/configs/generation-configs/dataset-generation) make a folder named after your config and make the `config.json` in the folder e.g. see the `my-workload` in the sample [/data](/data) folder [/data/configs/generation-configs/workload-generation/my-workload/config.json](/data/configs/generation-configs/dataset-generation/my-workload/config.json). For a full list of `config.json` see TODO. Then run the [experiments/dataset/generate_dataset.py](experiments/dataset/generate_dataset.py) with the following script:
```
python generate_workload.py [OPTIONS]

Options:
  --workload-config-folder TEXT      config-folder
  [default:                          my-workload] 
```
For a full list of `config.json` parameters options see [workload-configs-options](/docs/configs-parameters/workload-generation.md). The results will be saved in [/data/datasets/<dataset_id>/<workload_id>](/data/datasets).
<br />
### 4.4.1.3. [Generating the Networks](experiments/dataset/generate_network.py)

Go to the your dataset generation config [/data/configs/generation-configs/network-generation](/data/configs/generation-configs/network-generation) make a folder named after your config and make the `config.json` in the folder e.g. see the `my-network` in the sample [/data](/data) folder [/data/configs/generation-configs/dataset-generation/my-network/config.json](/data/configs/generation-configs/dataset-generation/my-dataset/config.json). Then run the [experiments/dataset/generate_network.py](experiments/dataset/generate_network.py) with the following script:
```
python generate_network.py [OPTIONS]

Options:
  --network-config-folder TEXT      config-folder
  [default:                         my-network] 
```
For a full list of `config.json` parameters options see [network-configs-options](/docs/configs-parameters/network-generation.md). The results will be saved in [/data/datasets/<dataset_id>/<network_id>](/data/datasets).

### 4.4.1.4. [Generating the Traces](experiments/dataset/generate_trace.py)

Go to the your trace generation config [/data/configs/generation-configs/trace-generation](/data/configs/generation-configs/network-generation) make a folder named after your config and make the `config.json` in the folder e.g. see the `my-trace` in the sample [/data](/data) folder [/data/configs/generation-configs/trace-generation/my-trace/config.json](/data/configs/generation-configs/dataset-generation/my-dataset/config.json). Then run the [experiments/dataset/generate_trace.py](experiments/dataset/generate_trace.py) with the following script:
```
python generate_trace.py [OPTIONS]

Options:
  --trace-config-folder TEXT        config-folder
  [default:                         my-trace] 
```
For a full list of `config.json` parameters options see [trace-configs-options](/docs/configs-parameters/trace-generation.md). The results will be saved in [/data/datasets/<dataset_id>/<network_id>/<trace_id>](/data/datasets).

### 4.4.2. [Learning](experiments/learning), [checking](experiments/check_scripts) and [analysis](experiments/analysis)

### 4.4.2.1. [Training the agent](experiments/learning/learner.py)

1. change the training parameters in `<configs-path>/real/<experiment-folder>/config_run.json`. For more information about the hyperparamters in this json file see [hyperparameter guide](docs/learning/hyperparameter-guide.md)
2. To train the environments go to the parent folder and run the following command.
```
python experiments/learning/learners.py --mode real --local-mode false --config-folder PPO --type-env 0 --dataset-id 0 --workload-id 0 --use-callback true
```

### 4.4.2.2. [Test the learned agent - simulation](experiments/check_scripts/check_learned.py)

TODO

### 4.4.2.3. [Test the learned agent - kubernetes](experiments/check_scripts/check_learned.py)

1. set the kubernetes INTERVAL variable in the [utilisation server](Dockerfile/Kubernetes-engine/utilization-server/app.py) 
TODO

### 4.4.2.4. [Analysis](experiments/analysis)

TODO

## 4.4.3. [Kubernetes interface](gym_cloudsim/src/gym_cloudsim/util/kubernetes_utils)

The Kubernetes interface is designed based-on the Kubernetes [api version 1](https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.21/).

The main operations that are currently implemented are:
* creating
  * cluster
  * utilisation server
  * pods
* actions
  * moving pods
  * deleting pods
  * cleaning nemespace
* monitoring
  * get nodes resource usage
  * get pods resource usage

a sample of using the interface can be found [here](/experiments/kube_operations.py)
## 4.4.4. [Extra scripts](experiments/utils)

1. [/experiments/utils/data_backup.py](/experiments/utils/data_backup.py): Use this for backing up your data folder.

2. 

# 5. Other

1. [Step by step guide to trainig the code on EECS](docs/step-by-step-guides/EECS.md)

2. [Step by step guide to trainig the code on GKE](docs/step-by-step-guides/GKE.md) TODO

3. [List of running ray problems](docs/problems/ray.md)

4. [List of QMUL EECS problems](docs/problems/eecs-server-setup.md)

5. [Tensorboard Monitoring](docs/monitorings/tensorboard.md)

6. [Cluster Monitoring](docs/monitorings/cluster.md) TODO
