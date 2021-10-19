import os
from mobile_kube.envs import (
    SimEdgeEnv,
    SimBinpackingEnv,
    SimGreedyEnv,
    KubeEdgeEnv,
    KubeBinpackingEnv,
    KubeGreedyEnv
)
# dfined by the user
# DATA_PATH = "/home/jdoyledithencom/data-repos/myenv"
DATA_PATH = "/Users/saeid/Codes/mobile-kube/data"
# DATA_PATH = "/homes/sg324/CCGrid-data-repo/myenv"
# DATA_PATH = "/home/sdghafouri/mobile-kube/data"

# generated baesd on the users' path
DATASETS_PATH = os.path.join(DATA_PATH, "datasets")
RESULTS_PATH = os.path.join(DATA_PATH, "results")
CONFIGS_PATH = os.path.join(DATA_PATH, "configs")
BACKUP_PATH = os.path.join(DATA_PATH, "backup")
PLOTS_PATH = os.path.join(DATA_PATH, "plots")
EXPERIMENTS_PATH = os.path.join(DATA_PATH, "experiments")
DATASETS_METADATA_PATH = os.path.join(DATA_PATH, "dataset_metadata") 

def _create_dirs():
    """
    create directories if they don't exist
    """
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    if not os.path.exists(CONFIGS_PATH):
        os.makedirs(CONFIGS_PATH)
    if not os.path.exists(BACKUP_PATH):
        os.makedirs(BACKUP_PATH)
    if not os.path.exists(EXPERIMENTS_PATH):
        os.makedirs(EXPERIMENTS_PATH)
    if not os.path.exists(DATASETS_METADATA_PATH):
        os.makedirs(DATASETS_METADATA_PATH)
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

_create_dirs()

ENVS = {
    'sim-edge': SimEdgeEnv,
    'sim-binpacking': SimBinpackingEnv,
    'sim-greedy': SimGreedyEnv,
    'kube-edge': KubeEdgeEnv,
    'kube-binpacking': KubeBinpackingEnv,
    'kube-greedy': KubeGreedyEnv
}

ENVSMAP = {
    'sim-edge': 'SimEdgeEnv-v0',
    'sim-binpacking': 'SimBinpackingEnv-v0',
    'sim-greedy': 'SimGreedyEnv-v0',
    'kube-edge': 'KubeEdgeEnv-v0',
    'kube-binpacking': 'KubeBinpackingEnv-v0',
    'kube-greedy': 'KubeGreedyEnv-v0',
}
