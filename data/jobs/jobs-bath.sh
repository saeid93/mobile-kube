ADDRESS=/homes/sg324/CCGrid-paper
python ${ADDRESS}/experiments_scripts/learning_scripts/learners.py --mode real --local-mode true --config-folder constants-PPO-search --series 4 --type-env 3 --dataset-id 0 --workload-id 0 --use-callback true --checkpoint-freq 100
