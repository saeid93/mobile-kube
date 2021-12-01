for i in {0,2,4}
do
   /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series $1 --config-file PPO-search-$i --dataset-id 6 --network-id $2 --trace-id $3
done
