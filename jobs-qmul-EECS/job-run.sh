for i in {0..4}
do
   /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series $1 --config-file $2-$i --dataset-id 6 --network-id 1 --trace-id 2
done
