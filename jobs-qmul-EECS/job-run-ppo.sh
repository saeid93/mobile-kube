for i in {0,2,4}
do
   /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series 55 --config-file PPO-$i --dataset-id 6 --network-id 5 --trace-id 3
done

for i in {0,2,4}
do
   /homes/sg324/.conda/envs/central/bin/python /homes/sg324/mobile-kube/experiments/training/train.py --series 55 --config-file PPO-$i --dataset-id 6 --network-id 8 --trace-id 3
done
