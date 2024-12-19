acc_vs_gamma_study=True

arch='resnet18'
for dataset in 'mini' 'tiered' 'cub'; do
    for alpha in $(seq 0 5 500); do
        /usr/bin/python3 /media/eliott/inner_disk/PADDLE-master/main.py --opts dataset ${dataset} arch ${arch} evaluate True acc_vs_gamma_study ${acc_vs_gamma_study} \
                        alpha ${alpha} iter 100 train_unrolling False method paddle
    done
done

arch='wideres'
  for dataset in 'mini' 'tiered'; do
    for alpha in $(seq 0 5 500); do
        /usr/bin/python3 /media/eliott/inner_disk/PADDLE-master/main.py --opts dataset ${dataset} arch ${arch} evaluate True acc_vs_gamma_study ${acc_vs_gamma_study} \
                        alpha ${alpha} iter 100 train_unrolling False method paddle
    done
done

#/usr/bin/python3 plot.py