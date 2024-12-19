acc_vs_gamma_study=True

for dataset in 'caltech101' 'dtd' 'eurosat' 'flowers102' 'food101' 'oxfordpets' 'ucf101' 'fgvcaircraft' 'stanfordcars'; do
    for lamda in $(seq 0 75 7500); do
        /usr/bin/python3 main.py --opts dataset ${dataset} acc_vs_gamma_study ${acc_vs_gamma_study} \
                        lamda ${lamda} iter 10 u_train False method 'em_dirichlet'
    done
done

arch='resnet18'
for dataset in 'sun397'; do
    for lamda in $(seq 0 75 7500); do
        /usr/bin/python3 main.py --opts dataset ${dataset} acc_vs_gamma_study ${acc_vs_gamma_study} \
                        lamda ${lamda} iter 10 u_train False method 'em_dirichlet' batch_size 10
    done
done


#/usr/bin/python3 plot.py