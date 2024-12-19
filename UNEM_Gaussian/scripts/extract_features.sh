sampling='uniform'
arch='resnet18'
train_unrolling=False
evaluate=True

for dataset in 'tiered'
do
/usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} \
train_unrolling ${train_unrolling} method 'paddle' evaluate ${evaluate} number_tasks 1 batch_size 1
done


arch='wideres'

for dataset in 'mini' 'tiered'
do
/usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} \
train_unrolling ${train_unrolling} method 'paddle' evaluate ${evaluate} number_tasks 1 batch_size 1
done



