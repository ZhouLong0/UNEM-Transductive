sampling='uniform'


# ==============================================  mini-imagenet
train_unrolling=False
evaluate=True
used_set_support='test'       
used_set_query='test'

# method=baseline
# for dataset in 'mini' 'tiered'; do
#     for arch in 'resnet18' 'wideres'; do
#         /usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} \
#         train_unrolling ${train_unrolling} method ${method} evaluate ${evaluate} used_set_support ${used_set_support} used_set_query ${used_set_query} 
#     done
# done


# method=ici
# for dataset in 'mini'; do
#     for arch in 'resnet18' 'wideres'; do
#         /usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} \
#         train_unrolling ${train_unrolling} method ${method} evaluate ${evaluate} used_set_support ${used_set_support} used_set_query ${used_set_query} 
#     done
# done

# method=ici
# for dataset in 'mini'; do
#     for arch in 'resnet18' 'wideres'; do
#         /usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} \
#         train_unrolling ${train_unrolling} method ${method} evaluate ${evaluate} used_set_support ${used_set_support} used_set_query ${used_set_query} 
#     done
# done

# method=bdcspn
# for dataset in 'mini' 'tiered'; do
#     for arch in 'resnet18' 'wideres'; do
#         /usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} \
#         train_unrolling ${train_unrolling} method ${method} evaluate ${evaluate} used_set_support ${used_set_support} used_set_query ${used_set_query} 
#     done
# done

# method=pt_map
# for dataset in 'mini' 'tiered'; do
#     for arch in 'resnet18' 'wideres'; do
#         /usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} \
#         train_unrolling ${train_unrolling} method ${method} evaluate ${evaluate} used_set_support ${used_set_support} used_set_query ${used_set_query} 
#     done
# done

# method=laplacianshot
# for dataset in 'mini' 'tiered'; do
#     for arch in 'resnet18' 'wideres'; do
#         /usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} \
#         train_unrolling ${train_unrolling} method ${method} evaluate ${evaluate} used_set_support ${used_set_support} used_set_query ${used_set_query} 
#     done
# done

method=tim
for dataset in 'tiered'; do
    for arch in 'wideres'; do
        /usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} batch_size 20\
        train_unrolling ${train_unrolling} method ${method} evaluate ${evaluate} used_set_support ${used_set_support} used_set_query ${used_set_query} 
    done
done

# method=alpha_tim
# for dataset in 'tiered'; do
#     for arch in 'resnet18' 'wideres'; do
#         /usr/bin/python3 -m main --opts dataset ${dataset} arch ${arch} sampling ${sampling} batch_size 50\
#         train_unrolling ${train_unrolling} method ${method} evaluate ${evaluate} used_set_support ${used_set_support} used_set_query ${used_set_query} 
#     done
# done