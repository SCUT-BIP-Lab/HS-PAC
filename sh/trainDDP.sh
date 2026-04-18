torchrun --nproc_per_node=2 ../src/main/train.py \
    --conf_file ../conf/Mobrecon_freihand_mesh.conf \
    --mode train \
    --gpu 0,1
