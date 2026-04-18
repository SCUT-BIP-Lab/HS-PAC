unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
export HF_ENDPOINT="https://hf-mirror.com"
python ../src/main/train.py \
    --conf_file ../conf/MultiScaleHMR_ho3d_mesh.conf \
    --mode train \
    --gpu 4



