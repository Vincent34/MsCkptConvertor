python ./ms_and_tf_checkpoint_transfer_tools.py \
    --tf_ckpt_path="/home/chz/Downloads/mlperf_ckpt/bs64k_32k_ckpt_model.ckpt-28252" \
    --ms_ckpt_path="./bert_large.ckpt" \
    --new_ckpt_path="./ms_bert_large.ckpt" \
    --transfer_option="tf2ms"

