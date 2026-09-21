for seed in {2024..2025}; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --category=Musical_Instruments \
        --rand_seed=${seed} \
        --weight_decay=0.07 \
        --lr=0.001 \
        --d_model=256 \
        --n_hash_buckets=256
done

# for seed in {2024..2025}; do
#     CUDA_VISIBLE_DEVICES=2 python main.py \
#         --category=Industrial_and_Scientific \
#         --rand_seed=${seed} \
#         --weight_decay=0.15 \
#         --lr=0.005 \
#         --n_hash_buckets=256
# done

# for seed in {2024..2025}; do
#     CUDA_VISIBLE_DEVICES=1 python main.py \
#         --category=Video_Games \
#         --rand_seed=${seed} \
#         --weight_decay=0.07 \
#         --lr=0.001 \
#         --d_model=256 \
#         --d_ff=2048 \
#         --n_hash_buckets=256
# done