# Supported --dataset: AmazonReviews2014 / AmazonReviews2018 / AmazonReviews2023
# Raw files are downloaded into cache/<dataset>/<category>/raw/ on first run
# (or put them there manually with the same file names).

# Amazon 2018 (amazon_v2), 5-core reviews
for seed in {2024..2025}; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --dataset=AmazonReviews2018 \
        --category=Video_Games \
        --rand_seed=${seed} \
        --weight_decay=0.07 \
        --lr=0.001 \
        --d_model=256 \
        --d_ff=2048 \
        --n_hash_buckets=256
done

# Amazon 2014 (original ActionPiece setting, e.g. Sports_and_Outdoors / Beauty / CDs_and_Vinyl)
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset=AmazonReviews2014 --category=Sports_and_Outdoors

# Amazon 2023
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset=AmazonReviews2023 --category=Video_Games \
#     --weight_decay=0.07 --lr=0.001 --d_model=256 --d_ff=2048 --n_hash_buckets=256
