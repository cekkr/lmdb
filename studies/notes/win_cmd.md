python3.11 src/train.py datasets/emotion_data.json --ngram-order 5 --recursive --reset --json-chunk-size 50 --eval-samples 3 --eval-pool-size 20 --chunk-eval-percent 1.0

# Exclude for no waiting eval logging: --eval-interval 500