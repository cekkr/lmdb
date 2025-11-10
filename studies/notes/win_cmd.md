python3.11 src/train.py datasets/emotion_data.json --ngram-order 5 --recursive --reset --json-chunk-size 500 --eval-samples 2 --eval-pool-size 200 --chunk-eval-percent 1.0 --eval-interval 5000 --profile-ingest

# Number of evaluation to do: --eval-interval 50