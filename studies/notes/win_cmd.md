python3.11 src/train.py datasets/emotion_data.json --ngram-order 4 --recursive --reset --json-chunk-size 200 --eval-samples 2 --eval-pool-size 200 --chunk-eval-percent 10.0 --eval-interval 5000 --profile-ingest

# Number of evaluation to do: --eval-interval 50