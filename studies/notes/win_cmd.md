python3.11 src/train.py datasets/emotion_data.json --ngram-order 6 --recursive --reset --json-chunk-size 1000 --eval-samples 3 --eval-pool-size 500 --chunk-eval-percent 10.0 --eval-interval 1000 --profile-ingest --decoder-presence-penalty 0.5 --decoder-frequency-penalty 0.25

# Number of evaluation to do: --eval-interval 50