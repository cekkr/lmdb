python3.11 src/train.py datasets/emotion_data.json --ngram-order 6 --recursive --reset --json-chunk-size 500 --eval-samples 3 --eval-pool-size 200 --chunk-eval-percent 10.0 --eval-interval 1000 --profile-ingest --decoder-presence-penalty 0.4 --decoder-frequency-penalty 0.2

# Number of evaluation to do: --eval-interval 50