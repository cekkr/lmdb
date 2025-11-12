python3.11 src/train.py datasets/emotion_data.json --ngram-order 5 --recursive --reset --json-chunk-size 50 --eval-samples 2 --eval-pool-size 5000 --chunk-eval-percent 10.0 --eval-interval 5000 --profile-ingest --decoder-presence-penalty 0.5 --decoder-frequency-penalty 0.25 --context-dimensions 1-4,5-13

# Number of evaluation to do: --eval-interval 50