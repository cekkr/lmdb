- The tokenization is too arbitrary especially in punctuation splitting, in fact results use to have difficulties in puntactions writing while generating.
- In @sentence_parts.py, there is another arbitrary forcing of words: _EMOTION_WORDS. Why having a dataset if you force the words? Remove any ARBITRARY reference.
- Try to add the support to an external and well established tokenizator, to use overall during training and in-training evaluations.

Let's take this "default train command":
python3.11 src/train.py datasets/emotion_data.json --ngram-order 4 --recursive --reset --json-chunk-size 50 --eval-samples 2 --eval-pool-size 5000 --chunk-eval-percent 10.0 --eval-interval 5000 --profile-ingest --decoder-presence-penalty 0.5 --decoder-frequency-penalty 0.25 --context-dimensions 1-4,5-13
Is it resetting the cheetah db? Is it using an ad hoc database name for this training not using the default one?
The penalty parameters ar respected seen the too many repeatitions?

Then, try to make context-dimension parameter better, with more dimensions instead of simple "word-sentence". It should be more "irrational" and with support of more dimension.
Example --context-dimension 4,8,4 (3 dimension, the first of 4 vectors, the second 8 and the third 4)