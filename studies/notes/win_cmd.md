Datasets:
emotion_data.json
GPTeacher.json

Windows:
Complex training:
python3.11 src/train.py datasets/GPTeacher.json --ngram-order 5 --json-chunk-size 50 --eval-samples 2 --eval-variants 2 --chunk-eval-percent 20.0 --eval-interval 50000 --profile-ingest --decoder-presence-penalty 0.3 --decoder-frequency-penalty 0.1 --context-dimensions 8,12,16,22,32 --reset

Very complex:
python3.13 src/train.py datasets/GPTeacher.json --ngram-order 6 --json-chunk-size 100 --eval-samples 2 --eval-variants 2 --chunk-eval-percent 20.0 --eval-interval 1000 --profile-ingest --decoder-presence-penalty 0.3 --decoder-frequency-penalty 0.15 --context-dimensions 16,24,32,48,64 --cheetah-context-probe "Summaries about remote work" --cheetah-context-probe "Reflect on the previous lesson" --cheetah-eval-predict --reset

Very complex -1:
python3.13 src/train.py datasets/GPTeacher.json --ngram-order 4 --merge-max-tokens 3 --json-chunk-size 200 --eval-samples 2 --eval-variants 2 --chunk-eval-percent 5.0 --eval-interval 10000 --decoder-presence-penalty 0.3 --decoder-frequency-penalty 0.15 --context-dimensions 16,24,32,64 --reset --profile-ingest

Very complex but not too much:
python3.13 src/train.py datasets/GPTeacher.json --ngram-order 5 --json-chunk-size 100 --eval-samples 2 --eval-variants 2 --chunk-eval-percent 20.0 --eval-interval 1000 --profile-ingest --decoder-presence-penalty 0.4 --decoder-frequency-penalty 0.15 --context-dimensions 12,16,24,32,48 --cheetah-context-probe "Summaries about remote work" --cheetah-context-probe "Reflect on the previous lesson" --cheetah-system-stats --cheetah-token-weight 0.5 --cheetah-token-learning-rate 0.01 --reset

Very complex mac:
python3.14 src/train.py datasets/GPTeacher.json --ngram-order 6 --json-chunk-size 500 --eval-samples 3 --eval-variants 3 --chunk-eval-percent 10.0 --eval-interval 50000 --profile-ingest --decoder-presence-penalty 0.2 --decoder-frequency-penalty 0.05 --context-dimensions 12,16,24,32,48,64,96 --reset

Linux Mint:
python3.13 src/train.py datasets/GPTeacher.json --ngram-order 8 --json-chunk-size 500 --eval-samples 2 --eval-variants 2 --chunk-eval-percent 10.0 --eval-interval 10000 --profile-ingest --decoder-presence-penalty 0.4 --decoder-frequency-penalty 0.2 --context-dimensions 16,20,24,38,42,46 --reset

Lighter training:
python3.11 src/train.py datasets/GPTeacher.json --ngram-order 3 --json-chunk-size 100 --eval-samples 2 --eval-variants 2 --chunk-eval-percent 20.0 --eval-interval 1000 --profile-ingest --decoder-presence-penalty 0.3 --decoder-frequency-penalty 0.1 --context-dimensions 12,16,22 --reset
 
Ubuntu:
DEVICE=cuda python3.13 src/train.py datasets/emotion_data.json --ngram-order 4 --recursive --reset --json-chunk-size 100 --eval-samples 2 --eval-pool-size 500 --chunk-eval-percent 10.0 --eval-interval 100 --profile-ingest --decoder-presence-penalty 0.5 --decoder-frequency-penalty 0.25 --context-dimensions 6,12,24

AlmaLinux: 
python3.12 src/train.py datasets/GPTeacher.json --ngram-order 4 --reset --json-chunk-size 500 --eval-samples 3 --eval-variants 3 --chunk-eval-percent 10.0 --eval-interval 1000 --profile-ingest --decoder-presence-penalty 0.5 --decoder-frequency-penalty 0.25 --context-dimensions 6,12,24

macOS:
python3.11 src/train.py datasets/GPTeacher.json --ngram-order 4 --reset --json-chunk-size 500 --eval-samples 3 --eval-variants 3 --chunk-eval-percent 10.0 --eval-interval 1000 --profile-ingest --decoder-presence-penalty 0.5 --decoder-frequency-penalty 0.25 --context-dimensions 6,12,24

CheetahDB run: go run .

python -m spacy download en_core_web_lg
python -m spacy download en_core_web_sm

- Number of evaluation to do: --eval-interval 50

# How to execute through Ubuntu 24 WSL on Windows:
> PS C:\Users\Riccardo Cecchini> wsl.exe -d Ubuntu-24.04 -- whoami
> riccardo

### Default "next steps" prompt:
[AI_REFERENCE.md](AI_REFERENCE.md): continue with NEXT_STEPS.md implementations to do in code or later updating [NEXT_STEPS.md](NEXT_STEPS.md). 

The current OS is Windows: use Ubuntu 24 through WSL, golang available, pip packages installed on python3.11.
Example execution command on Ubuntu on Powershell: wsl.exe -d Ubuntu-24.04 -- echo example
When executing cheetah-db server and both train/smoke-trains, use screen or you get stuck. Example:
`wsl.exe -d Ubuntu-24.04 -- screen -dmS cheetahdb bash -c 'cd /mnt/c/Sources/GitHub/lmdb/cheetah-db && env CHEETAH_HEADLESS=1 ./cheetah-server-linux'`.
Remember to kill the session (`screen -ls` + `screen -wipe` or `pkill -f cheetah-server`) when you finish or re-compile the server, and even before beginning check.
Remember that you have to run in parallel cheetah-db server with other scripts, and to stop both of them before recompilation and after end of executions.
You can fallback on tmux, but for keeping alive a screen session detaching it you have to do: screen -d -m -S session_name "your_command_here".
Example: screen -d -m -S web_server "python3 -m http.server 8000"
Save and check logs in real time (every 30 seconds circa in long term cases, for preventing excessive number of calls), prevent stucks (no more outputs), infinity loops and running test no longer than 30 minutes.
