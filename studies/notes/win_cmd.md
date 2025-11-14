DBSLM_BACKEND=cheetah-db python3.11 src/train.py datasets/emotion_data.json --ngram-order 5 --recursive --reset --json-chunk-size 50 --eval-samples 2 --eval-pool-size 5000 --chunk-eval-percent 10.0 --eval-interval 5000 --profile-ingest --decoder-presence-penalty 0.5 --decoder-frequency-penalty 0.25 --context-dimensions 8,6,6

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

