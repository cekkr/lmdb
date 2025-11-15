#!/bin/bash

# ./run_parallel.sh \
#  --command "my_script.py" \
#  --arguments "--user admin --mode prod" \
#  --python python3.9 \
#  --log-path /var/logs/my_app

# export COMMAND="my_script.py"
# export ARGUMENTS="--user admin"
# export LOG_PATH="/var/logs/my_app"
# ./run_parallel.sh

# --- Default Values ---
DEFAULT_LOG_PATH="/dev/null"
DEFAULT_PYTHON="python3"

# --- Help Function ---
print_help() {
  echo "Usage: $0 --command <COMMAND> [OPTIONS]"
  echo
  echo "Runs a Go command and a Python command in parallel detached 'screen' sessions."
  echo "This script will remain running; press Ctrl+C to exit and kill both sessions."
  echo
  echo "Required:"
  echo "  --command <COMMAND>     The Python command/script to run (e.s., 'main.py')."
  echo "                          (Can also be set via \$COMMAND)"
  echo
  echo "Options:"
  echo "  --arguments \"<ARGS>\"    Optional arguments for the Python command, in quotes."
  echo "                          (Can also be set via \$ARGUMENTS)"
  echo "  --log-path <PATH>       Directory to store log files. (Default: /dev/null)"
  echo "                          (Can also be set via \$LOG_PATH)"
  echo "  --python <EXECUTABLE>   Python executable to use. (Default: python3)"
  echo "                          (Can also be set via \$PYTHON)"
  echo "  -h, --help              Show this help message."
}

# --- Initialize from Environment or Defaults ---
# Command-line flags will override these values
SCRIPT_LOG_PATH="${LOG_PATH:-$DEFAULT_LOG_PATH}"
SCRIPT_PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
SCRIPT_COMMAND="${COMMAND:-}"
SCRIPT_ARGUMENTS="${ARGUMENTS:-}"

# --- Parse Command-Line Arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    --log-path)
      SCRIPT_LOG_PATH="$2"
      shift 2
      ;;
    --python)
      SCRIPT_PYTHON="$2"
      shift 2
      ;;
    --command)
      SCRIPT_COMMAND="$2"
      shift 2
      ;;
    --arguments)
      SCRIPT_ARGUMENTS="$2"
      shift 2
      ;;
    *)
      echo "Error: Unknown option $1" >&2
      print_help
      exit 1
      ;;
  esac
done

# --- Validate Required Arguments ---
if [[ -z "$SCRIPT_COMMAND" ]]; then
  echo "Error: --command is required." >&2
  print_help
  exit 1
fi

# --- Set up Log File Paths ---
if [[ "$SCRIPT_LOG_PATH" == "/dev/null" ]]; then
  CHEETAH_LOG_FILE="/dev/null"
  PYTHON_LOG_FILE="/dev/null"
else
  # Create directory if it doesn't exist
  mkdir -p "$SCRIPT_LOG_PATH"
  CHEETAH_LOG_FILE="$SCRIPT_LOG_PATH/cheetahdb.log"
  PYTHON_LOG_FILE="$SCRIPT_LOG_PATH/py.log"
fi

# --- Define Unique Session Names (using script PID) ---
CHEETAH_SESSION="cheetah_job_$$"
PYTHON_SESSION="python_job_$$"

# --- Cleanup Function ---
cleanup() {
  echo
  echo "SIGINT/EXIT received. Cleaning up screen sessions..."
  # Use -X quit to send the 'quit' command to the screen session
  screen -S "$CHEETAH_SESSION" -X quit >/dev/null 2>&1
  screen -S "$PYTHON_SESSION" -X quit >/dev/null 2>&1
  echo "Cleanup complete. Exiting."
}

# --- Set Trap ---
# This will call the 'cleanup' function when the script receives
# a SIGINT (Ctrl+C), SIGTERM, or EXIT signal.
trap cleanup SIGINT SIGTERM EXIT

# --- Build Commands ---
# We use 'bash -c' inside screen to reliably handle complex commands
# and redirection.
# The '&>' syntax redirects both stdout and stderr to the file.
CMD1="go run ./cheetah-db &> \"$CHEETAH_LOG_FILE\""
CMD2="$SCRIPT_PYTHON $SCRIPT_COMMAND $SCRIPT_ARGUMENTS &> \"$PYTHON_LOG_FILE\""

# --- Start Sessions ---
echo "Starting Cheetah DB session ($CHEETAH_SESSION)..."
echo "Log: $CHEETAH_LOG_FILE"
screen -d -m -S "$CHEETAH_SESSION" bash -c "$CMD1"

echo "Starting Python session ($PYTHON_SESSION)..."
echo "Log: $PYTHON_LOG_FILE"
screen -d -m -S "$PYTHON_SESSION" bash -c "$CMD2"

echo
echo "Both sessions started in detached mode."
echo "This script is now waiting. Press Ctrl+C to stop and clean up sessions."

# --- Wait Indefinitely ---
# This script needs to stay alive for the 'trap' to be active.
# When you press Ctrl+C, the 'trap' will fire the 'cleanup' function.
sleep infinity