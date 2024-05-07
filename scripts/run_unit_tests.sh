#!/bin/bash
set -e

export PIXELBRAIN_PATH=$(git rev-parse --show-toplevel)
export PYTHONPATH=$PIXELBRAIN_PATH/src/:$PYTHONPATH
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    VENV_PATH="$PIXELBRAIN_PATH/venv"
    source "$VENV_PATH/bin/activate"
fi

print_help() {
    echo "Usage: $0 [--include-slow-suit] [--time-tests]"
    echo "  --include-slow-suit: Include slow tests in the test run"
    echo "  --time-tests: Time each test"
}

# Parsing command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --include-slow-suit) 
            include_slow_suit=1
            echo "Including slow suit in the test run."
            ;;
        --time-tests) 
            time_tests="--durations=0"
            echo "Timing each test."
            ;;
        -h|--help) 
            print_help
            exit 0 
            ;;
        *) 
            echo "Unknown parameter passed: $1"
            exit 1 
            ;;
    esac
    shift
done

if [[ "$include_slow_suit" == "1" ]]; then
  pytest $PIXELBRAIN_PATH/src/tests -n auto $time_tests
else
  pytest $PIXELBRAIN_PATH/src/tests -m "not slow_suit" -n auto $time_tests
fi
