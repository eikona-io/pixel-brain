#!/bin/bash
export PIXELBRAIN_PATH=$(git rev-parse --show-toplevel)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    VENV_PATH="$PIXELBRAIN_PATH/venv"
    source "$VENV_PATH/bin/activate"
fi


# Parsing command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --include-slow-suit) include_slow_suit=1 ;;
        --time-tests) time_tests="--durations=0" ;;
        -h|--help) echo "Usage: $0 [--include-slow-suit] [--time-tests]"; echo "  --include-slow-suit: Include slow tests in the test run"; echo "  --time-tests: Time each test"; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ "$include_slow_suit" == "1" ]]; then
  pytest $PIXELBRAIN_PATH/src/tests $time_tests
else
  pytest $PIXELBRAIN_PATH/src/tests -m "not slow_suit" $time_tests
fi
