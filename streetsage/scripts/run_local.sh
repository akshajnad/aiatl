#!/bin/bash

# StreetSage - Convenience launcher script

set -e

cd "$(dirname "$0")/.."

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Warning: Virtual environment not activated"
    echo "Run: source .venv/bin/activate"
    echo ""
fi

# Check if .env exists
if [[ ! -f .env ]]; then
    echo "Error: .env file not found"
    echo "Copy .env.example to .env and fill in your credentials"
    exit 1
fi

# Default: run with laptop webcam
echo "Starting StreetSage..."
echo "Press Ctrl+C to stop"
echo ""

# Parse arguments
VIZ_FLAG=""
VOICE_FLAG=""
SOURCE_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --viz)
            VIZ_FLAG="--viz"
            shift
            ;;
        --voice-qa)
            VOICE_FLAG="--voice-qa"
            shift
            ;;
        --source)
            SOURCE_FLAG="--source $2"
            shift 2
            ;;
        --init-db)
            python app.py --init-db
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--viz] [--voice-qa] [--source SOURCE] [--init-db]"
            exit 1
            ;;
    esac
done

# Run the app
exec python app.py $SOURCE_FLAG $VIZ_FLAG $VOICE_FLAG
