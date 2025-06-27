# Get the full path to the script
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
DIR_NAME="$(basename "$(dirname "$SCRIPT_DIR")")"
IMG_NAME="$(echo "$DIR_NAME" | tr '[:upper:]' '[:lower:]')" # Convert to lowercase
USER_NAME=$(id -un)

echo "version is set to: $DIR_NAME"

docker build \
    --build-arg USER_NAME=$(id -un) \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -t $IMG_NAME:$USER_NAME .

# docker build \
#     # --no-cache \
#     --build-arg USER_NAME=$(id -un) \
#     --build-arg USER_ID=$(id -u) \
#     --build-arg GROUP_ID=$(id -g) \
#     -t $IMG_NAME:latest .