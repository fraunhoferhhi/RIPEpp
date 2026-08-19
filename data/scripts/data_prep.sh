#!/bin/bash

set -euo pipefail

ROOT_DIR="${1:-.}"

find "$ROOT_DIR" -type f \( -name "frame_data.tar.gz" -o -name "scene_points.tar.gz" \) -print0 |
while IFS= read -r -d '' archive; do
    dir="$(dirname "$archive")"

    if [[ "$archive" == */frame_data.tar.gz ]]; then
        target="$dir/frame_data"
        mkdir -p "$target"
        echo "Extracting $archive to $target"
        tar -xzf "$archive" -C "$target"
        rm -f "$archive"
    elif [[ "$archive" == */scene_points.tar.gz ]]; then
        target="$dir/scene_points"
        mkdir -p "$target"
        echo "Extracting $archive to $target"
        tar -xzf "$archive" -C "$target"
        rm -f "$archive"
    else
        echo "Unknown archive type: $archive" >&2
        exit 1
    fi
done

# create folder val inside ROOT_DIR if it does not exist
if [ ! -d "$ROOT_DIR/val" ]; then
    echo "Creating val directory inside $ROOT_DIR"
    mkdir -p "$ROOT_DIR/val"

    # move dataset_7 into val
    mv "$ROOT_DIR/dataset_7" "$ROOT_DIR/val/dataset_7"
else
    echo "val directory already exists inside $ROOT_DIR."
    # check if dataset_7 exists inside val
    if [ -d "$ROOT_DIR/val/dataset_7" ]; then
        echo "dataset_7 already exists inside val. Skipping move."
    else
        echo "Moving dataset_7 into val directory."
        mv "$ROOT_DIR/dataset_7" "$ROOT_DIR/val/dataset_7"
    fi
fi

