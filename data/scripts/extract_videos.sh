#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-.}"

while IFS= read -r -d '' video; do
    dir="$(dirname "$video")"
    name="$(basename "$video" .mp4)"
    outdir_left="$dir/${name}_frames_left"
    outdir_right="$dir/${name}_frames_right"

    # Skip if both output folders exist and are non-empty
    if [[ -d "$outdir_left" ]] && [ "$(find "$outdir_left" -mindepth 1 -print -quit)" ] && \
       [[ -d "$outdir_right" ]] && [ "$(find "$outdir_right" -mindepth 1 -print -quit)" ]; then
        echo "Skipping $video: output folders already exist and are non-empty"
        continue
    fi

    mkdir -p "$outdir_left" "$outdir_right"
    
    ffmpeg -nostdin -i "$video" \
        -filter_complex "[0:v]crop=iw:ih/2:0:0[left];[0:v]crop=iw:ih/2:0:ih/2[right]" \
        -map "[left]" -start_number 0 "$outdir_left/frame_%06d.png" \
        -map "[right]" -start_number 0 "$outdir_right/frame_%06d.png"

done < <(find "$ROOT_DIR" -type f -name "*.mp4" -print0)