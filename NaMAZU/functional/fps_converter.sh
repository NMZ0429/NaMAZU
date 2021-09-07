#!/bin/zsh

#read -p 'Enter the 1 path: ' 1
#read -p 'Enter frame rate: ' fps
for file in "$1"/*; do
  echo "$file"
  ffmpeg -i "$file"  -r $2 "$file"_"$2"fps.mp4
done