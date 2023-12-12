#!/bin/bash

# keyの複数の値を定義
folder="./out/evogym_poet/default/niche"

# keysの各要素に対してコマンドを実行
find "$folder" -maxdepth 1 -type d | while read -r subfolder
do
    xvfb-run python make_figures_ppo.py default "$(basename "$subfolder")"
done