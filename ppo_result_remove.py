import csv
import glob
import os
import shutil

# ディレクトリppo_result内の1~5までのフォルダを取得
files = glob.glob("./out/evogym_poet/default/niche/*")
folders = [f"./out/evogym_poet/default/ppo_result/{i}" for i in range(1, 6)]

# 各フォルダのhistory.csvからmax(reward)を取得
for folder in files:
    rewards = []
    for i in range(1, 6):
        with open(f"{folder}/ppo_result/{i}/history.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            numbers = []
            for row in reader:
                numbers.append(float(row[1]))
        rewards.append(max(numbers))

    # 最大のrewardを取得
    max_reward = max(rewards)
    print(max_reward)
    max_index = rewards.index(max_reward)

    # 最大のreward以外のフォルダを削除
    for i in range(5):
        if rewards[i] != max_reward:
            shutil.rmtree(f"{folder}/ppo_result/{i+1}")
