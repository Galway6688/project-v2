# run_predictions.py

import pandas as pd
import os
import time
from tqdm import tqdm

# 只导入 agent.py 中需要的部分
from agent import MultimodalAgent, image_to_base64

# --- 1. 配置区 ---
BASE_IMAGE_PATH = r"E:\Touch-Vision-Language-Dataset\tvl_dataset\ssvtp"
TEST_CSV_PATH = 'test5.csv'  # 确保使用包含'tactile'列的完整测试文件
RAW_OUTPUT_CSV = "evaluation_predictions_raw.csv"


def run_predictions_only():
    """
    只执行预测步骤，并将所有模型的原始输出结果保存到CSV。
    """
    print("--- Step 1: Loading Resources ---")
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"Successfully loaded {TEST_CSV_PATH} with {len(test_df)} samples.")
    except FileNotFoundError:
        print(f"Error: {TEST_CSV_PATH} not found. Please ensure it's in the same directory.")
        return

    print("Initializing agents (0-shot baseline and 4-shot agent)...")  # <-- 修改日志文本
    baseline_agent = MultimodalAgent(num_shots=0)
    five_shot_agent = MultimodalAgent(num_shots=4)  # <-- 将这里的 5 改为 4
    print("Agents initialized.\n")

    print("--- Step 2: Running Predictions for all modes ---")
    all_results = []
    modes_to_evaluate = ['combined', 'vision', 'tactile']

    for mode in modes_to_evaluate:
        print(f"\n--- Predicting for MODE: {mode.upper()} ---")
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Mode: {mode}"):
            vision_path = os.path.join(BASE_IMAGE_PATH, row['url'].replace('/', '\\'))
            tactile_path = os.path.join(BASE_IMAGE_PATH, row['tactile'].replace('/', '\\'))
            ground_truth = str(row['caption'])

            vision_input, tactile_input = None, None
            if mode in ['vision', 'combined'] and os.path.exists(vision_path):
                vision_input = image_to_base64(vision_path)
            if mode in ['tactile', 'combined'] and os.path.exists(tactile_path):
                tactile_input = image_to_base64(tactile_path)

            try:
                baseline_pred = baseline_agent.process_request(
                    "Describe tactile properties.", mode=mode,
                    vision_image=vision_input, tactile_image=tactile_input
                )['response']

                fiveshot_pred = five_shot_agent.process_request(
                    "Describe tactile properties.", mode=mode,
                    vision_image=vision_input, tactile_image=tactile_input
                )['response']

                all_results.append({
                    "mode": mode,
                    "vision_path": row['url'],
                    "ground_truth": ground_truth,
                    "baseline_output": baseline_pred,
                    "five_shot_output": fiveshot_pred
                })
            except Exception as e:
                print(f"\nError on row {index}, mode {mode}: {e}")
                all_results.append({
                    "mode": mode,
                    "vision_path": row['url'],
                    "ground_truth": ground_truth,
                    "baseline_output": "ERROR",
                    "five_shot_output": "ERROR"
                })



    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RAW_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ All predictions are complete and saved to '{RAW_OUTPUT_CSV}'. You can now run the analysis script.")


if __name__ == '__main__':
    run_predictions_only()