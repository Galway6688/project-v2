# run_predictions_rerun.py

import pandas as pd
import os
import time
from tqdm import tqdm

# 只导入 agent.py 中需要的部分
from agent import MultimodalAgent, image_to_base64

# --- 1. 配置区 ---
BASE_IMAGE_PATH = r"E:\Touch-Vision-Language-Dataset\tvl_dataset\ssvtp"
# 确保这个 test.csv 文件是完整的，以便能从中查找到 tactile 路径
TEST_CSV_PATH = 'test.csv'
# 将结果输出到一个新的文件中，避免覆盖原有结果
RAW_OUTPUT_CSV = "rerun_tactile_errors.csv"


def run_specific_predictions():
    """
    只为指定的图片列表执行 'tactile' 模式的预测。
    """
    # --- 新增：定义需要重新运行的目标图片列表 ---
    target_vision_paths = [
        "images_rgb/image_2096_rgb.jpg",
        "images_rgb/image_464_rgb.jpg",
        "images_rgb/image_4490_rgb.jpg",
        "images_rgb/image_2997_rgb.jpg",
        "images_rgb/image_4330_rgb.jpg",
        "images_rgb/image_473_rgb.jpg",
        "images_rgb/image_3177_rgb.jpg",
        "images_rgb/image_2262_rgb.jpg",
        "images_rgb/image_2187_rgb.jpg",
        "images_rgb/image_2480_rgb.jpg",
        "images_rgb/image_691_rgb.jpg",
        "images_rgb/image_906_rgb.jpg",
        "images_rgb/image_2764_rgb.jpg",
        "images_rgb/image_1391_rgb.jpg",
        "images_rgb/image_700_rgb.jpg",
        "images_rgb/image_1346_rgb.jpg",
        "images_rgb/image_1697_rgb.jpg",
        "images_rgb/image_2447_rgb.jpg",
        "images_rgb/image_3940_rgb.jpg"
    ]

    print("--- Step 1: Loading Resources ---")
    try:
        # 加载完整的 test.csv 以便进行筛选
        full_test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"Successfully loaded {TEST_CSV_PATH} with {len(full_test_df)} samples.")

        # --- 修改：根据目标列表筛选 DataFrame ---
        test_df = full_test_df[full_test_df['url'].isin(target_vision_paths)].copy()
        print(f"Filtered down to {len(test_df)} target samples to re-run.")

        if len(test_df) == 0:
            print("Warning: None of the target images were found in test.csv. Please check the paths.")
            return

    except FileNotFoundError:
        print(f"Error: {TEST_CSV_PATH} not found. Please ensure it's in the same directory.")
        return

    print("Initializing agents (0-shot baseline and 4-shot agent)...")
    baseline_agent = MultimodalAgent(num_shots=0)
    five_shot_agent = MultimodalAgent(num_shots=4)
    print("Agents initialized.\n")

    print("--- Step 2: Running Predictions for specific items in TACTILE mode ---")
    all_results = []
    # --- 修改：只运行 'tactile' 模式 ---
    mode = 'tactile'

    print(f"\n--- Predicting for MODE: {mode.upper()} ---")
    # --- 修改：遍历筛选后的 DataFrame ---
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Mode: {mode}"):
        # vision_path 仅用于记录，在 tactile mode 下不会被加载
        vision_path = os.path.join(BASE_IMAGE_PATH, row['url'].replace('/', '\\'))
        tactile_path = os.path.join(BASE_IMAGE_PATH, row['tactile'].replace('/', '\\'))
        ground_truth = str(row['caption'])

        # 在 tactile mode 下，vision_input 永远是 None
        vision_input = None
        tactile_input = None

        if os.path.exists(tactile_path):
            tactile_input = image_to_base64(tactile_path)
        else:
            print(f"Warning: Tactile image not found at {tactile_path}. Skipping.")
            continue

        try:
            baseline_pred = baseline_agent.process_request(
                "Describe tactile properties.", mode=mode,
                vision_image=vision_input, tactile_image=tactile_input
            )['response']
            time.sleep(31)  # 保持 API 访问间隔
            fiveshot_pred = five_shot_agent.process_request(
                "Describe tactile properties.", mode=mode,
                vision_image=vision_input, tactile_image=tactile_input
            )['response']

            all_results.append({
                "mode": mode,
                "vision_path": row['url'], # 仍然记录 vision_path 作为唯一标识
                "ground_truth": ground_truth,
                "baseline_output": baseline_pred,
                "five_shot_output": fiveshot_pred
            })
        except Exception as e:
            print(f"\nError on row {index} (vision_path: {row['url']}): {e}")
            all_results.append({
                "mode": mode,
                "vision_path": row['url'],
                "ground_truth": ground_truth,
                "baseline_output": "ERROR",
                "five_shot_output": "ERROR"
            })
        time.sleep(31) # 保持 API 访问间隔

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RAW_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ Re-run predictions are complete and saved to '{RAW_OUTPUT_CSV}'.")


if __name__ == '__main__':
    run_specific_predictions()