# run_predictions_combined_only.py

import pandas as pd
import os
import time
from tqdm import tqdm

# 确保 agent.py 和此脚本在同一目录下
from agent import MultimodalAgent, image_to_base64

# --- 1. 配置区 ---
BASE_IMAGE_PATH = r"E:\Touch-Vision-Language-Dataset\tvl_dataset\ssvtp"
TEST_CSV_PATH = 'test.csv'  # 您的完整测试集描述文件
ORIGINAL_RAW_CSV = "evaluation_predictions_raw.csv"  # 上次运行生成的、需要被补齐的文件
TEMP_COMBINED_CSV = "combined_results_temp.csv"  # 用于临时存储combined结果的文件


def run_combined_and_merge():
    """
    只执行 'combined' 模式的预测，然后将结果合并回主要的原始预测文件中。
    """
    # --- Part 1: 只运行 Combined 模式的预测 ---
    print("--- Part 1: Running Predictions for COMBINED mode only ---")

    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Main test file '{TEST_CSV_PATH}' not found.")
        return

    print("Initializing agents...")
    baseline_agent = MultimodalAgent(num_shots=0)
    five_shot_agent = MultimodalAgent(num_shots=5)
    print("Agents initialized.\n")

    combined_results = []
    mode = 'combined'  # 只跑 combined 模式

    print(f"--- Predicting for MODE: {mode.upper()} ---")
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Mode: {mode}"):
        vision_path = os.path.join(BASE_IMAGE_PATH, row['url'].replace('/', '\\'))
        tactile_path = os.path.join(BASE_IMAGE_PATH, row['tactile'].replace('/', '\\'))
        ground_truth = str(row['caption'])

        vision_input, tactile_input = None, None
        if os.path.exists(vision_path):
            vision_input = image_to_base64(vision_path)
        if os.path.exists(tactile_path):
            tactile_input = image_to_base64(tactile_path)

        try:
            baseline_pred = \
            baseline_agent.process_request("...", mode=mode, vision_image=vision_input, tactile_image=tactile_input)[
                'response']
            fiveshot_pred = \
            five_shot_agent.process_request("...", mode=mode, vision_image=vision_input, tactile_image=tactile_input)[
                'response']

            combined_results.append({
                "mode": mode, "vision_path": row['url'], "ground_truth": ground_truth,
                "baseline_output": baseline_pred, "five_shot_output": fiveshot_pred
            })
        except Exception as e:
            print(f"\nError on row {index}, mode {mode}: {e}")
            combined_results.append({
                "mode": mode, "vision_path": row['url'], "ground_truth": ground_truth,
                "baseline_output": "ERROR", "five_shot_output": "ERROR"
            })
        time.sleep(21)  # 避免速率超限

    temp_df = pd.DataFrame(combined_results)
    temp_df.to_csv(TEMP_COMBINED_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ '{mode.upper()}' mode predictions are complete and saved to temporary file '{TEMP_COMBINED_CSV}'.")

    # --- Part 2: 合并结果到主文件中 ---
    print(f"\n--- Part 2: Merging new results into '{ORIGINAL_RAW_CSV}' ---")

    try:
        original_df = pd.read_csv(ORIGINAL_RAW_CSV)
        new_combined_df = pd.read_csv(TEMP_COMBINED_CSV)

        # 1. 从原始结果中，筛选出所有非 'combined' 模式的行
        other_modes_df = original_df[original_df['mode'] != 'combined']

        # 2. 将筛选出的旧结果和新的 'combined' 结果合并
        final_df = pd.concat([other_modes_df, new_combined_df], ignore_index=True)

        # 3. 按原始顺序排序（如果需要的话）
        final_df = final_df.sort_values(by=['vision_path', 'mode']).reset_index(drop=True)

        # 4. 覆盖保存回主文件
        final_df.to_csv(ORIGINAL_RAW_CSV, index=False, encoding='utf-8-sig')

        print(f"✅ Successfully merged results. '{ORIGINAL_RAW_CSV}' is now complete and updated.")

    except FileNotFoundError:
        print(f"Error: Could not find '{ORIGINAL_RAW_CSV}'. Saving new results as is.")
        temp_df.rename(columns={}, inplace=False).to_csv(ORIGINAL_RAW_CSV, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"An error occurred during merging: {e}")
    finally:
        # 5. 删除临时文件
        if os.path.exists(TEMP_COMBINED_CSV):
            os.remove(TEMP_COMBINED_CSV)
            print(f"Temporary file '{TEMP_COMBINED_CSV}' has been deleted.")


if __name__ == '__main__':
    run_combined_and_merge()