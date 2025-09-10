# run_predictions_9shot.py

import pandas as pd
import os
import time
from tqdm import tqdm

# Only import the necessary parts from agent.py
from agent import MultimodalAgent, image_to_base64

# --- 1. Configuration Section ---
BASE_IMAGE_PATH = r"E:\Touch-Vision-Language-Dataset\tvl_dataset\ssvtp"
TEST_CSV_PATH = 'test.csv'  # Ensure using complete test file containing 'tactile' column
RAW_OUTPUT_CSV = "evaluation_predictions_9shot.csv"


def run_predictions_9shot():
    """
    Execute 9-shot predictions for tactile and vision modes only.
    Compares 0-shot baseline with 9-shot agent performance.
    """
    print("--- Step 1: Loading Resources ---")
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"Successfully loaded {TEST_CSV_PATH} with {len(test_df)} samples.")
    except FileNotFoundError:
        print(f"Error: {TEST_CSV_PATH} not found. Please ensure it's in the same directory.")
        return

    print("Initializing agents (0-shot baseline and 9-shot agent)...")
    baseline_agent = MultimodalAgent(num_shots=0)
    nine_shot_agent = MultimodalAgent(num_shots=9)
    print("Agents initialized.\n")

    print("--- Step 2: Running 9-Shot Predictions for tactile and vision modes ---")
    all_results = []
    modes_to_evaluate = ['tactile', 'vision']  # Only tactile and vision, no combined

    for mode in modes_to_evaluate:
        print(f"\n--- Predicting for MODE: {mode.upper()} ---")
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Mode: {mode}"):
            vision_path = os.path.join(BASE_IMAGE_PATH, row['url'].replace('/', '\\'))
            tactile_path = os.path.join(BASE_IMAGE_PATH, row['tactile'].replace('/', '\\'))
            ground_truth = str(row['caption'])

            vision_input, tactile_input = None, None
            
            # Load images based on mode
            if mode == 'vision' and os.path.exists(vision_path):
                vision_input = image_to_base64(vision_path)
            elif mode == 'tactile' and os.path.exists(tactile_path):
                tactile_input = image_to_base64(tactile_path)

            # Skip if required image is missing
            if (mode == 'vision' and not vision_input) or (mode == 'tactile' and not tactile_input):
                print(f"Warning: Missing image for {mode} mode, row {index}")
                continue

            try:
                # Get predictions from both agents
                baseline_pred = baseline_agent.process_request(
                    "Describe tactile properties.", mode=mode,
                    vision_image=vision_input, tactile_image=tactile_input
                )['response']

                nine_shot_pred = nine_shot_agent.process_request(
                    "Describe tactile properties.", mode=mode,
                    vision_image=vision_input, tactile_image=tactile_input
                )['response']

                all_results.append({
                    "mode": mode,
                    "vision_path": row['url'],
                    "ground_truth": ground_truth,
                    "baseline_output": baseline_pred,
                    "nine_shot_output": nine_shot_pred
                })
                
            except Exception as e:
                print(f"\nError on row {index}, mode {mode}: {e}")
                all_results.append({
                    "mode": mode,
                    "vision_path": row['url'],
                    "ground_truth": ground_truth,
                    "baseline_output": "ERROR",
                    "nine_shot_output": "ERROR"
                })

            # Add delay to avoid API rate limiting
            time.sleep(1)  # Adjust as needed based on API limits

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RAW_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ All 9-shot predictions are complete and saved to '{RAW_OUTPUT_CSV}'.")
    print(f"Total predictions: {len(results_df)}")
    print(f"Tactile mode predictions: {len(results_df[results_df['mode'] == 'tactile'])}")
    print(f"Vision mode predictions: {len(results_df[results_df['mode'] == 'vision'])}")
    print("You can now run the analysis script.")


if __name__ == '__main__':
    run_predictions_9shot()
