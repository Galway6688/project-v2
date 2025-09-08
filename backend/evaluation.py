import pandas as pd
import os
from bert_score import BERTScorer  # <-- 修改点：导入 BERTScorer 类
from tqdm import tqdm
import torch
import time

from agent import MultimodalAgent, image_to_base64

BASE_IMAGE_PATH = r"E:\Touch-Vision-Language-Dataset\tvl_dataset\ssvtp"


def run_evaluation_for_mode(mode_to_test: str, test_df: pd.DataFrame, baseline_agent: MultimodalAgent,
                            five_shot_agent: MultimodalAgent, scorer: BERTScorer):
    """
    Execute evaluation process for a specified single mode, using the loaded scorer.
    """
    print(f"\n--- Starting Evaluation for MODE: {mode_to_test.upper()} ---")

    results = []

    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {mode_to_test}"):
        # ... (This loop logic is exactly the same as before, no changes needed) ...
        vision_image_path = os.path.join(BASE_IMAGE_PATH, row['url'].replace('/', '\\'))
        tactile_image_path = os.path.join(BASE_IMAGE_PATH, row['tactile'].replace('/', '\\'))
        ground_truth = str(row['caption'])

        vision_input, tactile_input = None, None
        if mode_to_test in ['vision', 'combined']:
            if os.path.exists(vision_image_path):
                vision_input = image_to_base64(vision_image_path)
        if mode_to_test in ['tactile', 'combined']:
            if os.path.exists(tactile_image_path):
                tactile_input = image_to_base64(tactile_image_path)

        if (mode_to_test == 'vision' and not vision_input) or \
                (mode_to_test == 'tactile' and not tactile_input) or \
                (mode_to_test == 'combined' and (not vision_input or not tactile_input)):
            continue

        try:
            baseline_response = baseline_agent.process_request(
                question="Describe the tactile properties.", mode=mode_to_test,
                vision_image=vision_input, tactile_image=tactile_input
            )['response']

            five_shot_response = five_shot_agent.process_request(
                question="Describe the tactile properties.", mode=mode_to_test,
                vision_image=vision_input, tactile_image=tactile_input
            )['response']

            results.append({
                "vision_path": row['url'], "ground_truth": ground_truth,
                "baseline_output": baseline_response, "five_shot_output": five_shot_response
            })
        except Exception as e:
            print(f"Error processing row {index} for mode '{mode_to_test}': {e}")
            results.append({
                "vision_path": row['url'], "ground_truth": ground_truth,
                "baseline_output": "ERROR", "five_shot_output": "ERROR"
            })

        # Add the delay here to avoid rate limiting
        time.sleep(21)

    results_df = pd.DataFrame(results)
    output_filename = f"evaluation_results_{mode_to_test}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Evaluation for mode '{mode_to_test}' complete. Results saved to {output_filename}")

    valid_results_df = results_df[results_df['baseline_output'] != 'ERROR']
    if len(valid_results_df) > 0:
        candidates_baseline = valid_results_df['baseline_output'].tolist()
        candidates_five_shot = valid_results_df['five_shot_output'].tolist()
        references = valid_results_df['ground_truth'].tolist()

        print(f"Calculating BERTScore for mode '{mode_to_test}' using pre-loaded model...")
        # <-- Modification: use scorer.score() method instead of score() function -->
        P_base, R_base, F1_base = scorer.score(candidates_baseline, references)
        P_5shot, R_5shot, F1_5shot = scorer.score(candidates_five_shot, references)

        print(f"--- Results for MODE: {mode_to_test.upper()} (BERTScore F1) ---")
        print(f"Zero-Shot Baseline Agent | Average F1: {F1_base.mean():.4f}")
        print(f"5-Shot Agent             | Average F1: {F1_5shot.mean():.4f}")
        print("-----------------------------------------------------")
    else:
        print(f"No valid results to score for mode '{mode_to_test}'.")


def main_evaluation():
    """
    Main function that loads all resources (including judge models), then starts evaluation.
    """
    try:
        test_df = pd.read_csv('test.csv')
        print(f"Successfully loaded test.csv with {len(test_df)} samples.")
    except FileNotFoundError:
        print("Error: test.csv not found. Make sure it's in the same directory as this script.")
        return

    print("Initializing agents...")
    baseline_agent = MultimodalAgent(num_shots=0)
    five_shot_agent = MultimodalAgent(num_shots=5)
    print("Agents initialized.")

    # <-- New section: load (or download) judge model once here -->
    print("\nLoading/Downloading BERTScore model (This is a one-time process)...")
    scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en')
    print("BERTScore model loaded successfully.\n")

    modes_to_evaluate = ['combined', 'vision', 'tactile']

    for mode in modes_to_evaluate:
        # <-- Modification: pass the loaded scorer to the evaluation function -->
        run_evaluation_for_mode(mode, test_df, baseline_agent, five_shot_agent, scorer)

    print("\nAll evaluation modes are complete.")


if __name__ == '__main__':
    main_evaluation()