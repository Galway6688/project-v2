# run_predictions.py (Final corrected version: fixed True Baseline image input logic)

import pandas as pd
import os
import time
from tqdm import tqdm
import random

from agent import MultimodalAgent, image_to_base64
from langchain_core.messages import HumanMessage
from noisy_questions import NOISY_QUESTIONS

# --- 1. Configuration Section ---
BASE_IMAGE_PATH = r"E:\Touch-Vision-Language-Dataset\tvl_dataset\ssvtp"
TEST_CSV_PATH = 'test.csv'
RAW_OUTPUT_CSV = "evaluation_predictions_raw.csv"


def run_final_experiment():
    print("--- Step 1: Loading Resources ---")
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"Successfully loaded {TEST_CSV_PATH} with {len(test_df)} samples.")
    except FileNotFoundError:
        print(f"Error: {TEST_CSV_PATH} not found.")
        return

    print("Initializing agents for 3-way comparison...")
    agent_baseline = MultimodalAgent(num_shots=0)
    four_shot_agent = MultimodalAgent(num_shots=4)
    print("Agents initialized.\n")

    print("--- Step 2: Running Predictions with Randomized Noisy Questions ---")
    all_results = []
    modes_to_evaluate = ['combined', 'vision', 'tactile']

    for mode in modes_to_evaluate:
        print(f"\n--- Predicting for MODE: {mode.upper()} ---")
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Images for {mode}"):

            noisy_question = random.choice(NOISY_QUESTIONS)

            vision_path = os.path.join(BASE_IMAGE_PATH, row['url'].replace('/', '\\'))
            tactile_path = os.path.join(BASE_IMAGE_PATH, row['tactile'].replace('/', '\\'))
            ground_truth = str(row['caption'])

            vision_input, tactile_input = None, None
            if mode in ['vision', 'combined'] and os.path.exists(vision_path):
                vision_input = image_to_base64(vision_path)
            if mode in ['tactile', 'combined'] and os.path.exists(tactile_path):
                tactile_input = image_to_base64(tactile_path)

            try:
                # --- Core modification: fix True Baseline prompt construction logic ---
                simple_prompt_text = f"Based on the image(s), list the key tactile attributes as a comma-separated list. My question is: {noisy_question}"
                true_baseline_messages = []

                # Create first message containing instructions
                content_part1 = [{"type": "text", "text": simple_prompt_text}]

                # Based on mode, decide which image to attach in the first message
                if mode == 'vision':
                    if vision_input: content_part1.append({"type": "image_url", "image_url": {"url": vision_input}})
                elif mode == 'tactile':
                    if tactile_input: content_part1.append({"type": "image_url", "image_url": {"url": tactile_input}})
                elif mode == 'combined':
                    if vision_input: content_part1.append({"type": "image_url", "image_url": {"url": vision_input}})

                true_baseline_messages.append(HumanMessage(content=content_part1))

                # Only in combined mode, need to create second message to send second image
                if mode == 'combined' and tactile_input:
                    content_part2 = [{"type": "text", "text": "And here is the corresponding tactile data:"}]
                    content_part2.append({"type": "image_url", "image_url": {"url": tactile_input}})
                    true_baseline_messages.append(HumanMessage(content=content_part2))

                true_baseline_pred = agent_baseline.llm.invoke(true_baseline_messages).content.strip()

                # --- (Agent Baseline and 4-Shot Agent call logic unchanged) ---
                agent_baseline_pred = \
                agent_baseline.process_request(question=noisy_question, mode=mode, vision_image=vision_input,
                                               tactile_image=tactile_input)['response']
                four_shot_pred = \
                four_shot_agent.process_request(question=noisy_question, mode=mode, vision_image=vision_input,
                                                tactile_image=tactile_input)['response']

                all_results.append({
                    "noisy_question": noisy_question, "mode": mode, "vision_path": row['url'],
                    "ground_truth": ground_truth, "true_baseline_output": true_baseline_pred,
                    "agent_baseline_output": agent_baseline_pred, "four_shot_output": four_shot_pred
                })
            except Exception as e:
                print(f"\nError on row {index}, mode {mode}, question '{noisy_question}': {e}")

            time.sleep(61)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RAW_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ All predictions using randomized noisy inputs are complete and saved to '{RAW_OUTPUT_CSV}'.")


if __name__ == '__main__':
    run_final_experiment()