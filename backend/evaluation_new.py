# evaluation.py (最终优化版：优先使用GPU，并保存中间进度)

import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import nltk

# 评测指标库
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 您的 Agent 脚本
# 确保 agent.py 和此脚本在同一目录下
from agent import MultimodalAgent, image_to_base64

# --- 1. 配置区 ---
# 请在此处确认您的文件路径
BASE_IMAGE_PATH = r"E:\Touch-Vision-Language-Dataset\tvl_dataset\ssvtp"
TEST_CSV_PATH = 'test.csv'  # 确保使用包含'tactile'列的、修正后的完整测试文件


# --- 2. 辅助函数 ---
def calculate_keyword_recall(candidate_str: str, reference_str: str) -> float:
    """计算生成结果中包含了多少标准答案里的关键词"""
    if not isinstance(candidate_str, str) or not isinstance(reference_str, str):
        return 0.0
    ref_keywords = set([kw.strip() for kw in reference_str.split(',') if kw.strip()])
    cand_keywords = set([kw.strip() for kw in candidate_str.split(',') if kw.strip()])

    if not ref_keywords:
        return 1.0 if not cand_keywords else 0.0

    matched_keywords = ref_keywords.intersection(cand_keywords)
    return len(matched_keywords) / len(ref_keywords)


# --- 3. 主评测函数 ---
def run_full_evaluation():
    """
    执行专为毕业设计报告设计的、完整且详细的评测流程。
    """
    # 步骤 3.1: 加载所有资源
    print("--- Step 1: Loading Resources ---")

    # 自动检测并设置计算设备 (GPU优先)
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"✅ Found available GPU. Using device: {device}")
    else:
        device = 'cpu'
        print("⚠️ No GPU found. Using device: cpu. This may be slow or cause memory issues.")

    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"Successfully loaded {TEST_CSV_PATH} with {len(test_df)} samples.")
    except FileNotFoundError:
        print(f"Error: {TEST_CSV_PATH} not found. Please ensure the file is in the correct directory.")
        return

    print("Initializing agents (0-shot baseline and 5-shot agent)...")
    baseline_agent = MultimodalAgent(num_shots=0)
    five_shot_agent = MultimodalAgent(num_shots=5)
    print("Agents initialized.")

    print("Loading evaluation models (BERTScorer, NLTK, ROUGE)...")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' model...")
        nltk.download('punkt')

    # 将检测到的 device 传递给 BERTScorer，强制其使用GPU
    bert_scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en', device=device)
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction()
    print("Evaluation models loaded and set to run on designated device.\n")

    # 步骤 3.2: 运行预测
    print("--- Step 2: Running Predictions for all modes ---")
    all_results = []
    modes_to_evaluate = ['combined', 'vision', 'tactile']

    for mode in modes_to_evaluate:
        print(f"\n--- Evaluating MODE: {mode.upper()} ---")
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
                baseline_pred = \
                    baseline_agent.process_request("Describe tactile properties.", mode=mode, vision_image=vision_input,
                                                   tactile_image=tactile_input)['response']
                fiveshot_pred = \
                    five_shot_agent.process_request("Describe tactile properties.", mode=mode,
                                                    vision_image=vision_input,
                                                    tactile_image=tactile_input)['response']

                all_results.append({
                    "mode": mode,
                    "vision_path": row['url'],
                    "ground_truth": ground_truth,
                    "baseline_output": baseline_pred,
                    "five_shot_output": fiveshot_pred
                })
            except Exception as e:
                print(f"Error on row {index}, mode {mode}: {e}")

            # 如果遇到 API 速率限制错误，可以取消下面这行的注释
            # time.sleep(21)

    results_df = pd.DataFrame(all_results)

    # 在计算指标前，立即保存原始预测结果，确保进度不丢失
    raw_output_filename = "evaluation_predictions_raw.csv"
    results_df.to_csv(raw_output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ Raw predictions saved to '{raw_output_filename}' as a backup.")

    # 步骤 3.3: 计算各项详细指标
    print("\n--- Step 3: Calculating Detailed Metrics for All Samples (on GPU) ---")

    # 因为使用GPU，内存通常足够，一般不再需要batch_size。如果GPU显存依然不足，可以再加回来。
    P_base, R_base, F1_base = bert_scorer.score(results_df['baseline_output'].tolist(),
                                                results_df['ground_truth'].tolist())
    P_5shot, R_5shot, F1_5shot = bert_scorer.score(results_df['five_shot_output'].tolist(),
                                                   results_df['ground_truth'].tolist())
    results_df['baseline_bert_f1'] = F1_base.numpy()
    results_df['five_shot_bert_f1'] = F1_5shot.numpy()

    # BLEU, ROUGE, Keyword Recall
    other_metrics = []
    for index, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Calculating other metrics"):
        gt_list = str(row['ground_truth']).split()
        base_list = str(row['baseline_output']).split()
        fiveshot_list = str(row['five_shot_output']).split()

        rouge_base = rouge.score(row['ground_truth'], row['baseline_output'])
        rouge_5shot = rouge.score(row['ground_truth'], row['five_shot_output'])

        other_metrics.append({
            'baseline_bleu': sentence_bleu([gt_list], base_list, smoothing_function=chencherry.method1),
            'five_shot_bleu': sentence_bleu([gt_list], fiveshot_list, smoothing_function=chencherry.method1),
            'baseline_rougeL_f1': rouge_base['rougeL'].fmeasure,
            'five_shot_rougeL_f1': rouge_5shot['rougeL'].fmeasure,
            'baseline_kw_recall': calculate_keyword_recall(row['baseline_output'], row['ground_truth']),
            'five_shot_kw_recall': calculate_keyword_recall(row['five_shot_output'], row['ground_truth']),
        })

    other_metrics_df = pd.DataFrame(other_metrics, index=results_df.index)
    results_df = pd.concat([results_df, other_metrics_df], axis=1)

    # 计算分数提升
    results_df['bert_f1_improvement'] = results_df['five_shot_bert_f1'] - results_df['baseline_bert_f1']

    output_filename = "evaluation_results_detailed.csv"
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"Detailed results for all modes saved to {output_filename}")

    # 步骤 3.4: 生成统计总结和可视化
    print("\n--- Step 4: Generating Statistical Summary & Visualization ---")

    for mode in modes_to_evaluate:
        mode_df = results_df[results_df['mode'] == mode]
        if mode_df.empty:
            continue

        print(f"\n----- Statistical Summary for MODE: {mode.upper()} -----")

        stats = {
            "Metric": ["BERT-F1", "BLEU", "ROUGE-L-F1", "KW-Recall"],
            "Baseline Mean": [
                mode_df['baseline_bert_f1'].mean(), mode_df['baseline_bleu'].mean(),
                mode_df['baseline_rougeL_f1'].mean(), mode_df['baseline_kw_recall'].mean()
            ],
            "5-Shot Mean": [
                mode_df['five_shot_bert_f1'].mean(), mode_df['five_shot_bleu'].mean(),
                mode_df['five_shot_rougeL_f1'].mean(), mode_df['five_shot_kw_recall'].mean()
            ],
            "5-Shot Std Dev": [
                mode_df['five_shot_bert_f1'].std(), mode_df['five_shot_bleu'].std(),
                mode_df['five_shot_rougeL_f1'].std(), mode_df['five_shot_kw_recall'].std()
            ]
        }
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))

        # 可视化
        plt.figure(figsize=(8, 6))
        plt.boxplot([mode_df['baseline_bert_f1'].dropna(), mode_df['five_shot_bert_f1'].dropna()],
                    labels=['Zero-Shot Baseline', '5-Shot Agent'])
        plt.title(f'BERT F1 Score Distribution for {mode.upper()} Mode')
        plt.ylabel('BERT F1 Score')
        plt.grid(True, linestyle='--', alpha=0.6)

        plot_filename = f"score_distribution_{mode}.png"
        plt.savefig(plot_filename)
        print(f"Score distribution plot saved to {plot_filename}")
        plt.close()

    print("\nAll tasks complete. Your detailed analysis is ready!")


# --- 4. 脚本入口 ---
if __name__ == '__main__':
    run_full_evaluation()