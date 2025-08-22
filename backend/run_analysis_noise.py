# run_analysis.py (最终、最全功能版)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import nltk
from tqdm import tqdm

# 评测指标库
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# --- 1. 配置区 ---
RAW_PREDICTIONS_CSV = "evaluation_predictions_raw.csv"
FINAL_RESULTS_CSV = "evaluation_results_detailed.csv"


# --- 2. 辅助函数 ---
def calculate_keyword_recall(candidate_str: str, reference_str: str) -> float:
    """计算生成结果中包含了多少标准答案里的关键词"""
    if not isinstance(candidate_str, str) or not isinstance(reference_str, str): return 0.0
    ref_keywords = set([kw.strip() for kw in reference_str.split(',') if kw.strip()])
    cand_keywords = set([kw.strip() for kw in candidate_str.split(',') if kw.strip()])
    if not ref_keywords: return 1.0 if not cand_keywords else 0.0
    matched_keywords = ref_keywords.intersection(cand_keywords)
    return len(matched_keywords) / len(ref_keywords)


# --- 3. 主分析函数 ---
def run_analysis_from_file():
    """
    从原始预测文件中读取数据，并执行所有指标计算、统计分析和可视化。
    """
    print("--- Step 1: Loading Resources for Analysis ---")
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"✅ Found available GPU. Using device: {device}")
    else:
        device = 'cpu'
        print("⚠️ No GPU found. Using device: cpu.")

    try:
        results_df = pd.read_csv(RAW_PREDICTIONS_CSV)
        print(f"Successfully loaded raw predictions from '{RAW_PREDICTIONS_CSV}'.")
    except FileNotFoundError:
        print(f"Error: '{RAW_PREDICTIONS_CSV}' not found. Please run 'run_predictions.py' first.")
        return

    print("Loading evaluation models...")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    bert_scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en', device=device)
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction()
    print("Evaluation models loaded.\n")

    print("--- Step 2: Calculating Detailed Metrics ---")

    batch_size = 1 if device == 'cpu' else 32

    # 过滤掉出错或为空的行
    valid_results_df = results_df.dropna(
        subset=['true_baseline_output', 'agent_baseline_output', 'four_shot_output']).copy()
    valid_results_df = valid_results_df[valid_results_df['true_baseline_output'] != 'ERROR'].copy()

    if not valid_results_df.empty:
        # 为三组输出分别计算 BERTScore
        _, _, F1_true_base = bert_scorer.score(valid_results_df['true_baseline_output'].tolist(),
                                               valid_results_df['ground_truth'].tolist(), batch_size=batch_size)
        _, _, F1_agent_base = bert_scorer.score(valid_results_df['agent_baseline_output'].tolist(),
                                                valid_results_df['ground_truth'].tolist(), batch_size=batch_size)
        _, _, F1_4shot = bert_scorer.score(valid_results_df['four_shot_output'].tolist(),
                                           valid_results_df['ground_truth'].tolist(), batch_size=batch_size)

        valid_results_df['true_baseline_bert_f1'] = F1_true_base.numpy()
        valid_results_df['agent_baseline_bert_f1'] = F1_agent_base.numpy()
        valid_results_df['four_shot_bert_f1'] = F1_4shot.numpy()

        # --- 核心改动点：恢复为所有三组模型计算所有详细指标 ---
        other_metrics = []
        for index, row in tqdm(valid_results_df.iterrows(), total=len(valid_results_df),
                               desc="Calculating other metrics"):
            gt_list = str(row['ground_truth']).split()
            true_base_list = str(row['true_baseline_output']).split()
            agent_base_list = str(row['agent_baseline_output']).split()
            four_shot_list = str(row['four_shot_output']).split()

            rouge_true_base = rouge.score(row['ground_truth'], row['true_baseline_output'])
            rouge_agent_base = rouge.score(row['ground_truth'], row['agent_baseline_output'])
            rouge_4shot = rouge.score(row['ground_truth'], row['four_shot_output'])

            other_metrics.append({
                'true_baseline_bleu': sentence_bleu([gt_list], true_base_list, smoothing_function=chencherry.method1),
                'agent_baseline_bleu': sentence_bleu([gt_list], agent_base_list, smoothing_function=chencherry.method1),
                'four_shot_bleu': sentence_bleu([gt_list], four_shot_list, smoothing_function=chencherry.method1),

                'true_baseline_rougeL_f1': rouge_true_base['rougeL'].fmeasure,
                'agent_baseline_rougeL_f1': rouge_agent_base['rougeL'].fmeasure,
                'four_shot_rougeL_f1': rouge_4shot['rougeL'].fmeasure,

                'true_baseline_kw_recall': calculate_keyword_recall(row['true_baseline_output'], row['ground_truth']),
                'agent_baseline_kw_recall': calculate_keyword_recall(row['agent_baseline_output'], row['ground_truth']),
                'four_shot_kw_recall': calculate_keyword_recall(row['four_shot_output'], row['ground_truth']),
            })

        other_metrics_df = pd.DataFrame(other_metrics, index=valid_results_df.index)
        results_df = valid_results_df.merge(other_metrics_df, left_index=True, right_index=True, how="left")

        # 保存包含所有指标的详细结果
        results_df.to_csv(FINAL_RESULTS_CSV, index=False, encoding='utf-8-sig')
        print(f"Detailed results saved to '{FINAL_RESULTS_CSV}'")
    else:
        print("No valid predictions found in the raw file to analyze.")
        return

    print("\n--- Step 3: Generating Statistical Summary & Visualization ---")
    modes_to_evaluate = ['combined', 'vision', 'tactile']
    for mode in modes_to_evaluate:
        mode_df = results_df[results_df['mode'] == mode]
        if mode_df.empty: continue
        print(f"\n----- Statistical Summary for MODE: {mode.upper()} -----")

        # --- 核心改动点：恢复完整的统计表格 ---
        stats = {
            "Metric": ["BERT-F1", "BLEU", "ROUGE-L-F1", "KW-Recall"],
            "True Baseline Mean": [
                mode_df['true_baseline_bert_f1'].mean(), mode_df['true_baseline_bleu'].mean(),
                mode_df['true_baseline_rougeL_f1'].mean(), mode_df['true_baseline_kw_recall'].mean()
            ],
            "Agent Baseline (0-Shot) Mean": [
                mode_df['agent_baseline_bert_f1'].mean(), mode_df['agent_baseline_bleu'].mean(),
                mode_df['agent_baseline_rougeL_f1'].mean(), mode_df['agent_baseline_kw_recall'].mean()
            ],
            "4-Shot Agent Mean": [
                mode_df['four_shot_bert_f1'].mean(), mode_df['four_shot_bleu'].mean(),
                mode_df['four_shot_rougeL_f1'].mean(), mode_df['four_shot_kw_recall'].mean()
            ]
        }
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))

        # 可视化部分不变，仍然只对比最重要的BERT-F1分数
        plt.figure(figsize=(10, 6))
        plt.boxplot([
            mode_df['true_baseline_bert_f1'].dropna(),
            mode_df['agent_baseline_bert_f1'].dropna(),
            mode_df['four_shot_bert_f1'].dropna()
        ],
            labels=['True Baseline', 'Agent (0-Shot)', 'Agent (4-Shot)'])
        plt.title(f'BERT F1 Score Distribution for {mode.upper()} Mode')
        plt.ylabel('BERT F1 Score')
        plt.grid(True, linestyle='--', alpha=0.6)

        plot_filename = f"score_distribution_3-way_{mode}.png"
        plt.savefig(plot_filename)
        print(f"3-way comparison plot saved to {plot_filename}")
        plt.close()
    print("\nAll tasks complete.")


if __name__ == '__main__':
    run_analysis_from_file()