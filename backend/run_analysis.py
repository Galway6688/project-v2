# run_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import nltk
from tqdm import tqdm

# Evaluation metrics libraries
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# --- 1. Configuration Section ---
RAW_PREDICTIONS_CSV = "evaluation_predictions_raw.csv"
FINAL_RESULTS_CSV = "evaluation_results_detailed.csv"


# --- 2. Helper Functions ---
def calculate_keyword_recall(candidate_str: str, reference_str: str) -> float:
    """Calculate how many keywords from the reference answer are included in the generated result"""
    if not isinstance(candidate_str, str) or not isinstance(reference_str, str):
        return 0.0
    ref_keywords = set([kw.strip() for kw in reference_str.split(',') if kw.strip()])
    cand_keywords = set([kw.strip() for kw in candidate_str.split(',') if kw.strip()])

    if not ref_keywords:
        return 1.0 if not cand_keywords else 0.0

    matched_keywords = ref_keywords.intersection(cand_keywords)
    return len(matched_keywords) / len(ref_keywords)


# --- 3. Main Analysis Function ---
def run_analysis_from_file():
    """
    Read data from raw prediction files and execute all metric calculations, statistical analysis, and visualization.
    """
    print("--- Step 1: Loading Resources for Analysis ---")
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"✅ Found available GPU. Using device: {device}")
    else:
        device = 'cpu'
        print("⚠️ No GPU found. Using device: cpu. This may be slow.")

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

    # If running on CPU, use a safe batch_size. On GPU, larger batches can be processed.
    batch_size = 1 if device == 'cpu' else 32
    print(f"Using batch_size={batch_size} for BERTScore calculation on {device}.")

    # Filter out error rows for scoring
    valid_results_df = results_df[results_df['baseline_output'] != 'ERROR'].copy()

    if not valid_results_df.empty:
        P_base, R_base, F1_base = bert_scorer.score(valid_results_df['baseline_output'].tolist(),
                                                    valid_results_df['ground_truth'].tolist(), batch_size=batch_size)
        P_5shot, R_5shot, F1_5shot = bert_scorer.score(valid_results_df['five_shot_output'].tolist(),
                                                       valid_results_df['ground_truth'].tolist(), batch_size=batch_size)
        valid_results_df['baseline_bert_f1'] = F1_base.numpy()
        valid_results_df['five_shot_bert_f1'] = F1_5shot.numpy()

        other_metrics = []
        for index, row in tqdm(valid_results_df.iterrows(), total=len(valid_results_df),
                               desc="Calculating other metrics"):
            gt_list = str(row['ground_truth']).split()
            base_list = str(row['baseline_output']).split()
            fiveshot_list = str(row['five_shot_output']).split()
            rouge_base = rouge.score(row['ground_truth'], row['baseline_output'])
            rouge_5shot = rouge.score(row['ground_truth'], row['five_shot_output'])
            other_metrics.append(
                {'baseline_bleu': sentence_bleu([gt_list], base_list, smoothing_function=chencherry.method1),
                 'five_shot_bleu': sentence_bleu([gt_list], fiveshot_list, smoothing_function=chencherry.method1),
                 'baseline_rougeL_f1': rouge_base['rougeL'].fmeasure,
                 'five_shot_rougeL_f1': rouge_5shot['rougeL'].fmeasure,
                 'baseline_kw_recall': calculate_keyword_recall(row['baseline_output'], row['ground_truth']),
                 'five_shot_kw_recall': calculate_keyword_recall(row['five_shot_output'], row['ground_truth']), })
        other_metrics_df = pd.DataFrame(other_metrics, index=valid_results_df.index)
        results_df = valid_results_df.merge(other_metrics_df, left_index=True, right_index=True, how="left")
        results_df['bert_f1_improvement'] = results_df['five_shot_bert_f1'] - results_df['baseline_bert_f1']
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
        stats = {"Metric": ["BERT-F1", "BLEU", "ROUGE-L-F1", "KW-Recall"],
                 "Baseline Mean": [mode_df['baseline_bert_f1'].mean(), mode_df['baseline_bleu'].mean(),
                                   mode_df['baseline_rougeL_f1'].mean(), mode_df['baseline_kw_recall'].mean()],
                 "5-Shot Mean": [mode_df['five_shot_bert_f1'].mean(), mode_df['five_shot_bleu'].mean(),
                                 mode_df['five_shot_rougeL_f1'].mean(), mode_df['five_shot_kw_recall'].mean()],
                 "5-Shot Std Dev": [mode_df['five_shot_bert_f1'].std(), mode_df['five_shot_bleu'].std(),
                                    mode_df['five_shot_rougeL_f1'].std(), mode_df['five_shot_kw_recall'].std()]}
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))
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
    print("\nAll tasks complete.")


if __name__ == '__main__':
    run_analysis_from_file()