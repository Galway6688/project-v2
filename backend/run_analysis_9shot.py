# run_analysis_9shot.py

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
RAW_PREDICTIONS_CSV = "evaluation_predictions_9shot.csv"
FINAL_RESULTS_CSV = "evaluation_results_9shot_detailed.csv"


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
def run_analysis_9shot():
    """
    Analyze 9-shot experiment results for tactile and vision modes.
    Compare 0-shot baseline vs 9-shot performance.
    """
    print("--- Step 1: Loading Resources for 9-Shot Analysis ---")
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"✅ Found available GPU. Using device: {device}")
    else:
        device = 'cpu'
        print("⚠️ No GPU found. Using device: cpu. This may be slow.")

    try:
        results_df = pd.read_csv(RAW_PREDICTIONS_CSV)
        print(f"Successfully loaded 9-shot predictions from '{RAW_PREDICTIONS_CSV}'.")
        print(f"Total samples: {len(results_df)}")
        print(f"Modes: {results_df['mode'].value_counts().to_dict()}")
    except FileNotFoundError:
        print(f"Error: '{RAW_PREDICTIONS_CSV}' not found. Please run 'run_predictions_9shot.py' first.")
        return

    print("Loading evaluation models...")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' model...")
        nltk.download('punkt')
    
    bert_scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', lang='en', device=device)
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    chencherry = SmoothingFunction()
    print("Evaluation models loaded.\n")

    print("--- Step 2: Calculating Detailed Metrics for 9-Shot Experiment ---")

    # If running on CPU, use a safe batch_size. On GPU, larger batches can be processed.
    batch_size = 1 if device == 'cpu' else 32
    print(f"Using batch_size={batch_size} for BERTScore calculation on {device}.")

    # Filter out error rows for scoring
    valid_results_df = results_df[
        (results_df['baseline_output'] != 'ERROR') & 
        (results_df['nine_shot_output'] != 'ERROR')
    ].copy()

    if not valid_results_df.empty:
        print(f"Valid samples for evaluation: {len(valid_results_df)}")
        
        # Calculate BERTScore for both baseline and 9-shot
        P_base, R_base, F1_base = bert_scorer.score(
            valid_results_df['baseline_output'].tolist(),
            valid_results_df['ground_truth'].tolist(), 
            batch_size=batch_size
        )
        P_9shot, R_9shot, F1_9shot = bert_scorer.score(
            valid_results_df['nine_shot_output'].tolist(),
            valid_results_df['ground_truth'].tolist(), 
            batch_size=batch_size
        )
        
        valid_results_df['baseline_bert_f1'] = F1_base.numpy()
        valid_results_df['nine_shot_bert_f1'] = F1_9shot.numpy()

        # Calculate other metrics (BLEU, ROUGE, Keyword Recall)
        other_metrics = []
        for index, row in tqdm(valid_results_df.iterrows(), total=len(valid_results_df),
                               desc="Calculating other metrics"):
            gt_list = str(row['ground_truth']).split()
            base_list = str(row['baseline_output']).split()
            nine_shot_list = str(row['nine_shot_output']).split()
            
            rouge_base = rouge.score(row['ground_truth'], row['baseline_output'])
            rouge_9shot = rouge.score(row['ground_truth'], row['nine_shot_output'])
            
            other_metrics.append({
                'baseline_bleu': sentence_bleu([gt_list], base_list, smoothing_function=chencherry.method1),
                'nine_shot_bleu': sentence_bleu([gt_list], nine_shot_list, smoothing_function=chencherry.method1),
                'baseline_rougeL_f1': rouge_base['rougeL'].fmeasure,
                'nine_shot_rougeL_f1': rouge_9shot['rougeL'].fmeasure,
                'baseline_kw_recall': calculate_keyword_recall(row['baseline_output'], row['ground_truth']),
                'nine_shot_kw_recall': calculate_keyword_recall(row['nine_shot_output'], row['ground_truth']),
            })
        
        other_metrics_df = pd.DataFrame(other_metrics, index=valid_results_df.index)
        results_df = valid_results_df.merge(other_metrics_df, left_index=True, right_index=True, how="left")
        
        # Calculate improvement scores
        results_df['bert_f1_improvement'] = results_df['nine_shot_bert_f1'] - results_df['baseline_bert_f1']
        results_df['bleu_improvement'] = results_df['nine_shot_bleu'] - results_df['baseline_bleu']
        results_df['rougeL_improvement'] = results_df['nine_shot_rougeL_f1'] - results_df['baseline_rougeL_f1']
        results_df['kw_recall_improvement'] = results_df['nine_shot_kw_recall'] - results_df['baseline_kw_recall']
        
        # Save detailed results
        results_df.to_csv(FINAL_RESULTS_CSV, index=False, encoding='utf-8-sig')
        print(f"Detailed 9-shot results saved to '{FINAL_RESULTS_CSV}'")
    else:
        print("No valid predictions found in the raw file to analyze.")
        return

    print("\n--- Step 3: Generating Statistical Summary & Visualization for 9-Shot Experiment ---")
    modes_to_evaluate = ['tactile', 'vision']  # Only tactile and vision modes
    
    for mode in modes_to_evaluate:
        mode_df = results_df[results_df['mode'] == mode]
        if mode_df.empty: 
            print(f"No data found for mode: {mode}")
            continue
            
        print(f"\n----- 9-Shot Statistical Summary for MODE: {mode.upper()} -----")
        
        # Calculate statistics
        stats = {
            "Metric": ["BERT-F1", "BLEU", "ROUGE-L-F1", "KW-Recall"],
            "Baseline Mean": [
                mode_df['baseline_bert_f1'].mean(), 
                mode_df['baseline_bleu'].mean(),
                mode_df['baseline_rougeL_f1'].mean(), 
                mode_df['baseline_kw_recall'].mean()
            ],
            "9-Shot Mean": [
                mode_df['nine_shot_bert_f1'].mean(), 
                mode_df['nine_shot_bleu'].mean(),
                mode_df['nine_shot_rougeL_f1'].mean(), 
                mode_df['nine_shot_kw_recall'].mean()
            ],
            "9-Shot Std Dev": [
                mode_df['nine_shot_bert_f1'].std(), 
                mode_df['nine_shot_bleu'].std(),
                mode_df['nine_shot_rougeL_f1'].std(), 
                mode_df['nine_shot_kw_recall'].std()
            ],
            "Improvement": [
                mode_df['bert_f1_improvement'].mean(),
                mode_df['bleu_improvement'].mean(),
                mode_df['rougeL_improvement'].mean(),
                mode_df['kw_recall_improvement'].mean()
            ]
        }
        
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.boxplot([
            mode_df['baseline_bert_f1'].dropna(), 
            mode_df['nine_shot_bert_f1'].dropna()
        ], labels=['0-Shot Baseline', '9-Shot Agent'])
        plt.title(f'BERT F1 Score Distribution for {mode.upper()} Mode (9-Shot Experiment)')
        plt.ylabel('BERT F1 Score')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add improvement statistics to the plot
        improvement_mean = mode_df['bert_f1_improvement'].mean()
        plt.text(0.02, 0.98, f'Mean Improvement: {improvement_mean:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plot_filename = f"score_distribution_9shot_{mode}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"9-shot score distribution plot saved to {plot_filename}")
        plt.close()

    # Generate overall comparison summary
    print(f"\n----- Overall 9-Shot Experiment Summary -----")
    overall_stats = {
        "Mode": [],
        "Samples": [],
        "BERT-F1 Baseline": [],
        "BERT-F1 9-Shot": [],
        "BERT-F1 Improvement": [],
        "BLEU Improvement": [],
        "ROUGE-L Improvement": [],
        "KW-Recall Improvement": []
    }
    
    for mode in modes_to_evaluate:
        mode_df = results_df[results_df['mode'] == mode]
        if not mode_df.empty:
            overall_stats["Mode"].append(mode.upper())
            overall_stats["Samples"].append(len(mode_df))
            overall_stats["BERT-F1 Baseline"].append(f"{mode_df['baseline_bert_f1'].mean():.4f}")
            overall_stats["BERT-F1 9-Shot"].append(f"{mode_df['nine_shot_bert_f1'].mean():.4f}")
            overall_stats["BERT-F1 Improvement"].append(f"{mode_df['bert_f1_improvement'].mean():.4f}")
            overall_stats["BLEU Improvement"].append(f"{mode_df['bleu_improvement'].mean():.4f}")
            overall_stats["ROUGE-L Improvement"].append(f"{mode_df['rougeL_improvement'].mean():.4f}")
            overall_stats["KW-Recall Improvement"].append(f"{mode_df['kw_recall_improvement'].mean():.4f}")
    
    overall_df = pd.DataFrame(overall_stats)
    print(overall_df.to_string(index=False))
    
    print("\n✅ 9-shot analysis complete!")
    print(f"Results saved to: {FINAL_RESULTS_CSV}")
    print(f"Plots saved as: score_distribution_9shot_tactile.png, score_distribution_9shot_vision.png")


if __name__ == '__main__':
    run_analysis_9shot()
