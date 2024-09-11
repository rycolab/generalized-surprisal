import os
import glob
import numpy as np
import pandas as pd
from ast import literal_eval
from collections import defaultdict
from tqdm import tqdm

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# ----------------------- PATHS ------------------------- #
# --------------------------------------------------------
DATA_FOLDER_PATH = 'data'
ALIGNED_DATASET_PATH = f'{DATA_FOLDER_PATH}/psyling/aligned.csv'
ESTIMATES_FOLDER_PATH = f'{DATA_FOLDER_PATH}/estimates_emnlp2024/'
OUTPUT_PATH = f'{DATA_FOLDER_PATH}/psyling/aligned_with_estimates_23july.csv'
# --------------------------------------------------------


if __name__ == '__main__':
    # Load psycholinguistic measurements from the Aligned dataset
    aligned = pd.read_csv(ALIGNED_DATASET_PATH)

    # Drop all irrelevant columns
    aligned = aligned.drop(aligned.filter(like='GPT').columns, axis=1)
    aligned = aligned.drop(aligned.filter(like='psg').columns, axis=1)
    aligned = aligned.drop(aligned.filter(like='gram').columns, axis=1)
    aligned = aligned.drop(aligned.filter(like='rnn').columns, axis=1)
    aligned = aligned.drop(aligned.filter(like='is_start_end').columns, axis=1)
    aligned = aligned.drop(aligned.filter(like='word2').columns, axis=1)
    aligned = aligned.drop(aligned.filter(like='item_id').columns, axis=1)

    # Punctuation and clitics
    # From Frank: "Words attached to a comma, clitics, sentence-initial, and sentence-final words 
    #              were discarded from further analysis [...]."
    # note that tokens at the beginning of sentences are already excluded (no cloze/rating data)
    contains_punct = []
    for index, row in aligned.iterrows():
        if "." in row.word:
            contains_punct.append(1)
        elif "," in row.word:
            contains_punct.append(1)
        elif "'" in row.word:
            contains_punct.append(1) # clitics
        else:
            contains_punct.append(0)

    aligned["contains_punct"] = contains_punct
    aligned = aligned[aligned.contains_punct == 0]


    # Load estimates of predictability and anticipation

    def parse_path_sampling(path):
        path = os.path.basename(path)
        path = path.split('_')
        try:
            return {
                'model': path[0],
                'n_samples': int(path[1][1:]),                    # e.g., n1024
                'ctx_layer': int(path[2][3:]),                    # e.g., ctx-1
                'selfctx_layer': int(path[3][7:]),                # e.g., selfctx-1
                'max_tokens': int(path[4][7:]),                   # e.g., ntokens10
                'seed': int(path[5][4:]),                         # e.g., seed0
                'temperature': float(f"{path[6][-1]}.{path[7]}")  # e.g., temp0_8
            }
        except ValueError:
            return {
                'model': path[0],
                'n_samples': int(path[1][1:]),                    # e.g., n1024
                'max_tokens': int(path[2][7:]),                   # e.g., ntokens10
                'seed': int(path[3][4:]),                         # e.g., seed0
                'temperature': float(f"{path[4][-1]}.{path[5]}")
            }

    def parse_path_exact(path):
        path = os.path.basename(path)
        path = path.split('_')
        return {
            'model': path[0],
            'n_samples': None,
            'ctx_layer': None,
            'selfctx_layer': None,
            'max_tokens': None,
            'seed': None
        }

    # load all files
    all_dirs = glob.glob(os.path.join(ESTIMATES_FOLDER_PATH, "*"))
    li = []
    for dir in all_dirs:
        if 'exact' in dir:
            df = pd.read_csv(os.path.join(dir, 'estimates.csv'), index_col=None, header=0)
            df = df.assign(**parse_path_exact(dir))
        else:
            df = pd.read_csv(os.path.join(dir, 'estimates.csv'), index_col=None, header=0)
            df = df.assign(**parse_path_sampling(dir))
        li.append(df)

    # concatenate all files
    estimates = pd.concat(li, axis=0, ignore_index=True)

    def get_list(s):
        try:
            return literal_eval(s)
        except SyntaxError:
            return list(map(float, s[1:-1].split()))
        except ValueError:
            return list(map(float, s[1:-1].split()))
        
    estimates.score = estimates.score.apply(get_list)
    estimates = estimates.rename(columns={'id': 'sent_id'})

    # drop rows where metric is "tokens"
    estimates = estimates[estimates.metric != 'tokens']


    # Collect all metrics, n_samples, and models
    METRICS = list(estimates.metric.unique())
    EXACT_METRICS = ['prob', 'surprisal', 'expected_prob', 'entropy']
    SAMPLING_METRICS = list(set(METRICS) - set(EXACT_METRICS))

    # remove all elements containing "prob_when" or "surprisal_when" from sampling metrics
    SAMPLING_METRICS = [metric for metric in SAMPLING_METRICS if 'prob_when' not in metric and 'surprisal_when' not in metric]

    NSAMPLES = list(estimates.n_samples.unique())
    NSAMPLES.remove(None)
    NSAMPLES = sorted(NSAMPLES)

    MODELS = list(estimates.model.unique())

    print("EXACT_METRICS:", EXACT_METRICS)
    print("SAMPLING_METRICS:", SAMPLING_METRICS)
    print("NSAMPLES:", NSAMPLES)
    print("MODELS:", MODELS)


    # Apply estimates to the Aligned dataset

    word_level_estimates = defaultdict(list)

    for _, row in tqdm(aligned.iterrows(), total=len(aligned)):
        sent_id = row['sent_id']
        word_position = row['context_length']
        
        # Get previous-words baseline predictors 
        # t - 1
        prev_item_df = aligned[(aligned.sent_id == sent_id) & (aligned.context_length == word_position - 1)]
        if prev_item_df.empty:
            word_level_estimates['Subtlex_log10_prev'].append(0.)
            word_level_estimates['length_prev'].append(0.)
        else:
            word_level_estimates['Subtlex_log10_prev'].append(prev_item_df['Subtlex_log10'].item())
            word_level_estimates['length_prev'].append(prev_item_df['length'].item())
        # t - 2
        prev_prev_item_df = aligned[(aligned.sent_id == sent_id) & (aligned.context_length == word_position - 2)]
        if prev_prev_item_df.empty:
            word_level_estimates['Subtlex_log10_prev_prev'].append(0.)
            word_level_estimates['length_prev_prev'].append(0.)
        else:
            word_level_estimates['Subtlex_log10_prev_prev'].append(prev_prev_item_df['Subtlex_log10'].item())
            word_level_estimates['length_prev_prev'].append(prev_prev_item_df['length'].item())
        
        # Get sampling-based estimates 
        sampling_tmp_df = estimates[(estimates.sent_id == sent_id) & (estimates.metric.isin(SAMPLING_METRICS))]

        sampling_tmp_df_grouped = sampling_tmp_df.groupby(['metric', 'model', 'n_samples', 'max_tokens', 'temperature'])
        for group, df_group in sampling_tmp_df_grouped:
            
            metric, model, n_samples, max_tokens, temperature = group
 
            # current timestep t
            current_score = df_group['score'].item()[word_position]
            word_level_estimates[f'{metric}_{model.replace("-", "_")}_{n_samples}_{max_tokens}_{str(temperature).replace(".", "_")}'].append(current_score)
            # t - 1
            try:
                prev_score = np.mean(df_group['score'].item()[word_position - 1])
            except IndexError:
                prev_score = 0.
            word_level_estimates[f'{metric}_{model.replace("-", "_")}_{n_samples}_{max_tokens}_{str(temperature).replace(".", "_")}_prev'].append(prev_score)
            # t - 2
            try:
                prev_prev_score = np.mean(df_group['score'].item()[word_position - 2])
            except IndexError:
                prev_prev_score = 0.
            word_level_estimates[f'{metric}_{model.replace("-", "_")}_{n_samples}_{max_tokens}_{str(temperature).replace(".", "_")}_prev_prev'].append(prev_prev_score)

            if n_samples <= 1024 and metric in ['when_in_sequence', 'when_first_token', 'when_in_first_2_tokens', 'when_in_first_3_tokens', 'when_in_first_4_tokens', 'when_in_first_5_tokens']:
                word_level_estimates[f'neg_log2_{metric}_{model.replace("-", "_")}_{n_samples}_{max_tokens}_{str(temperature).replace(".", "_")}'].append(-np.log2(current_score + 1e-4))
                word_level_estimates[f'neg_log2_{metric}_{model.replace("-", "_")}_{n_samples}_{max_tokens}_{str(temperature).replace(".", "_")}_prev'].append(-np.log2(prev_score + 1e-4))
                word_level_estimates[f'neg_log2_{metric}_{model.replace("-", "_")}_{n_samples}_{max_tokens}_{str(temperature).replace(".", "_")}_prev_prev'].append(-np.log2(prev_prev_score + 1e-4))

        exact_tmp_df = estimates[(estimates.sent_id == sent_id) & (estimates.metric.isin(EXACT_METRICS))]
        exact_tmp_df_grouped = exact_tmp_df.groupby(['metric', 'model'])
        for group, df_group in exact_tmp_df_grouped:
            metric, model = group
            # current timestep t
            current_score = df_group['score'].item()[word_position]
            word_level_estimates[f'{metric}_{model.replace("-", "_")}'].append(current_score)
            # t - 1
            try:
                prev_score = np.mean(df_group['score'].item()[word_position - 1])
            except IndexError:
                prev_score = 0.
            word_level_estimates[f'{metric}_{model.replace("-", "_")}_prev'].append(prev_score)
            # t - 2
            try:
                prev_prev_score = np.mean(df_group['score'].item()[word_position - 2])
            except IndexError:
                prev_prev_score = 0.
            word_level_estimates[f'{metric}_{model.replace("-", "_")}_prev_prev'].append(prev_prev_score)
   
    # Add estimates to the main dataframe
    for k, v in word_level_estimates.items():
        aligned[k] = v

    # Save the dataset
    aligned.to_csv(OUTPUT_PATH, index=False)
