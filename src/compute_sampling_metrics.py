import argparse
import os
import pandas as pd
from collections import defaultdict
from measures import SamplingScorer


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute sampling metrics for a given dataset')
    parser.add_argument('--dataset', type=str, default='data/texts/devarda/texts.csv', help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='data/estimates_test', help='Path to save the output')
    parser.add_argument('--return_tokens', default=True, action='store_true', help='Return tokens')
    parser.add_argument('--model_name_or_path', type=str, default='openai-community/gpt2', help='Model name or path')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--batch_size_distances', type=int, default=256, help='Batch size for distance calculation')
    # parser.add_argument('--batch_size_sequence_scores', type=int, default=512, help='Batch size for sequence score calculation')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples')
    parser.add_argument('--n_tokens_per_sample', type=int, default=2, help='Number of tokens per sample')
    parser.add_argument('--self_contextualisation_layer', type=int, default=-1, help='Layer to use for self_contextualisation')
    parser.add_argument('--contextualisation_layer', type=int, default=-1, help='Layer to use for contextualisation')
    parser.add_argument('--aggregate_by_word', default=True, action='store_true', help='Aggregate scores by word')  
    parser.add_argument('--importance_temperature', type=float, default=1, help='Temperature for importance sampling')
    parser.add_argument('--seed', type=int, default=0, help='Seed to use')
    parser.add_argument('--debug_n', type=int, default=0, help='Number of documents to debug')
    parser.add_argument('--log_every_n', type=int, default=10, help='Log progress every n documents')
    parser.add_argument('--bootstrap', default=True, action='store_true', help='Whether to compute variance with bootstrap samples')
    args = parser.parse_args()

    # Load csv dataset of documents using pandas
    dataset = pd.read_csv(args.dataset)
    assert 'text' in dataset.columns, 'The dataset should contain a column named "text"'
    assert 'id' in dataset.columns, 'The dataset should contain a column named "id"'

    texts = dataset['text'].tolist()
    ids = dataset['id'].tolist()
    if args.debug_n:
        texts = texts[:args.debug_n]
        ids = ids[:args.debug_n]

    scorer = SamplingScorer(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        batch_size_distances=args.batch_size_distances,
        # batch_size_sequence_scores=args.batch_size_sequence_scores,
        seed=args.seed,
    )

    metrics = [
        'decontextualised', 
        'when_in_sequence', 'when_first_token', 
        'surprisal_when_in_sequence', 'surprisal_when_first_token', 
        'expected_decontextualised', 
        'expected_seq_decontextualised', 
        'sequence_entropy', 'sequence_expected_prob'
    ]
    if args.return_tokens:
        metrics = ['tokens'] + metrics
               
    scores_df_content = []
    runtimes_df_content = []
    if args.bootstrap:
        summary_stats_df_content = []

    # Compute information value
    for i, (text_id, text) in enumerate(list(zip(ids, texts)), start=1):

        # print progress every log_every_n documents
        if i % args.log_every_n == 0:
            print(f'Processing document {i} of {len(texts)}')

        if not args.aggregate_by_word:
            scores, runtimes, summary_stats = scorer.token_score(
                text,
                n_samples=args.n_samples,
                return_tokens=args.return_tokens,
                self_contextualisation_layer=args.self_contextualisation_layer,
                contextualisation_layer=args.contextualisation_layer,
                max_new_tokens=args.n_tokens_per_sample,
                importance_temp=args.importance_temperature,
                bootstrap=args.bootstrap
            )
        else:
            scores, runtimes, summary_stats = scorer.word_score(
                text,
                n_samples=args.n_samples,
                return_tokens=args.return_tokens,
                self_contextualisation_layer=args.self_contextualisation_layer,
                contextualisation_layer=args.contextualisation_layer,
                max_new_tokens=args.n_tokens_per_sample,
                importance_temp=args.importance_temperature,
                bootstrap=args.bootstrap
            )
        for metric in metrics:
            row_dict = {
                'id': text_id,
                'metric': metric,
                'score': scores[metric],
            }
            scores_df_content.append(row_dict)
        for metric in runtimes:
            row_dict = {
                'id': text_id,
                'metric': metric,
                'runtime': runtimes[metric],
            }
            runtimes_df_content.append(row_dict)
        if args.bootstrap:
            for metric in summary_stats:
                row_dict = {
                    'id': text_id,
                    'metric': metric,
                    'mean': [x[0] for x in summary_stats[metric]],
                    'variance': [x[1] for x in summary_stats[metric]],
                    'resamples': [x[2].tolist() for x in summary_stats[metric]]
                }
                summary_stats_df_content.append(row_dict)

    scores_df = pd.DataFrame(scores_df_content)
    scores_df.to_csv(os.path.join(args.output_dir, 'estimates.csv'), index=False)

    runtimes_df = pd.DataFrame(runtimes_df_content)
    runtimes_df.to_csv(os.path.join(args.output_dir, 'runtimes.csv'), index=False)

    if args.bootstrap:
        summary_stats_df = pd.DataFrame(summary_stats_df_content)
        summary_stats_df.to_csv(os.path.join(args.output_dir, 'summary_stats.csv'), index=False)

