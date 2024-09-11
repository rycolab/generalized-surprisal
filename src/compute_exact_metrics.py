import os
import argparse
import pandas as pd
from measures import ExactScorer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute surprisal for a given dataset')
    parser.add_argument('--dataset', type=str, default='data/texts/devarda/texts.csv', help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='data/estimates_test', help='Path to save the output')
    parser.add_argument('--aggregate_by_word', default=True, action='store_true', help='Aggregate surprisal by word')
    parser.add_argument('--return_tokens', default=True, action='store_true', help='Return tokens')
    parser.add_argument('--model_name_or_path', type=str, default='openai-community/gpt2', help='Model name or path')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--debug_n', type=int, default=0, help='Number of documents to debug')
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

    # Initialize the scorer
    scorer = ExactScorer(model_name_or_path=args.model_name_or_path, device=args.device)

    # metrics = ['surprisal', 'prob', 'entropy', 'expected_prob']
    # if args.return_tokens:
    #     metrics = ['tokens'] + metrics

    scores_df_content = []
    runtimes_df_content = []

    # Compute exact metrics
    for text_id, text in list(zip(ids, texts)):
        if args.aggregate_by_word:
            scores, runtimes = scorer.word_score(
                text,
                return_tokens=args.return_tokens
            )
        else:
            scores, runtimes = scorer.token_score(
                text,
                return_tokens=args.return_tokens
            )

        for metric in scores:
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

    scores_df = pd.DataFrame(scores_df_content)
    scores_df.to_csv(os.path.join(args.output_dir, 'estimates.csv'), index=False)

    runtimes_df = pd.DataFrame(runtimes_df_content)
    runtimes_df.to_csv(os.path.join(args.output_dir, 'runtimes.csv'), index=False)

