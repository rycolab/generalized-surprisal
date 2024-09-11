import copy
import torch
import numpy as np
from collections import defaultdict
from functools import partial
from scipy import spatial
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

try:
    from . import utils
except ImportError:
    import utils



class Scorer:

    def __init__(self, model_name_or_path: str, device: Optional[str] = "cuda"):
        """
        :param model_name_or_path (str):
            the name or path to a model compatible with AutoModelWithLMHead
        :param device (str):
            "cpu", "cuda", or "mps"
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
        
        self.model.eval()
        # get the model's context window size
        if "n_positions" in self.model.config.__dict__.keys():
            self.max_seq_len = self.model.config.n_positions
        else:
            self.max_seq_len = self.model.config.max_position_embeddings
        
        if "hidden_size" in self.model.config.__dict__.keys():
            self.embed_size = self.model.config.hidden_size  # GPT-Neo and OPT
        else:
            self.embed_size = self.model.config.n_embd  # GPT-2


class ExactScorer(Scorer):

    def __init__(self, model_name_or_path: str, device: Optional[str] = "cpu"):
        """
        :param model_name_or_path: the name or path to a model
            compatible with AutoModelWithLMHead
        :param device: "cpu" or "cuda"
        """
        super().__init__(model_name_or_path, device)

    def _score(
            self,
            input: str
    ):
        
        STRIDE = 200
        scores = {'surprisal': [], 'prob': [], 'entropy': [], 'expected_prob': []}
        runtimes = {'surprisal': np.array([]), 'prob': np.array([]), 'entropy': np.array([]), 'expected_prob': np.array([])}
        input_tokenized = defaultdict(list)
        start_ind = 0

        while True:
            encodings = self.tokenizer(input[start_ind:], max_length=self.max_seq_len - 1, truncation=True, return_attention_mask=False, return_offsets_mapping=True)
            tensor_input = torch.tensor([self.tokenizer.bos_token_id] + encodings['input_ids'], device=self.device)
            offset = 0 if start_ind == 0 else STRIDE - 1

            # forward pass
            t1 = time.time()
            outputs = self.model(tensor_input.unsqueeze(0))

            # tranform logits into log probabilities (for the entire vocabulary)
            log_probs = torch.nn.functional.log_softmax(outputs.logits.squeeze(0), dim=-1)
            probs = torch.exp(log_probs)
            t2 = time.time()

            # next-token log probabilities
            next_token_surprisals = - log_probs[
                range(len(tensor_input) - 1),
                tensor_input[1:],
            ].detach()
            next_token_surprisals /= torch.tensor(2).log()
            t3 = time.time()

            # next-token probabilities
            next_token_probs = probs[
                range(len(tensor_input) - 1),
                tensor_input[1:],
            ].detach()
            t4 = time.time()

            # next-token entropy
            next_token_entropies = - (log_probs * probs).nansum(-1)
            next_token_entropies = next_token_entropies[range(len(tensor_input) - 1)].detach()
            t5 = time.time()

            # expected next-token probability
            expected_next_token_prob = (probs * probs).nansum(-1)
            expected_next_token_prob = expected_next_token_prob[range(len(tensor_input) - 1)].detach()
            t6 = time.time()

            assert len(next_token_surprisals) == len(next_token_probs) == len(next_token_entropies) == len(expected_next_token_prob)

            scores['surprisal'].extend(next_token_surprisals.cpu())
            scores['prob'].extend(next_token_probs.cpu())
            scores['entropy'].extend(next_token_entropies.cpu())
            scores['expected_prob'].extend(expected_next_token_prob.cpu())

            t_shared = t2-t1
            runtimes['surprisal'] = np.concatenate((runtimes['surprisal'], (t_shared+t3-t2)/len(next_token_surprisals) * np.ones(len(next_token_surprisals))))
            runtimes['prob'] = np.concatenate((runtimes['prob'], (t_shared+t4-t3)/len(next_token_probs) * np.ones(len(next_token_probs))))
            runtimes['entropy'] = np.concatenate((runtimes['entropy'], (t_shared+t5-t4)/len(next_token_entropies) * np.ones(len(next_token_entropies))))
            runtimes['expected_prob'] = np.concatenate((runtimes['expected_prob'], (t_shared+t6-t5)/len(expected_next_token_prob) * np.ones(len(expected_next_token_prob))))

            input_tokenized['offset_mapping'].extend([(i + start_ind, j + start_ind) for i, j in encodings['offset_mapping'][offset:]])
            input_tokenized["input_ids"].extend(encodings['input_ids'][offset:])
            if encodings['offset_mapping'][-1][1] + start_ind == len(input):
                break
            start_ind += encodings['offset_mapping'][-STRIDE][1]

        return scores, input_tokenized, runtimes

    
    def token_score(
            self,
            input: str,
            return_tokens: Optional[bool]=False
    ):
        """
        :param input (str): 
            the input string to obtain surprisal scores for
        :param return_tokens (bool):
            whether to return the tokens (or words) corresponding to each surprisal score

        :return (dict):
            dictionary containing token level 'surprisal', 'entropy', 'deviation' stored as numpy
            arrays, and 'tokens' (a list of unit_sized strings) if return_tokens.
        """
        scores, input_tokenized, runtimes = self._score(input)

        rdict = {
            "surprisal": np.asarray(scores['surprisal']),
            "prob": np.asarray(scores['prob']),
            "entropy": np.asarray(scores['entropy']),
            "expected_prob": np.asarray(scores['expected_prob'])
        }
        if return_tokens:
            next_tokens = self.tokenizer.convert_ids_to_tokens(input_tokenized['input_ids'])
            assert len(next_tokens) == len(scores['surprisal'])
            rdict['tokens'] = next_tokens

        return rdict, runtimes


    def word_score(
            self,
            input: str,
            return_tokens: Optional[bool]=False
    ):
        """
        :param input (str): 
            the input string to obtain surprisal scores for
        :param return_tokens (bool):
            whether to return the tokens (or words) corresponding to each surprisal score

        :return (dict):
            dictionary containing word level 'surprisal', 'entropy', 'deviation' stored as numpy
            arrays, and 'tokens' (a list of unit_sized strings) if return_tokens.
        """

        scores, input_tokenized, runtimes = self._score(input)

        # aggregate token scores that belong to the same word (as delimited by whitespace)
        offsets = input_tokenized['offset_mapping']
        next_token_surprisals = utils.aggregate_score_by_word(input, scores['surprisal'], offsets, mode='sum')
        next_token_probs = utils.aggregate_score_by_word(input, scores['prob'], offsets, mode='multiply')
        next_token_entropies = utils.aggregate_score_by_word(input, scores['entropy'], offsets, mode='first')
        next_token_expected_probs = utils.aggregate_score_by_word(input, scores['expected_prob'], offsets, mode='first')
        if return_tokens:
            next_tokens = utils.aggregate_score_by_word(input, input_tokenized["input_ids"], offsets, mode='string', tokenizer=self.tokenizer)
            assert len(next_tokens) == len(next_token_surprisals)

        rdict = {
            "surprisal": np.asarray(next_token_surprisals),
            "prob": np.asarray(next_token_probs),
            "entropy": np.asarray(next_token_entropies),
            "expected_prob": np.asarray(next_token_expected_probs)
        }
        if return_tokens:
            rdict['tokens'] = next_tokens
        
        return rdict, runtimes
    

class SamplingScorer(Scorer):

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = "cpu",
        seed: Optional[int] = 0,
        batch_size_distances: Optional[int] = 256,
        batch_size_sequence_scores: Optional[int] = 512
    ):
        """
        :param model_name_or_path: the name or path to a model compatible with AutoModelWithLMHead
        :param device: "cpu" or "cuda"
        """
        super().__init__(model_name_or_path, device)
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.bsize_dist = batch_size_distances
        self.bsize_seq_scores = batch_size_sequence_scores

    def _score(
        self,
        input: str,
        n_samples: int, 
        self_contextualisation_layer: int,
        contextualisation_layer: int,
        max_new_tokens: int=1,
        importance_temp: float=1.0,
        bootstrap: bool=False
    ):
        STRIDE = 200
        scores = defaultdict(list)
        runtimes = defaultdict(partial(np.ndarray, 0))
        summary_stats = defaultdict(list)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        input_tokenized = defaultdict(list)
        start_ind = 0

        while True:
            encodings = self.tokenizer(input[start_ind:], max_length=self.max_seq_len-1, truncation=True, return_attention_mask=False, return_offsets_mapping=True)
            tensor_input = torch.tensor([self.tokenizer.bos_token_id] + encodings['input_ids'], device=self.device)
            offset = 0 if start_ind == 0 else STRIDE - 1
            for t in range(1, len(tensor_input)):
                if t <= offset:
                    continue
                context_ids = tensor_input[:t]
                token_alternatives_at_t = []
                seq_alternatives_at_t = []

                # multi-token alternatives
                t1 = time.time()
                outputs = self.model.generate(
                    context_ids.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    num_return_sequences=n_samples,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_logits=True,
                    temperature=importance_temp
                )

                # compute log probabilities of sampled tokens
                logits = torch.stack(outputs.logits)
                logits = torch.reshape(logits, (n_samples, max_new_tokens, logits.shape[2]))
                standard_logprobs_all = torch.nn.functional.log_softmax(logits, dim=2)
                standard_logprobs = torch.gather(
                    standard_logprobs_all, 
                    dim=2, 
                    index=outputs.sequences[:, range(len(context_ids), len(context_ids)+max_new_tokens)].unsqueeze(-1)
                ).squeeze(-1)
                assert torch.isfinite(standard_logprobs).all()

                if importance_temp != 1: # compute probabilities for importance sampling
                    proposal_logprobs_all = torch.nn.functional.log_softmax(logits / importance_temp, dim=2)
                    proposal_logprobs = torch.gather(
                        proposal_logprobs_all, 
                        dim=2, 
                        index=outputs.sequences[:, range(len(context_ids), len(context_ids)+max_new_tokens)].unsqueeze(-1)
                    ).squeeze(-1)
                    assert torch.isfinite(proposal_logprobs).all()

                # alternative does not include context tokens
                for sample in outputs.sequences:
                    token_alternatives_at_t.append(sample[t])
                    seq_alternatives_at_t.append(sample[t:])
                t2 = time.time()
                t_shared = t2-t1

                # get next-token importance weights
                if importance_temp != 1:
                    standard_logprobs_token = standard_logprobs[:,0]
                    proposal_logprobs_token = proposal_logprobs[:,0]
                    importance_weights_token = torch.exp(standard_logprobs_token - proposal_logprobs_token)
                    # # if any of the importance weights is <0, print
                    # if (importance_weights_token < 0).any():
                    #     print("Negative importance weight found!")
                    #     print("Standard log probs: ", standard_logprobs_token)
                    #     print("Proposal log probs: ", proposal_logprobs_token)
                    #     print("Importance weights: ", importance_weights_token)
                    assert torch.isfinite(importance_weights_token).all()
                else:
                    importance_weights_token = None

                # compute mean distance between each target and its respective alternatives
                # dectx, selfctx, ctx, expected_dectx, expected_selfctx, expected_ctx = self.compute_token_distances(
                #     token_alternatives_at_t, context_ids, tensor_input[t], self_contextualisation_layer, contextualisation_layer, importance_weights_token, bootstrap
                # )
                dectx, expected_dectx = self.compute_token_distances(
                    token_alternatives_at_t, context_ids, tensor_input[t], self_contextualisation_layer, contextualisation_layer, importance_weights_token, bootstrap
                )

                # if dectx[0] < 0:
                #     print("Negative decontextualised distance found!")
                #     print("Decontextualised distance: ", dectx[0])
                #     print("Decontextualised distance: ", dectx[1])
                #     print("Decontextualised distance: ", dectx[2])

                # information value
                scores['decontextualised'].append(dectx[0])
                # scores['self_contextualised'].append(selfctx[0])
                # scores['contextualised'].append(ctx[0])
                runtimes['decontextualised'] = np.concatenate((runtimes['decontextualised'], np.array([dectx[1] + t_shared])))
                # runtimes['self_contextualised'] = np.concatenate((runtimes['self_contextualised'], np.array([selfctx[1]])))
                # runtimes['contextualised'] = np.concatenate((runtimes['contextualised'], np.array([ctx[1]])))
                if bootstrap:
                    summary_stats['decontextualised'].append(dectx[2])
                    # summary_stats['self_contextualised'].append(selfctx[2])
                    # summary_stats['contextualised'].append(ctx[2])
                
                # expected information value
                scores['expected_decontextualised'].append(expected_dectx[0])
                # scores['expected_self_contextualised'].append(expected_selfctx[0])
                # scores['expected_contextualised'].append(expected_ctx[0])
                runtimes['expected_decontextualised'] = np.concatenate((runtimes['expected_decontextualised'], np.array([expected_dectx[1] + t_shared])))
                # runtimes['expected_self_contextualised'] = np.concatenate((runtimes['expected_self_contextualised'], np.array([expected_selfctx[1]])))
                # runtimes['expected_contextualised'] = np.concatenate((runtimes['expected_contextualised'], np.array([expected_ctx[1]])))
                if bootstrap:
                    summary_stats['expected_decontextualised'].append(expected_dectx[2])
                    # summary_stats['expected_self_contextualised'].append(expected_selfctx[2])
                    # summary_stats['expected_contextualised'].append(expected_ctx[2])

                # sequence-level scores
                # if max_new_tokens > 1:
                
                standard_logprobs_sequence = torch.sum(standard_logprobs.double(), dim=1)  # todo: ignore every token after the first EOS token
                assert torch.isfinite(standard_logprobs_sequence).all()

                # sequence probabilities and surprisals as numpy arrays; to be used for (log)probability-based metrics
                seq_probs_np = torch.exp(standard_logprobs_sequence).cpu().numpy()
                seq_surprisals_np = - (standard_logprobs_sequence / torch.tensor(2).log()).cpu().numpy()
                # seq_surprisals_np = standard_logprobs_sequence.cpu().numpy()
                assert all(seq_probs_np) != 0

                # calculate sequence-level importance weights
                if importance_temp != 1:
                    proposal_logprobs_sequence = torch.sum(proposal_logprobs.double(), dim=1)  # todo: ignore every token after the first EOS token
                    assert torch.isfinite(proposal_logprobs_sequence).all()
                    importance_weights_sequence = torch.exp(standard_logprobs_sequence - proposal_logprobs_sequence)
                    assert importance_weights_sequence.all()
                else:
                    importance_weights_sequence = None

                # compute self-distances between sampled sequences
                # seq_expected_dectx, seq_expected_selfctx, seq_expected_ctx = self.compute_expected_sequence_distances(
                #     seq_alternatives_at_t, context_ids, self_contextualisation_layer, contextualisation_layer, importance_weights_sequence, bootstrap
                # )
                seq_expected_dectx = self.compute_expected_sequence_distances(
                    seq_alternatives_at_t, context_ids, self_contextualisation_layer, contextualisation_layer, importance_weights_sequence, bootstrap
                )

                # expected information value (sequence)
                scores['expected_seq_decontextualised'].append(seq_expected_dectx[0])
                # scores['expected_seq_self_contextualised'].append(seq_expected_selfctx[0])
                # scores['expected_seq_contextualised'].append(seq_expected_ctx[0])
                runtimes['expected_seq_decontextualised'] = np.concatenate((runtimes['expected_seq_decontextualised'], np.array([seq_expected_dectx[1] + t_shared])))
                # runtimes['expected_seq_self_contextualised'] = np.concatenate((runtimes['expected_seq_self_contextualised'], np.array([seq_expected_selfctx[1]])))
                # runtimes['expected_seq_contextualised'] = np.concatenate((runtimes['expected_seq_contextualised'], np.array([seq_expected_ctx[1]])))
                if bootstrap:
                    summary_stats['expected_seq_decontextualised'].append(seq_expected_dectx[2])
                    # summary_stats['expected_seq_self_contextualised'].append(seq_expected_selfctx[2])
                    # summary_stats['expected_seq_contextualised'].append(seq_expected_ctx[2])

                # check whether the target token is in the sequence or if it is among the first n token
                t6 = time.time()
                # --------------------------------------------------------------------------------------------------
                target_in_sequence = []
                target_is_first_token = []
                # target_in_first_2_tokens, target_in_first_3_tokens, target_in_first_4_tokens, target_in_first_5_tokens = [], [], [], []
                for seq in outputs.sequences:
                    target_in_sequence.append(tensor_input[t] in seq[t:])
                    target_is_first_token.append((tensor_input[t] == seq[t]).item())
                    # target_in_first_2_tokens.append(tensor_input[t] in seq[t:t+2])
                    # target_in_first_3_tokens.append(tensor_input[t] in seq[t:t+3])
                    # target_in_first_4_tokens.append(tensor_input[t] in seq[t:t+4])
                    # target_in_first_5_tokens.append(tensor_input[t] in seq[t:t+5])
                # transform into numpy arrays
                target_in_sequence = np.array(target_in_sequence)
                target_is_first_token = np.array(target_is_first_token)
                # target_in_first_2_tokens = np.array(target_in_first_2_tokens)
                # target_in_first_3_tokens = np.array(target_in_first_3_tokens)
                # target_in_first_4_tokens = np.array(target_in_first_4_tokens)
                # target_in_first_5_tokens = np.array(target_in_first_5_tokens)
                # --------------------------------------------------------------------------------------------------
                t7 = time.time()

                # --------------------------------------------------------------------------------------------------
                if importance_temp == 1:
                    scores['when_in_sequence'].append(np.mean(target_in_sequence))
                    t8 = time.time()
                    if bootstrap:
                        summary_stats['when_in_sequence'].append(self.bootstrap(target_in_sequence))
                else: 
                    scores['when_in_sequence'].append(np.mean(target_in_sequence * importance_weights_sequence.cpu().numpy()))
                    t8 = time.time()
                    if bootstrap:
                        summary_stats['when_in_sequence'].append(self.bootstrap(target_in_sequence * importance_weights_sequence.cpu().numpy()))
    
                # --------------------------------------------------------------------------------------------------
                t9 = time.time()
                # --------------------------------------------------------------------------------------------------
                if importance_temp == 1:
                    scores['when_first_token'].append(np.mean(target_is_first_token))
                    t10 = time.time()
                    if bootstrap:
                        summary_stats['when_first_token'].append(self.bootstrap(target_is_first_token))
                else:
                    scores['when_first_token'].append(np.mean(target_is_first_token * importance_weights_sequence.cpu().numpy()))
                    t10 = time.time()
                    if bootstrap:
                        summary_stats['when_first_token'].append(self.bootstrap(target_is_first_token * importance_weights_sequence.cpu().numpy()))
                # --------------------------------------------------------------------------------------------------
                
                # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['when_in_first_2_tokens'].append(np.mean(target_in_first_2_tokens))
                # else:
                #     scores['when_in_first_2_tokens'].append(np.mean(target_in_first_2_tokens * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t10 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['when_in_first_3_tokens'].append(np.mean(target_in_first_3_tokens))
                # else:
                #     scores['when_in_first_3_tokens'].append(np.mean(target_in_first_3_tokens * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t11 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['when_in_first_4_tokens'].append(np.mean(target_in_first_4_tokens))
                # else:
                #     scores['when_in_first_4_tokens'].append(np.mean(target_in_first_4_tokens * importance_weights_sequence.cpu().numpy()))
        
                # # --------------------------------------------------------------------------------------------------
                # t12 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['when_in_first_5_tokens'].append(np.mean(target_in_first_5_tokens))
                # else:
                #     scores['when_in_first_5_tokens'].append(np.mean(target_in_first_5_tokens * importance_weights_sequence.cpu().numpy()))
                # --------------------------------------------------------------------------------------------------
                t13 = time.time()
    
                # # Get sequence probabilities
                # # --------------------------------------------------------------------------------------------------
                # # prepare for batching
                # n_batches = n_samples // self.bsize_seq_scores
                # remainder = n_samples % self.bsize_seq_scores
                # transition_scores = torch.empty(outputs.sequences.shape[0], outputs.sequences.shape[1] - t, device=self.device)
                # # create batches
                # sequence_batches, scores_batches = [], []
                # for i in range(n_batches):
                #     sequence_batches.append(outputs.sequences[i*self.bsize_seq_scores:(i+1)*self.bsize_seq_scores, :])
                #     scores_batches.append([score[i*self.bsize_seq_scores:(i+1)*self.bsize_seq_scores, :] for score in outputs.scores])
                # if remainder:
                #     sequence_batches.append(outputs.sequences[-remainder:, :])
                #     scores_batches.append([score[-remainder:, :] for score in outputs.scores])
                
                # # get logits of samples
                # for i in range(n_batches + (remainder > 0)):
                #     batch_transition_scores = self.model.compute_transition_scores(
                #         sequence_batches[i], scores_batches[i], normalize_logits=True
                #     )
                #     transition_scores[i*self.bsize_seq_scores:(i+1)*self.bsize_seq_scores, :] = batch_transition_scores

                # # compute sample log probs and probs 
                # seq_logprobs = transition_scores.masked_fill_(transition_scores == float("-inf"), 0).sum(dim=1)
                # assert all(torch.isfinite(seq_logprobs))
                # seq_probs = torch.exp(seq_logprobs).cpu().numpy()
                # seq_logprobs = seq_logprobs.cpu().numpy()
                # assert all(seq_probs) != 0
                # # --------------------------------------------------------------------------------------------------
                # t14 = time.time()

                # --------------------------------------------------------------------------------------------------
                if importance_temp == 1:
                    scores['surprisal_when_in_sequence'].append(np.mean(seq_surprisals_np * target_in_sequence))
                    t14 = time.time()
                    if bootstrap:
                        summary_stats['surprisal_when_in_sequence'].append(self.bootstrap(seq_surprisals_np * target_in_sequence))
                else:
                    scores['surprisal_when_in_sequence'].append(np.mean(seq_surprisals_np * target_in_sequence * importance_weights_sequence.cpu().numpy()))
                    t14 = time.time()
                    if bootstrap:
                        summary_stats['surprisal_when_in_sequence'].append(self.bootstrap(seq_surprisals_np * target_in_sequence * importance_weights_sequence.cpu().numpy()))
                # --------------------------------------------------------------------------------------------------
                t15 = time.time()
                # --------------------------------------------------------------------------------------------------
                if importance_temp == 1:
                    scores['surprisal_when_first_token'].append(np.mean(seq_surprisals_np * target_is_first_token))
                    t16 = time.time()
                    if bootstrap:
                        summary_stats['surprisal_when_first_token'].append(self.bootstrap(seq_surprisals_np * target_is_first_token))
                else:
                    scores['surprisal_when_first_token'].append(np.mean(seq_surprisals_np * target_is_first_token * importance_weights_sequence.cpu().numpy()))
                    t16 = time.time()
                    if bootstrap:
                        summary_stats['surprisal_when_first_token'].append(self.bootstrap(seq_surprisals_np * target_is_first_token * importance_weights_sequence.cpu().numpy()))
    
                # --------------------------------------------------------------------------------------------------
                
                # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['surprisal_when_in_first_2_tokens'].append(np.mean(seq_surprisals_np * target_in_first_2_tokens))
                #     if bootstrap:
                #         summary_stats['surprisal_when_in_first_2_tokens'].append(self.bootstrap(seq_surprisals_np * target_in_first_2_tokens))
                # else:
                #     scores['surprisal_when_in_first_2_tokens'].append(np.mean(seq_surprisals_np * target_in_first_2_tokens * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['surprisal_when_in_first_2_tokens'].append(self.bootstrap(seq_surprisals_np * target_in_first_2_tokens * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t17 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['surprisal_when_in_first_3_tokens'].append(np.mean(seq_surprisals_np * target_in_first_3_tokens))
                #     if bootstrap:
                #         summary_stats['surprisal_when_in_first_3_tokens'].append(self.bootstrap(seq_surprisals_np * target_in_first_3_tokens))
                # else:
                #     scores['surprisal_when_in_first_3_tokens'].append(np.mean(seq_surprisals_np * target_in_first_3_tokens * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['surprisal_when_in_first_3_tokens'].append(self.bootstrap(seq_surprisals_np * target_in_first_3_tokens * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t18 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['surprisal_when_in_first_4_tokens'].append(np.mean(seq_surprisals_np * target_in_first_4_tokens))
                #     if bootstrap:
                #         summary_stats['surprisal_when_in_first_4_tokens'].append(self.bootstrap(seq_surprisals_np * target_in_first_4_tokens))
                # else:
                #     scores['surprisal_when_in_first_4_tokens'].append(np.mean(seq_surprisals_np * target_in_first_4_tokens * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['surprisal_when_in_first_4_tokens'].append(self.bootstrap(seq_surprisals_np * target_in_first_4_tokens * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t19 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['surprisal_when_in_first_5_tokens'].append(np.mean(seq_surprisals_np * target_in_first_5_tokens))
                #     if bootstrap:
                #         summary_stats['surprisal_when_in_first_5_tokens'].append(self.bootstrap(seq_surprisals_np * target_in_first_5_tokens))
                # else:
                #     scores['surprisal_when_in_first_5_tokens'].append(np.mean(seq_surprisals_np * target_in_first_5_tokens * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['surprisal_when_in_first_5_tokens'].append(self.bootstrap(seq_surprisals_np * target_in_first_5_tokens * importance_weights_sequence.cpu().numpy()))
                # # -------------------------------------------------------------------------------------------------- 
                t20 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['prob_when_in_sequence'].append(np.mean(seq_probs_np * target_in_sequence))
                #     if bootstrap:
                #         summary_stats['prob_when_in_sequence'].append(self.bootstrap(seq_probs_np * target_in_sequence))
                # else:
                #     scores['prob_when_in_sequence'].append(np.mean(seq_probs_np * target_in_sequence * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['prob_when_in_sequence'].append(self.bootstrap(seq_probs_np * target_in_sequence * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t21 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['prob_when_first_token'].append(np.mean(seq_probs_np * target_is_first_token))
                #     if bootstrap:
                #         summary_stats['prob_when_first_token'].append(self.bootstrap(seq_probs_np * target_is_first_token))
                # else:
                #     scores['prob_when_first_token'].append(np.mean(seq_probs_np * target_is_first_token * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['prob_when_first_token'].append(self.bootstrap(seq_probs_np * target_is_first_token * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t22 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['prob_when_in_first_2_tokens'].append(np.mean(seq_probs_np * target_in_first_2_tokens))
                #     if bootstrap:
                #         summary_stats['prob_when_in_first_2_tokens'].append(self.bootstrap(seq_probs_np * target_in_first_2_tokens))
                # else:
                #     scores['prob_when_in_first_2_tokens'].append(np.mean(seq_probs_np * target_in_first_2_tokens * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['prob_when_in_first_2_tokens'].append(self.bootstrap(seq_probs_np * target_in_first_2_tokens * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t23 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['prob_when_in_first_3_tokens'].append(np.mean(seq_probs_np * target_in_first_3_tokens))
                #     if bootstrap:
                #         summary_stats['prob_when_in_first_3_tokens'].append(self.bootstrap(seq_probs_np * target_in_first_3_tokens))
                # else:
                #     scores['prob_when_in_first_3_tokens'].append(np.mean(seq_probs_np * target_in_first_3_tokens * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['prob_when_in_first_3_tokens'].append(self.bootstrap(seq_probs_np * target_in_first_3_tokens * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t24 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['prob_when_in_first_4_tokens'].append(np.mean(seq_probs_np * target_in_first_4_tokens))
                #     if bootstrap:
                #         summary_stats['prob_when_in_first_4_tokens'].append(self.bootstrap(seq_probs_np * target_in_first_4_tokens))
                # else:
                #     scores['prob_when_in_first_4_tokens'].append(np.mean(seq_probs_np * target_in_first_4_tokens * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['prob_when_in_first_4_tokens'].append(self.bootstrap(seq_probs_np * target_in_first_4_tokens * importance_weights_sequence.cpu().numpy()))
                # # --------------------------------------------------------------------------------------------------
                # t25 = time.time()
                # # --------------------------------------------------------------------------------------------------
                # if importance_temp == 1:
                #     scores['prob_when_in_first_5_tokens'].append(np.mean(seq_probs_np * target_in_first_5_tokens))
                #     if bootstrap:
                #         summary_stats['prob_when_in_first_5_tokens'].append(self.bootstrap(seq_probs_np * target_in_first_5_tokens))
                # else:
                #     scores['prob_when_in_first_5_tokens'].append(np.mean(seq_probs_np * target_in_first_5_tokens * importance_weights_sequence.cpu().numpy()))
                #     if bootstrap:
                #         summary_stats['prob_when_in_first_5_tokens'].append(self.bootstrap(seq_probs_np * target_in_first_5_tokens * importance_weights_sequence.cpu().numpy()))
                # --------------------------------------------------------------------------------------------------
                t26 = time.time()
                # --------------------------------------------------------------------------------------------------
                if importance_temp == 1:
                    scores['sequence_entropy'].append(np.mean(seq_surprisals_np))
                    t27 = time.time()
                    if bootstrap:
                        summary_stats['sequence_entropy'].append(self.bootstrap(seq_surprisals_np))
                else:
                    scores['sequence_entropy'].append(np.mean(seq_surprisals_np * importance_weights_sequence.cpu().numpy()))
                    t27 = time.time()
                    if bootstrap:
                        summary_stats['sequence_entropy'].append(self.bootstrap(seq_surprisals_np * importance_weights_sequence.cpu().numpy()))
                # --------------------------------------------------------------------------------------------------
                t28 = time.time()
                # --------------------------------------------------------------------------------------------------
                if importance_temp == 1:
                    scores['sequence_expected_prob'].append(np.mean(seq_probs_np))
                    t29 = time.time()
                    if bootstrap:
                        summary_stats['sequence_expected_prob'].append(self.bootstrap(seq_probs_np))
                else:
                    scores['sequence_expected_prob'].append(np.mean(seq_probs_np * importance_weights_sequence.cpu().numpy()))
                    t29 = time.time()
                    if bootstrap:
                        summary_stats['sequence_expected_prob'].append(self.bootstrap(seq_probs_np * importance_weights_sequence.cpu().numpy()))
                # --------------------------------------------------------------------------------------------------
                
                
                # Collect runtimes
                runtimes['when_in_sequence'] = np.concatenate((runtimes['when_in_sequence'], np.array([t_shared+(t7-t6)+(t8-t7)])))
                runtimes['when_first_token'] = np.concatenate((runtimes['when_first_token'], np.array([t_shared+(t7-t6)+(t10-t9)])))
                # runtimes['when_in_first_2_tokens'] = np.concatenate((runtimes['when_in_first_2_tokens'], np.array([t_shared+(t7-t6)+(t10-t9)])))
                # runtimes['when_in_first_3_tokens'] = np.concatenate((runtimes['when_in_first_3_tokens'], np.array([t_shared+(t7-t6)+(t11-t10)])))
                # runtimes['when_in_first_4_tokens'] = np.concatenate((runtimes['when_in_first_4_tokens'], np.array([t_shared+(t7-t6)+(t12-t11)])))
                # runtimes['when_in_first_5_tokens'] = np.concatenate((runtimes['when_in_first_5_tokens'], np.array([t_shared+(t7-t6)+(t13-t12)])))
                runtimes['surprisal_when_in_sequence'] = np.concatenate((runtimes['surprisal_when_in_sequence'], np.array([t_shared+(t7-t6)+(t14-t13)])))
                runtimes['surprisal_when_first_token'] = np.concatenate((runtimes['surprisal_when_first_token'], np.array([t_shared+(t7-t6)+(t16-t15)])))
                # runtimes['surprisal_when_in_first_2_tokens'] = np.concatenate((runtimes['surprisal_when_in_first_2_tokens'], np.array([t_shared+(t7-t6)+(t17-t16)])))
                # runtimes['surprisal_when_in_first_3_tokens'] = np.concatenate((runtimes['surprisal_when_in_first_3_tokens'], np.array([t_shared+(t7-t6)+(t18-t17)])))
                # runtimes['surprisal_when_in_first_4_tokens'] = np.concatenate((runtimes['surprisal_when_in_first_4_tokens'], np.array([t_shared+(t7-t6)+(t19-t18)])))
                # runtimes['surprisal_when_in_first_5_tokens'] = np.concatenate((runtimes['surprisal_when_in_first_5_tokens'], np.array([t_shared+(t7-t6)+(t20-t19)])))
                # runtimes['prob_when_in_sequence'] = np.concatenate((runtimes['prob_when_in_sequence'], np.array([t_shared+(t7-t6)+(t21-t20)])))
                # runtimes['prob_when_first_token'] = np.concatenate((runtimes['prob_when_first_token'], np.array([t_shared+(t7-t6)+(t22-t21)])))
                # runtimes['prob_when_in_first_2_tokens'] = np.concatenate((runtimes['prob_when_in_first_2_tokens'], np.array([t_shared+(t7-t6)+(t23-t22)])))
                # runtimes['prob_when_in_first_3_tokens'] = np.concatenate((runtimes['prob_when_in_first_3_tokens'], np.array([t_shared+(t7-t6)+(t24-t23)])))
                # runtimes['prob_when_in_first_4_tokens'] = np.concatenate((runtimes['prob_when_in_first_4_tokens'], np.array([t_shared+(t7-t6)+(t25-t24)])))
                # runtimes['prob_when_in_first_5_tokens'] = np.concatenate((runtimes['prob_when_in_first_5_tokens'], np.array([t_shared+(t7-t6)+(t26-t25)])))
                runtimes['sequence_entropy'] = np.concatenate((runtimes['sequence_entropy'], np.array([t_shared+(t7-t6)+(t27-t26)])))
                runtimes['sequence_expected_prob'] = np.concatenate((runtimes['sequence_expected_prob'], np.array([t_shared+(t7-t6)+(t29-t28)])))

            input_tokenized['offset_mapping'].extend([(i + start_ind, j + start_ind) for i, j in encodings['offset_mapping'][offset:]])
            input_tokenized["input_ids"].extend(encodings['input_ids'][offset:])
            if encodings['offset_mapping'][-1][1] + start_ind == len(input):
                break
            start_ind += encodings['offset_mapping'][-STRIDE][1]
        
        if not bootstrap:
            summary_stats = None

        return scores, input_tokenized, runtimes, summary_stats
    

    def token_score(
        self,
        input: str,
        n_samples: int,
        self_contextualisation_layer: int=-1,
        contextualisation_layer: int=-1,
        max_new_tokens: int=1,
        importance_temp: float=1.0,
        return_tokens: Optional[bool]=False,
        bootstrap: bool=False
    ):
        scores, input_tokenized, runtimes, summary_stats = self._score(input, n_samples, self_contextualisation_layer, contextualisation_layer, max_new_tokens, importance_temp, bootstrap)
        rdict = {score_name: np.array(scores[score_name]) for score_name in scores}  
        if return_tokens:
            next_tokens = self.tokenizer.convert_ids_to_tokens(input_tokenized['input_ids'])
            assert len(next_tokens) == len(rdict['decontextualised'])
            rdict['tokens'] = next_tokens
        
        return rdict, runtimes, summary_stats
    

    def word_score(
        self,
        input: str,
        n_samples: int,
        self_contextualisation_layer: int=-1,
        contextualisation_layer: int=-1,
        max_new_tokens: int=1,
        importance_temp: float=1.0,
        return_tokens: Optional[bool]=False,
        bootstrap: Optional[bool]=False
    ):  
        scores, input_tokenized, runtimes, summary_stats = self._score(input, n_samples, self_contextualisation_layer, contextualisation_layer, max_new_tokens, importance_temp, bootstrap)

        offsets = input_tokenized['offset_mapping']

        rdict = {
            'decontextualised': np.array(utils.aggregate_score_by_word(input, scores['decontextualised'], offsets, mode='sum')),
            # 'self_contextualised': np.array(utils.aggregate_score_by_word(input, scores['self_contextualised'], offsets, mode='sum')),
            # 'contextualised': np.array(utils.aggregate_score_by_word(input, scores['contextualised'], offsets, mode='sum')),
            'expected_decontextualised': np.array(utils.aggregate_score_by_word(input, scores['expected_decontextualised'], offsets, mode='first')),
            # 'expected_self_contextualised': np.array(utils.aggregate_score_by_word(input, scores['expected_self_contextualised'], offsets, mode='first')),
            # 'expected_contextualised': np.array(utils.aggregate_score_by_word(input, scores['expected_contextualised'], offsets, mode='first')),
        }
        # if max_new_tokens > 1:
        rdict['expected_seq_decontextualised'] = np.array(utils.aggregate_score_by_word(input, scores['expected_seq_decontextualised'], offsets, mode='first'))
        # rdict['expected_seq_self_contextualised'] = np.array(utils.aggregate_score_by_word(input, scores['expected_seq_self_contextualised'], offsets, mode='first'))
        # rdict['expected_seq_contextualised'] = np.array(utils.aggregate_score_by_word(input, scores['expected_seq_contextualised'], offsets, mode='first'))
        rdict['when_in_sequence'] = np.array(utils.aggregate_score_by_word(input, scores['when_in_sequence'], offsets, mode='sum'))
        rdict['when_first_token'] = np.array(utils.aggregate_score_by_word(input, scores['when_first_token'], offsets, mode='sum'))
        # rdict['when_in_first_2_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['when_in_first_2_tokens'], offsets, mode='sum'))
        # rdict['when_in_first_3_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['when_in_first_3_tokens'], offsets, mode='sum'))
        # rdict['when_in_first_4_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['when_in_first_4_tokens'], offsets, mode='sum'))
        # rdict['when_in_first_5_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['when_in_first_5_tokens'], offsets, mode='sum'))
        rdict['surprisal_when_in_sequence'] = np.array(utils.aggregate_score_by_word(input, scores['surprisal_when_in_sequence'], offsets, mode='sum'))
        rdict['surprisal_when_first_token'] = np.array(utils.aggregate_score_by_word(input, scores['surprisal_when_first_token'], offsets, mode='sum'))
        # rdict['surprisal_when_in_first_2_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['surprisal_when_in_first_2_tokens'], offsets, mode='sum'))
        # rdict['surprisal_when_in_first_3_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['surprisal_when_in_first_3_tokens'], offsets, mode='sum'))
        # rdict['surprisal_when_in_first_4_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['surprisal_when_in_first_4_tokens'], offsets, mode='sum'))
        # rdict['surprisal_when_in_first_5_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['surprisal_when_in_first_5_tokens'], offsets, mode='sum'))
        # rdict['prob_when_in_sequence'] = np.array(utils.aggregate_score_by_word(input, scores['prob_when_in_sequence'], offsets, mode='multiply'))
        # rdict['prob_when_first_token'] = np.array(utils.aggregate_score_by_word(input, scores['prob_when_first_token'], offsets, mode='multiply'))
        # rdict['prob_when_in_first_2_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['prob_when_in_first_2_tokens'], offsets, mode='multiply'))
        # rdict['prob_when_in_first_3_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['prob_when_in_first_3_tokens'], offsets, mode='multiply'))
        # rdict['prob_when_in_first_4_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['prob_when_in_first_4_tokens'], offsets, mode='multiply'))
        # rdict['prob_when_in_first_5_tokens'] = np.array(utils.aggregate_score_by_word(input, scores['prob_when_in_first_5_tokens'], offsets, mode='multiply'))
        rdict['sequence_entropy'] = np.array(utils.aggregate_score_by_word(input, scores['sequence_entropy'], offsets, mode='first'))
        rdict['sequence_expected_prob'] = np.array(utils.aggregate_score_by_word(input, scores['sequence_expected_prob'], offsets, mode='first'))

        if return_tokens:
            next_tokens = utils.aggregate_score_by_word(input, input_tokenized["input_ids"], offsets, mode='string', tokenizer=self.tokenizer)
            assert len(next_tokens) == len(rdict['decontextualised'])
            rdict['tokens'] = next_tokens

        return rdict, runtimes, summary_stats
        

    def compute_token_distances(self, alternative_ids, context_ids, target_id, self_contextualisation_layer, contextualisation_layer, importance_weights=None, bootstrap=False):
        """
        Compute mean distance between each input target and its token-level alternatives
        """

        alternative_ids_copy = copy.copy(alternative_ids)

        # create a batch of context + target|alternatives token ids
        decontextualised_embeds = torch.empty(0, self.embed_size, device=self.device)
        # contextualised_embeds = torch.empty(0, self.embed_size, device=self.device)
        # self_contextualised_embeds = torch.empty(0, self.embed_size, device=self.device)
        
        # sample encoded in context
        # t1 = time.time()
        # samples_left = len(alternative_ids)
        # while samples_left:
        #     cur_samples = min(samples_left, self.bsize_dist)
        #     input_ids = []
        #     for j in range(cur_samples):
        #         input_ids.append(torch.cat([context_ids,torch.tensor([alternative_ids.pop()],dtype=torch.int32, device=self.device)]))
        #     if samples_left - cur_samples == 0:
        #         input_ids.append(torch.cat([context_ids, torch.tensor([target_id],dtype=torch.int32, device=self.device)]))
        #     input_ids = torch.stack(input_ids)

        #     # forward pass
        #     with torch.no_grad():
        #         outputs = self.model(input_ids, output_hidden_states=True)
        
        #     # collect hidden states
        #     contextualised_embeds = torch.cat([contextualised_embeds, outputs.hidden_states[contextualisation_layer][:, -1, :]])
        #     samples_left -= cur_samples
        
        # sample encoded out of context 
        t2 = time.time()
        samples_left = len(alternative_ids_copy)
        while samples_left:
            cur_samples = min(samples_left, self.bsize_dist)
            input_ids = []
            for j in range(cur_samples):
                input_ids.append(alternative_ids_copy.pop().unsqueeze(0))
            if samples_left - cur_samples == 0:
                input_ids.append(target_id.unsqueeze(0))
            input_ids = torch.stack(input_ids)

            # forward pass
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
        
            # collect hidden states
            decontextualised_embeds =  torch.cat([decontextualised_embeds, outputs.hidden_states[0][:, -1, :]])
            # self_contextualised_embeds = torch.cat([self_contextualised_embeds, outputs.hidden_states[self_contextualisation_layer][:, -1, :]])
            samples_left -= cur_samples
        t3 = time.time()

        # separate target from alternative embeddings
        decontextualised_embeds_alternatives, decontextualised_embeds_target = decontextualised_embeds[:-1], decontextualised_embeds[-1]
        # contextualised_embeds_alternatives, contextualised_embeds_target = contextualised_embeds[:-1], contextualised_embeds[-1]
        # self_contextualised_embeds_alternatives, self_contextualised_embeds_target = self_contextualised_embeds[:-1], self_contextualised_embeds[-1]
        
        # measure cosine distance between alternatives and target
        t4 = time.time()
        # --------------------------------------------------------------------------------------------------
        if importance_weights is not None:
            samples = (1 - torch.nn.functional.cosine_similarity(decontextualised_embeds_alternatives, decontextualised_embeds_target)) * importance_weights
        else:
            samples = 1 - torch.nn.functional.cosine_similarity(decontextualised_embeds_alternatives, decontextualised_embeds_target)
        decontextualised = samples.mean()
        t5 = time.time()
        if bootstrap:
            decontextualised_stats = self.bootstrap(samples)
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------
        # if importance_weights is not None:
        #     samples = 1 - (torch.nn.functional.cosine_similarity(contextualised_embeds_alternatives, contextualised_embeds_target) * importance_weights)
        # else:
        #     samples = 1 - torch.nn.functional.cosine_similarity(contextualised_embeds_alternatives, contextualised_embeds_target)
        # contextualised = samples.mean()
        # if bootstrap:
        #     contextualised_stats = self.bootstrap(samples)
        # # --------------------------------------------------------------------------------------------------
        # t6 = time.time()
        # if importance_weights is not None:
        #     samples = 1 - (torch.nn.functional.cosine_similarity(self_contextualised_embeds_alternatives, self_contextualised_embeds_target)* importance_weights)
        # else:
        #     samples = 1 - torch.nn.functional.cosine_similarity(self_contextualised_embeds_alternatives, self_contextualised_embeds_target)
        # self_contextualised = samples.mean()
        # if bootstrap:
        #     self_contextualised_stats = self.bootstrap(samples)
        # # --------------------------------------------------------------------------------------------------
        
        # measure cosine distance alternatives  (as self-variability; exclude similarities between same alternatives)
        if importance_weights is not None:
            importance_weights = importance_weights.cpu().numpy()
        t7 = time.time()
        # --------------------------------------------------------------------------------------------------
        pairwise_distances_dectx = spatial.distance.cdist(decontextualised_embeds_alternatives.cpu(), decontextualised_embeds_alternatives.cpu(), metric='cosine')
        np.fill_diagonal(pairwise_distances_dectx, np.nan)
        if importance_weights is not None:
            expected_decontextualised = np.mean(
                np.nanmean(pairwise_distances_dectx, axis=1) * importance_weights
            )
            t8 = time.time()
            if bootstrap:
                expected_decontextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_dectx, axis=1) * importance_weights)
        else:
            expected_decontextualised = np.nanmean(pairwise_distances_dectx)
            t8 = time.time()
            if bootstrap:
                expected_decontextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_dectx, axis=1))
        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------
        # pairwise_distances_ctx = spatial.distance.cdist(contextualised_embeds_alternatives.cpu(), contextualised_embeds_alternatives.cpu(), metric='cosine')
        # np.fill_diagonal(pairwise_distances_ctx, np.nan)
        # if importance_weights is not None:
        #     expected_contextualised = np.mean(
        #         np.nanmean(pairwise_distances_ctx, axis=1) * importance_weights
        #     )
        #     if bootstrap:
        #         expected_contextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_ctx, axis=1) * importance_weights)
        # else:
        #     expected_contextualised = np.nanmean(pairwise_distances_ctx)
        #     if bootstrap:
        #         expected_contextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_ctx, axis=1))
        # # --------------------------------------------------------------------------------------------------
        # t9 = time.time()
        # # -------------------------------------------------------------------------------------------------- 
        # pairwise_distances_selfctx = spatial.distance.cdist(self_contextualised_embeds_alternatives.cpu(), self_contextualised_embeds_alternatives.cpu(), metric='cosine')
        # np.fill_diagonal(pairwise_distances_selfctx, np.nan)
        # if importance_weights is not None:
        #     expected_self_contextualised = np.mean(
        #         np.nanmean(pairwise_distances_selfctx, axis=1) * importance_weights
        #     )
        #     if bootstrap:
        #         expected_self_contextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_selfctx, axis=1) * importance_weights)
        # else:
        #     expected_self_contextualised = np.nanmean(pairwise_distances_selfctx)
        #     if bootstrap:
        #         expected_self_contextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_selfctx, axis=1))
        # --------------------------------------------------------------------------------------------------

        if not bootstrap:
            # return (decontextualised.cpu(), (t3-t2)+(t5-t4)) , (self_contextualised.cpu(), (t3-t2)+(t7-t6)), (contextualised.cpu(), (t2-t1)+(t6-t5)), (expected_decontextualised, (t3-t2)+(t8-t7)), (expected_self_contextualised, (t3-t2)+(t10-t9)), (expected_contextualised, (t2-t1)+(t9-t8))
            return (decontextualised.cpu(), (t3-t2)+(t5-t4)), (expected_decontextualised, (t3-t2)+(t8-t7))

        else:
            # return (decontextualised.cpu(), (t3-t2)+(t5-t4), decontextualised_stats)#, (self_contextualised.cpu(), (t3-t2)+(t7-t6), self_contextualised_stats), (contextualised.cpu(), (t2-t1)+(t6-t5), contextualised_stats), (expected_decontextualised, (t3-t2)+(t8-t7), expected_decontextualised_stats), (expected_self_contextualised, (t3-t2)+(t10-t9), expected_self_contextualised_stats), (expected_contextualised, (t2-t1)+(t9-t8), expected_contextualised_stats)
            return (decontextualised.cpu(), (t3-t2)+(t5-t4), decontextualised_stats), (expected_decontextualised, (t3-t2)+(t8-t7), expected_decontextualised_stats)


    def compute_expected_sequence_distances(self, alternative_ids, context_ids, self_contextualisation_layer, contextualisation_layer, importance_weights=None, bootstrap=False):
        """
        Compute mean distance between two sets of sequence-level alternatives
        """

        alternative_ids_copy = copy.copy(alternative_ids)

        # create a batch of context + target|alternatives token ids
        decontextualised_embeds = torch.empty(0, self.embed_size, device=self.device)
        # contextualised_embeds = torch.empty(0, self.embed_size, device=self.device)
        # self_contextualised_embeds = torch.empty(0, self.embed_size, device=self.device)
        
        # # sample encoded in context
        # t1 = time.time()
        # samples_left = len(alternative_ids)
        # while samples_left:
        #     cur_samples = min(samples_left, self.bsize_dist)
        #     input_ids = []
        #     attention_masks = []
        #     for _ in range(cur_samples):
        #         sample_input_ids = torch.cat([context_ids, alternative_ids.pop()])
        #         input_ids.append(sample_input_ids)
        #         mask = torch.ones_like(sample_input_ids)
        #         mask[sample_input_ids == self.tokenizer.eos_token_id] = 0
        #         attention_masks.append(mask)
        #     input_ids = torch.stack(input_ids)
        #     attention_masks = torch.stack(attention_masks)

        #     # forward pass
        #     with torch.no_grad():
        #         outputs = self.model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        
        #     cur_ctx_embeds = self.mean_pooling(
        #         outputs.hidden_states[contextualisation_layer][:, context_ids.shape[0]:, :], 
        #         attention_masks[:, context_ids.shape[0]:]
        #     )
        #     contextualised_embeds = torch.cat([contextualised_embeds, cur_ctx_embeds])
        #     samples_left -= cur_samples
        
        # sample encoded out of context 
        t2 = time.time()
        samples_left = len(alternative_ids_copy)
        while samples_left:
            cur_samples = min(samples_left, self.bsize_dist)
            input_ids = []
            attention_masks = []
            for _ in range(cur_samples):
                sample_input_ids = alternative_ids_copy.pop()
                input_ids.append(sample_input_ids)
                mask = torch.ones_like(sample_input_ids)
                mask[sample_input_ids == self.tokenizer.eos_token_id] = 0
                attention_masks.append(mask)
            input_ids = torch.stack(input_ids)
            attention_masks = torch.stack(attention_masks)

            # forward pass
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
            
            # collect hidden states
            cur_dec_embeds = self.mean_pooling(
                outputs.hidden_states[0], 
                attention_masks,
            )
            # cur_selfctx_embeds = self.mean_pooling(
            #     outputs.hidden_states[self_contextualisation_layer], 
            #     attention_masks
            # )
            decontextualised_embeds =  torch.cat([decontextualised_embeds, cur_dec_embeds])
            # self_contextualised_embeds = torch.cat([self_contextualised_embeds, cur_selfctx_embeds])
            samples_left -= cur_samples

        # measure cosine distance between alternatives
        if importance_weights is not None:
            importance_weights = importance_weights.cpu().numpy()

        t3 = time.time()
        # --------------------------------------------------------------------------------------------------
        pairwise_distances_dectx = spatial.distance.cdist(decontextualised_embeds.cpu(), decontextualised_embeds.cpu(), metric='cosine')
        np.fill_diagonal(pairwise_distances_dectx, np.nan)
        if importance_weights is not None:
            expected_decontextualised = np.mean(
                np.nanmean(pairwise_distances_dectx, axis=1) * importance_weights
            )
            t4 = time.time()
            if bootstrap:
                expected_decontextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_dectx, axis=1) * importance_weights)
        else:
            expected_decontextualised = np.nanmean(pairwise_distances_dectx)
            t4 = time.time()
            if bootstrap:
                expected_decontextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_dectx, axis=1))
        # --------------------------------------------------------------------------------------------------
        # # --------------------------------------------------------------------------------------------------
        # pairwise_distances_ctx = spatial.distance.cdist(contextualised_embeds.cpu(), contextualised_embeds.cpu(), metric='cosine')
        # np.fill_diagonal(pairwise_distances_ctx, np.nan)
        # if importance_weights is not None:
        #     expected_contextualised = np.mean(
        #         np.nanmean(pairwise_distances_ctx, axis=1) * importance_weights
        #     )
        #     if bootstrap:
        #         expected_contextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_ctx, axis=1) * importance_weights)
        # else:
        #     expected_contextualised = np.nanmean(pairwise_distances_ctx)
        #     if bootstrap:
        #         expected_contextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_ctx, axis=1))
        # # --------------------------------------------------------------------------------------------------
        # t5 = time.time()
        # # --------------------------------------------------------------------------------------------------
        # pairwise_distances_selfctx = spatial.distance.cdist(self_contextualised_embeds.cpu(), self_contextualised_embeds.cpu(), metric='cosine')
        # np.fill_diagonal(pairwise_distances_selfctx, np.nan)
        # if importance_weights is not None:
        #     expected_self_contextualised = np.mean(
        #         np.nanmean(pairwise_distances_selfctx, axis=1) * importance_weights
        #     )
        #     if bootstrap:
        #         expected_self_contextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_selfctx, axis=1) * importance_weights)
        # else:
        #     expected_self_contextualised = np.nanmean(pairwise_distances_selfctx)
        #     if bootstrap:
        #         expected_self_contextualised_stats = self.bootstrap(np.nanmean(pairwise_distances_selfctx, axis=1))
        # # --------------------------------------------------------------------------------------------------
        # t6 = time.time()
        
        if not bootstrap:
            return (expected_decontextualised, (t3-t2)+(t4-t3))  #, (expected_self_contextualised, (t3-t2)+(t6-t5)), (expected_contextualised, (t2-t1)+(t5-t4))
        else:   
            return (expected_decontextualised, (t3-t2)+(t4-t3), expected_decontextualised_stats)  #, (expected_self_contextualised, (t3-t2)+(t6-t5), expected_self_contextualised_stats), (expected_contextualised, (t2-t1)+(t5-t4), expected_contextualised_stats)
    

    def mean_pooling(self, hidden_states, attention_masks): 
        return (hidden_states * attention_masks.unsqueeze(-1)).sum(dim=1) / attention_masks.sum(dim=1).unsqueeze(-1)
    
    def bootstrap(self, scores, n_boot=100, return_resamples=True):
        ''''
        return bootstrap estimates for the mean and variance
        '''
        if type(scores) is torch.Tensor:
            scores = scores.cpu().numpy()
        elif type(scores) is list:
            scores = np.array(scores)
        metric_values = np.random.choice(scores, size = (n_boot, len(scores))).mean(axis=1)
        mean = metric_values.mean()
        variance = metric_values.var()
        # if mean < 0:
        #     print(mean)
        if return_resamples:
            return mean, variance, metric_values
        else:
            return mean, variance


class MCProbabilityScorer(Scorer):

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = "cpu",
        batch_size: Optional[int] = 256,
        seed: Optional[int] = 0,
    ):
        """
        :param model_name_or_path: the name or path to a model compatible with AutoModelWithLMHead
        :param device: "cpu" or "cuda"
        """
        super().__init__(model_name_or_path, device)
        self.batch_size = batch_size
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def _score(
        self,
        input: str,
        n_samples: int,
        max_new_tokens: int=5,
    ):
        if max_new_tokens <= 1:
            raise ValueError("max_new_tokens must be greater than 1")
        
        if n_samples % self.batch_size != 0:
            raise ValueError("n_samples must be divisible by batch_size")
        
        scores = defaultdict(list)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        STRIDE = 200
        input_tokenized = defaultdict(list)
        start_ind = 0

        while True:
            encodings = self.tokenizer(input[start_ind:], max_length=self.max_seq_len-1, truncation=True, return_attention_mask=False, return_offsets_mapping=True)
            tensor_input = torch.tensor([self.tokenizer.bos_token_id] + encodings['input_ids'], device=self.device)
            offset = 0 if start_ind == 0 else STRIDE - 1
            for t in range(1, len(tensor_input)):
                if t <= offset:
                    continue
                context_ids = tensor_input[:t]

                target_in_sequence = []
                target_is_first_token = []
                target_in_first_2_tokens, target_in_first_3_tokens, target_in_first_4_tokens, target_in_first_5_tokens = [], [], [], []
        

                for _ in range(n_samples // self.batch_size):
                    outputs = self.model.generate(
                        context_ids.unsqueeze(0),
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        num_return_sequences=self.batch_size,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_logits=True,
                    )
                
                    # check whether the target token is in the sequence or if it is among the first n token
                    # --------------------------------------------------------------------------------------------------
                    for seq in outputs.sequences:
                        target_in_sequence.append(tensor_input[t] in seq[t:])
                        target_is_first_token.append((tensor_input[t] == seq[t]).item())
                        target_in_first_2_tokens.append(tensor_input[t] in seq[t:t+2])
                        target_in_first_3_tokens.append(tensor_input[t] in seq[t:t+3])
                        target_in_first_4_tokens.append(tensor_input[t] in seq[t:t+4])
                        target_in_first_5_tokens.append(tensor_input[t] in seq[t:t+5])

                # transform into numpy arrays
                target_in_sequence = np.array(target_in_sequence)
                target_is_first_token = np.array(target_is_first_token)
                target_in_first_2_tokens = np.array(target_in_first_2_tokens)
                target_in_first_3_tokens = np.array(target_in_first_3_tokens)
                target_in_first_4_tokens = np.array(target_in_first_4_tokens)
                target_in_first_5_tokens = np.array(target_in_first_5_tokens)
                
                scores['when_in_sequence'].append(np.mean(target_in_sequence))
                scores['when_first_token'].append(np.mean(target_is_first_token))
                scores['when_in_first_2_tokens'].append(np.mean(target_in_first_2_tokens))
                scores['when_in_first_3_tokens'].append(np.mean(target_in_first_3_tokens))
                scores['when_in_first_4_tokens'].append(np.mean(target_in_first_4_tokens))
                scores['when_in_first_5_tokens'].append(np.mean(target_in_first_5_tokens))

            input_tokenized['offset_mapping'].extend([(i + start_ind, j + start_ind) for i, j in encodings['offset_mapping'][offset:]])
            input_tokenized["input_ids"].extend(encodings['input_ids'][offset:])
            if encodings['offset_mapping'][-1][1] + start_ind == len(input):
                break
            start_ind += encodings['offset_mapping'][-STRIDE][1]

        return scores, input_tokenized
    

    def token_score(
        self,
        input: str,
        n_samples: int,
        max_new_tokens: int=1,
        return_tokens: Optional[bool]=False,
    ):
        scores, input_tokenized = self._score(input, n_samples, max_new_tokens)
        rdict = {score_name: np.array(scores[score_name]) for score_name in scores}  
        if return_tokens:
            next_tokens = self.tokenizer.convert_ids_to_tokens(input_tokenized['input_ids'])
            assert len(next_tokens) == len(rdict['when_first_token'])
            rdict['tokens'] = next_tokens
        
        return rdict
    

    def word_score(
        self,
        input: str,
        n_samples: int,
        max_new_tokens: int=1,
        return_tokens: Optional[bool]=False
    ):  
        scores, input_tokenized = self._score(input, n_samples, max_new_tokens)
        offsets = input_tokenized['offset_mapping']
        rdict = {
            score_name: np.array(utils.aggregate_score_by_word(input, scores[score_name], offsets, mode='sum')) for score_name in scores
        }
        if return_tokens:
            next_tokens = utils.aggregate_score_by_word(input, input_tokenized["input_ids"], offsets, mode='string', tokenizer=self.tokenizer)
            assert len(next_tokens) == len(rdict['when_first_token'])
            rdict['tokens'] = next_tokens

        return rdict
        