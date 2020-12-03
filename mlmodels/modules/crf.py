# -*- coding: utf-8 -*-
"""
Created on 2020-02-24
@author: duytinvo
"""
import torch
import torch.nn as nn


class NN_CRF(nn.Module):
    """
    Set se_words = True before using CRF
    """
    def __init__(self,  crf_HPs):
        super(NN_CRF, self).__init__()
        _, num_labels, se_transitions = crf_HPs
        self.se_transitions = se_transitions
        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.empty(num_labels, num_labels))
        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1) and end of sequence tags (num_tags + 2)
        constraint_mask = torch.empty(num_labels + 2, num_labels + 2).fill_(1.)
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        if self.se_transitions:
            # Also need logits for transitioning from "start" state and to "end" state.
            self.start_transitions = torch.nn.Parameter(torch.empty(num_labels))
            self.end_transitions = torch.nn.Parameter(torch.empty(num_labels))

        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)
        if self.se_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, input_tensors, mask_tensors):
        batch_size, sequence_length, num_tags = input_tensors.size()
        # mask_tensors: (seq_len, batch_size, num_tags)
        mask_tensors = mask_tensors.float().transpose(1, 0).contiguous()
        # input_tensors: (seq_len, batch_size, num_tags)
        input_tensors = input_tensors.transpose(1, 0).contiguous()
        # alpha: (batch_size, num_tags)
        if self.se_transitions:
            alpha = self.start_transitions.view(1, num_tags) + input_tensors[0]
        else:
            alpha = input_tensors[0]
        # alpha = NN_CRF.logsumexp(alpha0) * mask_tensors[0].view(batch_size, 1)
        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along the current_tag axis.
            # emit_scores: (batch_size, 1, num_tags)
            emit_scores = input_tensors[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along the instance axis.
            # transition_scores: (1, num_tags, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            # broadcast_alpha: (batch_size, num_tags, 1)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis
            # inner: (batch_size, num_tags, num_tags)
            inner = broadcast_alpha + emit_scores + transition_scores
            # In valid positions (mask == 1) we want to take the logsumexp over the current_tag dimension
            # of ``inner``. Otherwise (mask == 0) we want to retain the previous alpha.
            # alpha: (batch_size, num_tags)
            alpha = (NN_CRF.logsumexp(inner, 1) * mask_tensors[i].view(batch_size, 1) +
                     alpha * (1 - mask_tensors[i]).view(batch_size, 1))

        # Every sequence needs to end with a transition to the stop_tag.
        if self.se_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha
        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return NN_CRF.logsumexp(stops)

    def _joint_likelihood(self, input_tensors, label_tensors, mask_tensors):
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)
        """
        batch_size, sequence_length, _ = input_tensors.size()
        # Transpose batch size and sequence dimensions:
        input_tensors = input_tensors.transpose(0, 1).contiguous()
        mask_tensors = mask_tensors.float().transpose(0, 1).contiguous()
        label_tensors = label_tensors.transpose(0, 1).contiguous()
        # Start with the transition scores from start_tag to the first tag in each input
        if self.se_transitions:
            score = self.start_transitions.index_select(0, label_tensors[0])
        else:
            score = 0.0
        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = label_tensors[i], label_tensors[i+1]
            # The scores for transitioning from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            # The score for using current_tag
            emit_score = input_tensors[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * mask_tensors[i + 1] + emit_score * mask_tensors[i]
        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask_tensors.sum(0).long() - 1
        last_tags = label_tensors.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.se_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0
        # Add the last input if it's not masked.
        last_inputs = input_tensors[-1]                                         # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))         # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                           # (batch_size,)

        score = score + last_transition_score + last_input_score * mask_tensors[-1]
        return score

    def NLL_loss(self, input_tensors, label_tensors, mask_tensors):
        """
        Computes the log likelihood.
        """
        # batch_size, _, _ = input_tensors.size()
        log_denominator = self._input_likelihood(input_tensors, mask_tensors)
        log_numerator = self._joint_likelihood(input_tensors, label_tensors, mask_tensors)
        return torch.sum(log_denominator - log_numerator)/mask_tensors.sum().float()

    def inference(self, input_tensors, mask_tensors):
        best_paths = self.viterbi_tags(input_tensors, mask_tensors)
        return best_paths

    def viterbi_tags(self, input_tensors, mask_tensors):
        """
        Uses viterbi algorithm to find most likely tags for the given inputs.
        If constraints are applied, disallows all other transitions.
        """
        device = input_tensors.device
        _, max_seq_length, num_tags = input_tensors.size()

        # Get the tensors out of the variables
        # input_tensors ~ mask_tensors: [batch_size, seq_length, num_tags]
        input_tensors, mask_tensors = input_tensors.data, mask_tensors.data
        transitions = torch.empty(num_tags + 2, num_tags + 2, device=device).fill_(-10000.)

        # Apply transition constraints
        constrained_transitions = \
            (self.transitions * self._constraint_mask[:num_tags, :num_tags] +
             -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags]))
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        if self.se_transitions:
            transitions[start_tag, :num_tags] = \
                (self.start_transitions.detach() * self._constraint_mask[start_tag, :num_tags].data +
                 -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = \
                (self.end_transitions.detach() * self._constraint_mask[:num_tags, end_tag].data +
                 -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach()))
        else:
            transitions[start_tag, :num_tags] = (-10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach()))
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.empty(max_seq_length + 2, num_tags + 2, device=device)

        for prediction, prediction_mask in zip(input_tensors, mask_tensors):
            sequence_length = torch.sum(prediction_mask)
            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.

            # We pass the tags and the transitions to ``viterbi_decode``.
            viterbi_path, viterbi_score = NN_CRF.viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            # Get rid of START and END sentinels and append.
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score.item()))
        return best_paths

    # Compute log sum exp in a numerically stable way for the forward algorithm
    @staticmethod
    def logsumexp(tensor, dim=-1, keepdim=False):
        """
        A numerically stable computation of logsumexp. This is mathematically equivalent to
        `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
        probabilities.
        Parameters
        ----------
        tensor : torch.FloatTensor, required.
            A tensor of arbitrary size.
        dim : int, optional (default = -1)
            The dimension of the tensor to apply the logsumexp to.
        keepdim: bool, optional (default = False)
            Whether to retain a dimension of size one at the dimension we reduce over.
        """
        max_score, _ = tensor.max(dim, keepdim=keepdim)
        if keepdim:
            stable_vec = tensor - max_score
        else:
            stable_vec = tensor - max_score.unsqueeze(dim)
        return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

    @staticmethod
    def viterbi_decode(tag_sequence, transitions):
        """
        Perform Viterbi decoding in log space over a sequence given a transition matrix
        specifying pairwise (transition) potentials between tags and a matrix of shape
        (sequence_length, num_tags) specifying unary potentials for possible tags per
        timestep.
        Parameters
        ----------
        tag_sequence : torch.Tensor, required.
            A tensor of shape (sequence_length, num_tags) representing scores for
            a set of tags over a given sequence.
        transitions : torch.Tensor, required.
            A tensor of shape (num_tags, num_tags) representing the binary potentials
            for transitioning between a given pair of tags.

        Returns
        -------
        viterbi_path : List[int]
            The tag indices of the maximum likelihood tag sequence.
        viterbi_score : torch.Tensor
            The score of the viterbi path.
        """
        sequence_length, num_tags = list(tag_sequence.size())
        path_scores = []
        path_indices = []
        path_scores.append(tag_sequence[0, :])

        # Evaluate the scores for all possible paths.
        for timestep in range(1, sequence_length):
            # Add pairwise potentials to current scores.
            summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transitions
            scores, paths = torch.max(summed_potentials, 0)
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
            path_indices.append(paths.squeeze())

        # Construct the most likely sequence backwards.
        viterbi_score, best_path = torch.max(path_scores[-1], 0)
        viterbi_path = [int(best_path.cpu().numpy())]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()
        return viterbi_path, viterbi_score
