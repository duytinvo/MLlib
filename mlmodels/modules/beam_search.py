# -*- coding: utf-8 -*-
"""
Created on 2019-12-12
@author: duytinvo
"""
import warnings
import torch
import numpy as np


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


class PenaltyBuilder(object):
    """Returns the Length and Coverage Penalty function for Beam Search.

    Args:
        length_pen (str): option name of length pen: None/wu/avg
        cov_pen (str): option name of cov pen: None/wu/summary

    Attributes:
        has_cov_pen (bool): Whether coverage penalty is None (applying it
            is a no-op). Note that the converse isn't true. Setting beta
            to 0 should force coverage length to be a no-op.
        has_len_pen (bool): Whether length penalty is None (applying it
            is a no-op). Note that the converse isn't true. Setting alpha
            to 1 should force length penalty to be a no-op.
        coverage_penalty (callable[[FloatTensor, float], FloatTensor]):
            Calculates the coverage penalty.
        length_penalty (callable[[int, float], float]): Calculates
            the length penalty.
    """

    def __init__(self, cov_pen, length_pen):
        self.has_cov_pen = not self._pen_is_none(cov_pen)
        self.coverage_penalty = self._coverage_penalty(cov_pen)
        self.has_len_pen = not self._pen_is_none(length_pen)
        self.length_penalty = self._length_penalty(length_pen)

    @staticmethod
    def _pen_is_none(pen):
        return pen == "none" or pen is None

    def _coverage_penalty(self, cov_pen):
        if cov_pen == "wu":
            return self.coverage_wu
        elif cov_pen == "summary":
            return self.coverage_summary
        elif self._pen_is_none(cov_pen):
            return self.coverage_none
        else:
            raise NotImplementedError("No '{:s}' coverage penalty.".format(
                cov_pen))

    def _length_penalty(self, length_pen):
        if length_pen == "wu":
            return self.length_wu
        elif length_pen == "avg":
            return self.length_average
        elif self._pen_is_none(length_pen):
            return self.length_none
        else:
            raise NotImplementedError("No '{:s}' length penalty.".format(
                length_pen))

    # Below are all the different penalty terms implemented so far.
    # Subtract coverage penalty from topk log probs.
    # Divide topk log probs by length penalty.

    def coverage_wu(self, cov, beta=0.):
        """GNMT coverage re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        ``cov`` is expected to be sized ``(*, seq_len)``, where ``*`` is
        probably ``batch_size x beam_size`` but could be several
        dimensions like ``(batch_size, beam_size)``. If ``cov`` is attention,
        then the ``seq_len`` axis probably sums to (almost) 1.
        """

        penalty = -torch.min(cov, cov.clone().fill_(1.0)).log().sum(-1)
        return beta * penalty

    def coverage_summary(self, cov, beta=0.):
        """Our summary penalty."""
        penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(-1)
        penalty -= cov.size(-1)
        return beta * penalty

    def coverage_none(self, cov, beta=0.):
        """Returns zero as penalty"""
        none = torch.zeros((1,), device=cov.device,
                           dtype=torch.float)
        if cov.dim() == 3:
            none = none.unsqueeze(0)
        return none

    def length_wu(self, cur_len, alpha=0.):
        """GNMT length re-ranking score.

        See "Google's Neural Machine Translation System" :cite:`wu2016google`.
        """

        return ((5 + cur_len) / 6.0) ** alpha

    def length_average(self, cur_len, alpha=0.):
        """Returns the current sequence length."""
        return cur_len

    def length_none(self, cur_len, alpha=0.):
        """Returns unmodified scores."""
        return 1.0


class GNMTGlobalScorer(object):
    """NMT re-ranking.

    Args:
       alpha (float): Length parameter.
       beta (float):  Coverage parameter.
       length_penalty (str): Length penalty strategy.
       coverage_penalty (str): Coverage penalty strategy.

    Attributes:
        alpha (float): See above.
        beta (float): See above.
        length_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        coverage_penalty (callable): See :class:`penalties.PenaltyBuilder`.
        has_cov_pen (bool): See :class:`penalties.PenaltyBuilder`.
        has_len_pen (bool): See :class:`penalties.PenaltyBuilder`.
    """

    @classmethod
    def from_opt(cls, alpha=0.2, beta=0.2, length_penalty="wu", coverage_penalty="wu"):
        # alpha = 0.2, beta = 0.2
        return cls(alpha, beta, length_penalty, coverage_penalty)

    def __init__(self, alpha=0.2, beta=0.2, length_penalty="wu", coverage_penalty="wu"):
        # alpha = 0.2, beta = 0.2
        self._validate(alpha, beta, length_penalty, coverage_penalty)
        self.alpha = alpha
        self.beta = beta
        penalty_builder = PenaltyBuilder(coverage_penalty, length_penalty)
        self.has_cov_pen = penalty_builder.has_cov_pen
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty

        self.has_len_pen = penalty_builder.has_len_pen
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty

    @classmethod
    def _validate(cls, alpha, beta, length_penalty, coverage_penalty):
        # these warnings indicate that either the alpha/beta
        # forces a penalty to be a no-op, or a penalty is a no-op but
        # the alpha/beta would suggest otherwise.
        if length_penalty is None or length_penalty == "none":
            if alpha != 0:
                warnings.warn("Non-default `alpha` with no length penalty. "
                              "`alpha` has no effect.")
        else:
            # using some length penalty
            if length_penalty == "wu" and alpha == 0.:
                warnings.warn("Using length penalty Wu with alpha==0 "
                              "is equivalent to using length penalty none.")
        if coverage_penalty is None or coverage_penalty == "none":
            if beta != 0:
                warnings.warn("Non-default `beta` with no coverage penalty. "
                              "`beta` has no effect.")
        else:
            # using some coverage penalty
            if beta == 0.:
                warnings.warn("Non-default coverage penalty with beta==0 "
                              "is equivalent to using coverage penalty none.")


class DecodeStrategy(object):
    def __init__(self, batch_size, beam_size, pad, bos, eos, min_length, max_length, device=None):
        if device is None:
            device = torch.device('cpu')
        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos

        self.batch_size = batch_size
        self.beam_size = beam_size
        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]

        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length

        self.done = False
        self.alive_seq = torch.full([self.batch_size * self.beam_size, 1], self.bos, dtype=torch.long,
                                    device=device)
        self.is_finished = torch.zeros([self.batch_size, self.beam_size], dtype=torch.uint8, device=device)

    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def advance(self, log_probs):
        """DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """

        raise NotImplementedError()

    def update_finished(self):
        """DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """

        raise NotImplementedError()


class BeamSearch(DecodeStrategy):
    def __init__(self, batch_size, beam_size, pad, bos, eos, min_length, max_length,
                 n_best, global_scorer, device=None):
        super(BeamSearch, self).__init__(batch_size, beam_size, pad, bos, eos, min_length, max_length, device)
        if device is None:
            device = torch.device('cpu')
        # beam parameters
        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.n_best = n_best

        # result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
        # BoolTensor was introduced in pytorch 1.2
        try:
            self.top_beam_finished = self.top_beam_finished.bool()
        except AttributeError:
            pass
        self._batch_offset = torch.arange(batch_size, dtype=torch.long)

        self.select_indices = None
        self.done = False
        # "global state" of the old beam
        self._prev_penalty = None
        self._coverage = None
        self._cov_pen = self.global_scorer.has_cov_pen

        # tensors used for searching beams
        # self.best_scores = torch.full([self.batch_size], -1e10, dtype=torch.float, device=device)
        self._beam_offset = torch.arange(0, self.batch_size * self.beam_size, step=self.beam_size,
                                         dtype=torch.long, device=device)
        self.topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (self.beam_size - 1),
                                           device=device).repeat(self.batch_size)
        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty((self.batch_size, self.beam_size), dtype=torch.float, device=device)
        self.topk_ids = torch.empty((self.batch_size, self.beam_size), dtype=torch.long, device=device)
        self._batch_index = torch.empty([self.batch_size, self.beam_size], dtype=torch.long, device=device)

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size).fmod(self.beam_size)

    @property
    def batch_offset(self):
        return self._batch_offset

    def fn_map_state(self, state, dim):
        return tile(state, self.beam_size, dim=dim)

    def advance(self, log_probs):
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        _B = log_probs.shape[0] // self.beam_size

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(step + 1, alpha=self.global_scorer.alpha)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs / length_penalty
        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)
        self.topk_scores, self.topk_ids = torch.topk(curr_scores,  self.beam_size, dim=-1)

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        self.topk_log_probs = torch.mul(self.topk_scores, length_penalty)

        # Resolve beam origin and map to batch index flat representation.
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)
        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat([self.alive_seq.index_select(0, self.select_indices),
                                    self.topk_ids.view(_B * self.beam_size, 1)], -1)

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()

    def update_finished(self):
        # Penalize beams that finished.
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # on real data (newstest2017) with the pretrained transformer,
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:
                # convert back to probability by taking exp(log_softmax)
                self.hypotheses[b].append([self.topk_scores[i, j].exp().item(), predictions[i, j].tolist()])
            # End condition is the top beam finished and we can return
            # n_best hypotheses.
            finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                best_hyp = sorted(self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred) in enumerate(best_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0, non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished).view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)


if __name__ == '__main__':
    PAD_idx = 0
    SOT_idx = 1
    EOT_idx = 2
    UNK_idx = 3
    global_scorer = GNMTGlobalScorer(alpha=0., beta=0., length_penalty=None, coverage_penalty=None)
    beam_node = BeamSearch(
                            batch_size=8,
                            beam_size=2,
                            pad=PAD_idx,
                            bos=SOT_idx,
                            eos=EOT_idx,
                            min_length=1,
                            max_length=1000,
                            n_best=2,
                            global_scorer=GNMTGlobalScorer(alpha=0., beta=0., length_penalty=None, coverage_penalty=None))
