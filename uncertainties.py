"""
Author
* Lars Rumberg 2024
"""

from speechbrain.nnet.losses import ctc_loss
from speechbrain.decoders import ctc_greedy_decode
import torch
import math
import numpy as np
import kaldialign


NEG_INF = -float("inf")


def ctc_loss_python(log_probs, labels):
    """
    reimplement ctc loss (without gradients)
    :param log_probs:
    :return:
    """
    # remove batch dim
    labels = labels[0]
    log_probs = log_probs[0]
    T = log_probs.shape[0]

    # extend sequence by blanks inbetween each token and at start and end
    l_dash = torch.zeros(len(labels) * 2 + 1)
    l_dash[1:-1:2] = labels
    l_dash = l_dash.int()

    # initialize forward variable, first dim is time steps, second dim is sequence (extended by blanks)
    alpha = torch.ones(T, len(l_dash))
    alpha = alpha * NEG_INF  # alpha.numpy().transpose()
    alpha[0, 0] = log_probs[0, 0]
    alpha[0, 1] = log_probs[0, l_dash[1]]

    for t in range(1, T):
        for s in range(0, len(l_dash)):
            if (s + 1) < len(l_dash) - 2 * (T - (t + 1)) - 1:
                continue
            y_t_ld_s = log_probs[t, l_dash[s]]
            if s == 0:
                alpha_tmp = alpha[t - 1, s]
            elif s == 1 or l_dash[s] == 0 or l_dash[s - 2] == l_dash[s]:
                alpha_tmp = alpha[t - 1, s - 1 : s + 1].logsumexp(dim=-1)  # add alpha[t-1,s-1] and alpha[t-1,s]
            else:
                alpha_tmp = alpha[t - 1, s - 2 : s + 1].logsumexp(dim=-1)  # add alpha[t-1,s-2] to alpha[t-1,s]
            alpha[t, s] = alpha_tmp + y_t_ld_s

    forward_loss = alpha[-1, -2:].logsumexp(dim=-1)  # should be identical/close to loss

    # initialize backward variables:
    beta = torch.ones(T, len(l_dash))
    beta = beta * NEG_INF
    beta[-1, -1] = log_probs[-1, 0]
    beta[-1, -2] = log_probs[-1, l_dash[-2]]

    for t in range(T - 2, -1, -1):
        for s in range(len(l_dash) - 1, -1, -1):
            if (s + 1) > 2 * (t + 1):
                continue
            y_t_ld_s = log_probs[t, l_dash[s]]
            if s == len(l_dash) - 1:
                beta_tmp = beta[t + 1, s]
            elif s == len(l_dash) - 2 or l_dash[s] == 0 or l_dash[s + 2] == l_dash[s]:
                beta_tmp = beta[t + 1, s : s + 2].logsumexp(dim=-1)  # add alpha[t+1,s] and alpha[t+1,s+1]
            else:
                beta_tmp = beta[t + 1, s : s + 3].logsumexp(dim=-1)  # add alpha[t+1,s] to alpha[t+1,s+2]
            beta[t, s] = beta_tmp + y_t_ld_s

    backward_loss = beta[0, 0:2].logsumexp(dim=-1)

    # forward and backward loss should be the same
    assert (backward_loss - forward_loss).abs() < 1e-4

    return forward_loss, l_dash, alpha, beta


def logsubstractexp(tensor, other):
    a = torch.max(tensor, other)
    a = a + ((tensor - a).exp() - (other - a).exp()).log()
    a[torch.isnan(a)] = -torch.inf
    return a


def adjust_score_by_skipping_s(log_probs, l_dash, alpha, beta, s_start, s_end):
    """
    Computes the log probability of the sequence until before s_start and from after s_end, independent of what happens
    inbetween. Subtract from the full sequence log-prob (negative ctc-loss), to get the conditional log-prob of the
    tokens s_start:s_end given the rest.

    :param log_probs: model output
    :param l_dash:  token sequence extended by blank at start, end and between each token
    :param alpha:   ctc forward variable
    :param beta:    ctc backward variable
    :param s_start: indices of first token in l_dash of sub-sequence to estimate uncertainty of
    :param s_end:   indices of last token in l_dash of sub-sequence to estimate uncertainty of. if same as s_start,
                    compute uncertainty just for that token
    :return:
    """
    T = alpha.shape[0]
    S = len(l_dash)

    assert s_end >= s_start

    # alpha[:, previous] and beta[:, next] have to be recomputed they must not include paths from same s from previous t
    if s_start != 1:
        previous = s_start - 1 if l_dash[s_start] == l_dash[s_start - 2] else s_start - 2

        _alpha = torch.zeros(T)
        _alpha[0] = log_probs[0, l_dash[previous]] if previous == 1 else NEG_INF
        _alpha[1:] = logsubstractexp(alpha[1:, previous], alpha[:-1, previous] + log_probs[1:, l_dash[previous]])
        _alpha[torch.isnan(_alpha)] = NEG_INF
        # nan happens when probability is negative, because alpha was NEG_INF because we can't finish from this position

    else:
        _alpha = torch.ones(T) * NEG_INF
        _alpha[0] = 0

    if s_end != S - 2:
        _next = s_end + 1 if l_dash[s_end] == l_dash[s_end + 2] else s_end + 2

        _beta = torch.zeros(T)
        _beta[-1] = log_probs[-1, l_dash[_next]] if _next == S - 2 else NEG_INF
        _beta[:-1] = logsubstractexp(beta[:-1, _next], beta[1:, _next] + log_probs[:-1, l_dash[_next]])
        _beta[torch.isnan(_beta)] = NEG_INF
        # nan happens when probability is negative, because beta was NEG_INF because we can't finish from this position

    else:
        _beta = torch.ones(T) * NEG_INF
        _beta[-1] = 0

    score_going_forward_from_s_at_t = []
    for t in range(0, T - 1):
        _forward_until = _alpha[0 : t + 1].logsumexp(dim=-1)
        _backward_from_there = _beta[t + 1]
        score_going_forward_from_s_at_t.append(_forward_until + _backward_from_there)

    adjusted_score = torch.tensor([score_going_forward_from_s_at_t]).logsumexp(dim=-1)

    if s_end == S - 2:
        assert (_alpha.logsumexp(dim=-1) - adjusted_score).abs() < 1e-5
    if s_start == 1:
        assert (_beta.logsumexp(dim=-1) - adjusted_score).abs() < 1e-5

    return adjusted_score


def uncertainty_forward_backward(log_probs, word_boundary=None):
    """
    computes the forward score once for each token in the decoded sequence and let the paths trough the forward score
    matrix skip the row for the token for which the uncertainty is computed. This forward score should then be the
    probability of the rest of the sequence with anything (deletion, insertion, substitution) happening at the point of
    the token. Dividing the loss of the full sequence by that forward score results in the probability of that token
    given the rest of the sequence. Uncertainty is 1 minus that value.

    :param log_probs:
    :param word_boundary
    :param mode:
    :return:
    """

    decoded = torch.tensor(ctc_greedy_decode(log_probs, torch.tensor([1.0]), blank_id=0))
    # remove batch dim
    decoded = decoded[0]
    log_probs = log_probs[0]

    # extend sequence by blanks inbetween each token and at start and end
    l_dash = torch.zeros(len(decoded) * 2 + 1)
    l_dash[1:-1:2] = decoded
    l_dash = l_dash.int()

    # compute alpha and beta
    forward_score, l_dash, alpha, beta = ctc_loss_python(log_probs.unsqueeze(0), decoded.unsqueeze(0))

    # compute uncertainty on token level
    f_b_until_tokens = []
    for unc_s in range(1, len(l_dash), 2):
        f_b_until_tokens.append(adjust_score_by_skipping_s(log_probs, l_dash, alpha, beta, unc_s, unc_s))

    # compute uncertainty on word level
    f_b_until_tokens = torch.tensor(f_b_until_tokens)
    if word_boundary is not None:
        f_b_until_words = []
        word_boundaries = (l_dash == word_boundary).nonzero().squeeze(-1)
        if len(word_boundaries) == 0:
            # just one word
            f_b_until_words.append(adjust_score_by_skipping_s(log_probs, l_dash, alpha, beta, 1, len(l_dash) - 2))
        else:
            if word_boundaries[0] != 1:
                f_b_until_words.append(
                    adjust_score_by_skipping_s(log_probs, l_dash, alpha, beta, 1, word_boundaries[0])
                )
            for word_boundary_i in range(len(word_boundaries) - 1):
                f_b_until_words.append(
                    adjust_score_by_skipping_s(
                        log_probs,
                        l_dash,
                        alpha,
                        beta,
                        word_boundaries[word_boundary_i],
                        word_boundaries[word_boundary_i + 1],
                    )
                )
            if word_boundaries[-1] != len(l_dash) - 2:
                f_b_until_words.append(
                    adjust_score_by_skipping_s(log_probs, l_dash, alpha, beta, word_boundaries[-1], len(l_dash) - 2)
                )
        f_b_until_words = torch.tensor(f_b_until_words)

    loss = ctc_loss(
        log_probs.unsqueeze(0),
        decoded.unsqueeze(0),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        blank_index=0,
        reduction="batchmean",
    )

    forward_uncertainty = 1 - (-loss - f_b_until_tokens).exp()
    assert (-loss - forward_score).abs() < 1e-4
    if word_boundary is not None:
        forward_uncertainty_words = 1 - (-loss - f_b_until_words).exp()

    return_dict = {f"forward_uncertainty": [_.item() for _ in forward_uncertainty]}
    if word_boundary is not None:
        return_dict[f"word_forward_uncertainty"] = [_.item() for _ in forward_uncertainty_words]
    return return_dict


def vyas_uncertainty(ensemble, reference, level="token"):
    """
    compute uncertainty by comparing the decoded predictions of an ensemble
    see Vyas et al., “Analyzing Uncertainties in Speech Recognition Using Dropout”, ICASSP 2019
    """

    if level == "word":
        ref = reference[1:] if reference[0] == "" else reference
        ref = ref[:-1] if ref[-1] == "" else ref
        ref = ["_" if _ref == "" else _ref for _ref in ref]
        ref = "".join(ref).split("_")
    else:
        ref = reference

    disagreements = np.zeros(len(ref))
    for ensemble_sample in ensemble:

        if level == "word":
            hyp = ensemble_sample[1:] if ensemble_sample[0] == "" else ensemble_sample
            hyp = hyp[:-1] if hyp[-1] == "" else hyp
            hyp = ["_" if _hyp == "" else _hyp for _hyp in hyp]
            hyp = "".join(hyp).split("_")
        else:
            hyp = ensemble_sample

        alignment = kaldialign.align(ref, hyp, "*")
        alignment_ref = np.array([token[0] for token in alignment])
        alignment_hyp = np.array([token[1] for token in alignment])
        aligned_error = alignment_ref != alignment_hyp
        # add insertions to both neighbors
        insertions = np.array(alignment_ref) == "*"
        local_error_aligned_to_ref = (
            aligned_error + np.convolve(insertions, np.array([1] * min(len(aligned_error), 3)), "same")
        ) >= 1
        local_error_aligned_to_ref = local_error_aligned_to_ref[np.logical_not(insertions)]
        disagreements += local_error_aligned_to_ref
    uncertainty = disagreements / len(ensemble)

    return uncertainty.tolist()


def aggregated_frame_uncertainty(
    nll_out, blank, method=["sum_of_tokens_that_change"], aggregate=["max"], exclude_blanks=False, debug=False
):
    _debug = []
    left_neighbor = None
    token = nll_out[0].argmax()
    decoding = {_method: [[token.item(), []]] for _method in method}
    v = nll_out.shape[1]
    for frame in range(nll_out.shape[0]):
        right_neighbor = nll_out[frame + 1].argmax() if (frame + 1) < nll_out.shape[0] else None
        for _method in method:
            if _method == "p_max":
                uncertainty = 1 - nll_out[frame].max().exp()
            elif _method == "p_max_normalized":  # normalized maximum probability conﬁdence from Laptev2023
                uncertainty = 1 - ((nll_out[frame].max().exp() * v - 1) / (v - 1))
            elif "tsallis_exp" in _method:  # "tsallis_exp_03"
                # see
                # Laptev and Ginsburg,
                # “Fast Entropy-Based Methods of Word-Level Confidence Estimation for End-to-End Automatic Speech Recognition,”
                # in 2022 IEEE Spoken Language Technology Workshop (SLT), 2023
                alpha = int(_method.split("_")[2]) / 10
                neg_entropy_alpha = (nll_out[frame] * alpha).exp().sum()
                exp_neg_max_ent = math.exp((1 - math.pow(v, 1 - alpha)) / (1 - alpha))
                uncertainty = 1 - (((1 - neg_entropy_alpha) / (1 - alpha)).exp() - exp_neg_max_ent) / (
                    1 - exp_neg_max_ent
                )

            elif _method == "sum_of_tokens_that_change":
                # see
                # Rumberg et al.,
                # “Uncertainty Estimation for Connectionist Temporal Classification Based Automatic Speech Recognition,”
                # in Proceedings INTERSPEECH 2023
                tokens_that_change_decoding = (
                    [i for i in range(nll_out.shape[1]) if i != token]
                    # if the token is not the blank and not in the neighbors or the token and both neighbors are the same,
                    # all tokens except itself changes the decoding.
                    if token not in (right_neighbor, left_neighbor, blank)
                    or (token == right_neighbor and token == left_neighbor)
                    else [i for i in range(nll_out.shape[1]) if i not in (token, right_neighbor, left_neighbor, blank)]
                    # if the token is in exactly one neighbor, all tokens except the token, the neighbors and the blank change
                    # the decoding
                )
                uncertainty = nll_out[frame][tokens_that_change_decoding].exp().sum()

            if token != decoding[_method][-1][0]:
                decoding[_method].append([token.item(), []])
            decoding[_method][-1][1].append(uncertainty.item())

        left_neighbor = token
        token = right_neighbor

        # store frames where any uncertainty method is above a threshold
        if debug and any([decoding[_method][-1][1][-1] > 0.05 for _method in method]):
            _debug.append({_method: decoding[_method][-1][1][-1] for _method in method})
            _debug[-1]["nll"] = nll_out[frame].tolist()

    return_dict = dict()
    for _method in method:
        _exclude_blanks = True if "exclude_blanks" in _method else exclude_blanks
        for _aggregate in aggregate:
            aggregated = []
            tokens_with_blanks = []
            for i in range(len(decoding[_method])):
                tokens_with_blanks.append(decoding[_method][i][0])
                if _aggregate == "mean":
                    aggregated.append(torch.tensor(decoding[_method][i][1]).mean().item())
                elif _aggregate == "max":
                    aggregated.append(torch.tensor(decoding[_method][i][1]).max().item())
                elif _aggregate == "min":
                    aggregated.append(torch.tensor(decoding[_method][i][1]).min().item())
                elif _aggregate == "prod":  # product of confidences not uncertainties
                    aggregated.append(1 - (1 - torch.tensor(decoding[_method][i][1])).prod().item())

            # assign uncertainty of blank decoding to neighbouring tokens:
            uncertainty_with_blanks = np.array(aggregated)
            tokens_with_blanks = np.array(tokens_with_blanks)
            blanks = tokens_with_blanks == 0
            if not _exclude_blanks:
                # this can lead to uncertainties > 1:
                uncertainty_with_blanks = uncertainty_with_blanks + np.convolve(
                    uncertainty_with_blanks * blanks, np.array([1] * min(len(uncertainty_with_blanks), 3)), "same"
                )
            uncertainty = uncertainty_with_blanks[np.logical_not(blanks)]

            __method = f"{_method}_exclude_blanks" if exclude_blanks else _method
            return_dict[f"aggregated_frame_uncertainty_{__method}_{_aggregate}"] = uncertainty.tolist()

    if debug:
        return_dict["debug"] = _debug

    return return_dict


def merge_p_out_uncertainties_from_ensemble(reference, ensemble):
    ref = reference["tokens"]
    uncertainties = dict()
    for aggregate in ("_mean", "_max", "_min", "_prod", ""):
        for _method in [
            "sum_of_tokens_that_change",
            "p_max",
            "p_max_normalized",
            "alternative_losses",
            "forward_uncertainty",
            "word_forward_uncertainty",
            "tsallis_exp_03",
            "tsallis_exp_05",
            "tsallis_exp_08",
            "tsallis_exp_03_exclude_blanks",
            "tsallis_exp_05_exclude_blanks",
            "tsallis_exp_08_exclude_blanks",
        ]:
            method = f"{_method}{aggregate}"
            if method not in ensemble[0].keys():
                method = f"aggregated_frame_uncertainty_{method}"
                if method not in ensemble[0].keys():
                    continue

            ref = reference["tokens"]

            if "word" in method:
                ref = ref[1:] if ref[0] == "" else ref
                ref = ref[:-1] if ref[-1] == "" else ref
                ref = ["_" if _ref == "" else _ref for _ref in ref]
                ref = "".join(ref).split("_")

            uncertainties[method] = np.empty((len(ensemble), len(ref)))

            for i, ensemble_sample in enumerate(ensemble):
                hyp = ensemble_sample["tokens"]

                if "word" in method:
                    hyp = hyp[1:] if hyp[0] == "" else hyp
                    hyp = hyp[:-1] if hyp[-1] == "" else hyp
                    hyp = ["_" if _hyp == "" else _hyp for _hyp in hyp]
                    hyp = "".join(hyp).split("_")

                alignment = kaldialign.align(ref, hyp, "*")

                alignment_ref = np.array([token[0] for token in alignment])
                alignment_hyp = np.array([token[1] for token in alignment])
                insertions = alignment_ref == "*"
                deletions = alignment_hyp == "*"

                uncertainty = np.zeros_like(deletions, dtype=float)
                uncertainty[np.logical_not(deletions)] = np.array(ensemble_sample[method])
                # move uncertainty of insertions to neighbors
                if len(uncertainty) > 0:
                    uncertainty = uncertainty + np.convolve(
                        uncertainty * insertions, np.array([1] * min(len(uncertainty), 3)), "same"
                    )
                # deletions are set as nan to not be included in the mean, since this uncertainty is most likely
                # already in the uncertainty of the neighboring tokens
                uncertainty[deletions] = np.nan
                uncertainties[method][i] = uncertainty[np.logical_not(insertions)]
            uncertainties[method] = np.nansum(uncertainties[method], axis=0) / (
                1 - np.isnan(uncertainties[method])
            ).sum(axis=0)
    return {key: value.tolist() for key, value in uncertainties.items()}
