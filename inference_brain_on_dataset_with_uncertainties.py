"""
Author
* Lars Rumberg 2024
"""

import torch
import math
import kaldialign
import numpy as np
from speechbrain.decoders import ctc_greedy_decode
from speechbrain.core import Stage
from tqdm import tqdm

from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.dataio.encoder import CTCTextEncoder

from uncertainties import (
    uncertainty_forward_backward,
    vyas_uncertainty,
    aggregated_frame_uncertainty,
    merge_p_out_uncertainties_from_ensemble,
)


def inference_brain_on_dataset_with_uncertainty(
    asr_brain, dataset, ensemble_size, tokenizer, hparams, subset, min_key="WER"
):
    """
    uses the asr brain to do inference on the test split of the given dataset.
    Computes uncertainties on token and word level using all different explored methods.
    For all methods using any kind of aggregation (frame to token or/and token to word)
    all different (mean, max, min, product) aggregation methods are computed.
    If ensemble size is greater than 0, inference is done multiple times with activated dropout,
    all methods are than averaged both by averaging first the output probabilities than computing uncertainty (dropout_mean),
    and by computing uncertainties for each ensemble member and than averaging these (dropout_merged).

    With the combination of all methods, aggregations and especially with ensemble>0 this might be quite slow.
    Set subset<1 to only use a subset of the test set. For MCV for example the test set is so big,
    that using subset=0.1 leads to close to identical results.
    """

    def decode_label_encoder(encoded_labels):
        if isinstance(tokenizer, SentencePiece):
            return [tokenizer.sp.decode(_x) for _x in encoded_labels]
        elif isinstance(tokenizer, CTCTextEncoder):
            return tokenizer.decode_ndim(encoded_labels)

    test_set = asr_brain.make_dataloader(dataset, Stage.TEST, **hparams["test_dataloader_options"])
    asr_brain.on_evaluate_start(max_key=None, min_key=min_key)  # loads checkpoint
    asr_brain.on_stage_start(Stage.TEST, epoch=None)
    asr_brain.modules.eval()

    if isinstance(tokenizer, SentencePiece):
        word_delimiter = [tokenizer.sp.id_to_piece(_id) for _id in range(tokenizer.sp.piece_size())].index("▁")
    elif isinstance(tokenizer, CTCTextEncoder):
        word_delimiter = None

    decodings = dict()
    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(test_set, dynamic_ncols=True)):
            if batch_i > len(test_set) * subset:
                break
            if hasattr(batch, "dataset_name") and batch.dataset_name[0] == "talc":
                batch.tokens = [
                    torch.LongTensor([tokenizer.sp.encode_as_ids(batch.orthographic_clean[0].upper())]),
                    torch.Tensor([1]),
                ]
                batch._PaddedBatch__keys.append("tokens")
            batch_token_key = "tokens" if hasattr(batch, "tokens") else "phn_encoded"
            sample_id = batch["id"][0]
            outs = asr_brain.compute_forward(batch, stage=Stage.TEST)
            loss = asr_brain.compute_objectives(outs, batch, stage=Stage.TEST)
            nll_out_no_dropout, _ = outs
            nll_out_no_dropout = nll_out_no_dropout[0].detach().cpu()

            decodings[sample_id] = dict(split="test", ctc_loss=loss.detach().cpu().item())
            decodings[sample_id]["no_dropout_tokens"] = decode_label_encoder(
                ctc_greedy_decode(nll_out_no_dropout.unsqueeze(0), torch.Tensor([1]), blank_id=0)[0]
            )

            nll_out_ensemble_mean = None
            if ensemble_size > 0:
                nll_out_ensemble = []
                with asr_brain.dropout_active():
                    for i in range(ensemble_size):
                        nll_out, _ = asr_brain.compute_forward(batch, stage=Stage.TEST)
                        nll_out_ensemble.append(nll_out[0].detach().cpu())

                nll_out_ensemble_mean = torch.stack(nll_out_ensemble).logsumexp(0) - math.log(50)
                ensemble_decodings = [
                    decode_label_encoder(ctc_greedy_decode(nll_out.unsqueeze(0), torch.Tensor([1]), blank_id=0)[0])
                    for nll_out in nll_out_ensemble
                ]

                decodings[sample_id]["dropout_mean_tokens"] = decode_label_encoder(
                    ctc_greedy_decode(nll_out_ensemble_mean.unsqueeze(0), torch.Tensor([1]), blank_id=0)[0]
                )
                decodings[sample_id]["vyas_uncertainty"] = vyas_uncertainty(
                    ensemble_decodings, decodings[sample_id]["no_dropout_tokens"], level="token"
                )
                if word_delimiter is not None:
                    decodings[sample_id]["word_vyas_uncertainty"] = vyas_uncertainty(
                        ensemble_decodings, decodings[sample_id]["no_dropout_tokens"], level="word"
                    )

            decodings[sample_id].update(uncertainty_forward_backward(nll_out_no_dropout.unsqueeze(0), word_delimiter))

            for key, nll_out in zip(["no_dropout", "dropout_mean"], [nll_out_no_dropout, nll_out_ensemble_mean]):

                if ensemble_size == 0 and key == "dropout_mean":
                    continue

                decodings[sample_id].update(
                    {
                        f"{key}_{key2}": value
                        for key2, value in aggregated_frame_uncertainty(
                            nll_out,
                            method=[
                                "p_max",
                                "p_max_normalized",
                                "sum_of_tokens_that_change",
                                "tsallis_exp_03_exclude_blanks",
                                "tsallis_exp_03",
                                "tsallis_exp_08_exclude_blanks",
                                "tsallis_exp_08",
                            ],
                            aggregate=["mean", "max", "min", "prod"],
                            blank=0,
                            debug=True,
                        ).items()
                    }
                )

            if ensemble_size > 0:
                ensemble_decodings = []
                for nll_out in nll_out_ensemble + [nll_out_no_dropout]:
                    ensemble_sample = dict()
                    ensemble_sample.update(
                        aggregated_frame_uncertainty(
                            nll_out,
                            method=[
                                "p_max",
                                "p_max_normalized",
                                "sum_of_tokens_that_change",
                                "tsallis_exp_03_exclude_blanks",
                                "tsallis_exp_03",
                                "tsallis_exp_08_exclude_blanks",
                                "tsallis_exp_08",
                            ],
                            aggregate=["mean", "max", "min", "prod"],
                            blank=0,
                        )
                    )

                    ensemble_sample.update(uncertainty_forward_backward(nll_out.unsqueeze(0), word_delimiter))
                    ensemble_sample["tokens"] = decode_label_encoder(
                        ctc_greedy_decode(nll_out.unsqueeze(0), torch.Tensor([1]), blank_id=0)[0]
                    )
                    ensemble_decodings.append(ensemble_sample)
                key = "dropout_merged"

                decodings[sample_id].update(
                    {
                        f"{key}_{key2}": value
                        for key2, value in merge_p_out_uncertainties_from_ensemble(
                            dict(tokens=decodings[sample_id]["no_dropout_tokens"]), ensemble_decodings
                        ).items()
                    }
                )
                if word_delimiter is not None:
                    decodings[sample_id]["word_dropout_merged_forward_uncertainty"] = decodings[sample_id][
                        "dropout_merged_word_forward_uncertainty"
                    ]
                    del decodings[sample_id]["dropout_merged_word_forward_uncertainty"]

            for key in ["no_dropout", "dropout_mean"]:
                if ensemble_size == 0 and key == "dropout_mean":
                    continue
                alignment = kaldialign.align(
                    decode_label_encoder(batch[batch_token_key][0].tolist()[0]),
                    decodings[sample_id][f"{key}_tokens"],
                    "*",
                )
                alignment_ref = np.array([token[0] for token in alignment])
                alignment_hyp = np.array([token[1] for token in alignment])
                aligned_error = alignment_ref != alignment_hyp
                # if there is a deletion in the decoding, and we say both neighboring phones are incorrect
                deletions = np.array(alignment_hyp) == "*"
                local_error = (
                    aligned_error + np.convolve(deletions, np.array([1] * min(len(aligned_error), 3)), "same")
                ) >= 1
                local_error = local_error[np.logical_not(deletions)]
                decodings[sample_id][f"local_error_aligned_to_decoding_{key}"] = local_error.tolist()

            # aggregate uncertainties to word levels
            if word_delimiter is not None:
                ref = decode_label_encoder(batch[batch_token_key][0].tolist()[0])
                ref = ref[1:] if ref[0] == "" else ref
                ref = ref[:-1] if ref[-1] == "" else ref
                ref = ["_" if _ref == "" else _ref for _ref in ref]
                ref = "".join(ref).split("_")
                for key in ["no_dropout", "dropout_mean", "dropout_merged"]:
                    if ensemble_size == 0 and key == "dropout_mean":
                        continue
                    token_key = "no_dropout" if key == "dropout_merged" else key
                    hyp = decodings[sample_id][f"{token_key}_tokens"]
                    hyp = hyp[1:] if hyp[0] == "" else hyp
                    hyp = hyp[:-1] if hyp[-1] == "" else hyp
                    hyp = ["_" if _hyp == "" else _hyp for _hyp in hyp]
                    hyp = "".join(hyp).split("_")
                    alignment = kaldialign.align(ref, hyp, "*")
                    alignment_ref = np.array([word[0] for word in alignment])
                    alignment_hyp = np.array([word[1] for word in alignment])
                    aligned_error = alignment_ref != alignment_hyp
                    # if there is a deletion in the decoding, and we say both neighboring words are incorrect
                    deletions = np.array(alignment_hyp) == "*"
                    local_error = (
                        aligned_error + np.convolve(deletions, np.array([1] * min(len(aligned_error), 3)), "same")
                    ) >= 1
                    local_error = local_error[np.logical_not(deletions)]
                    decodings[sample_id][f"word_local_error_aligned_to_decoding_{token_key}"] = local_error.tolist()
                    token_uncertainties = [_k for _k in decodings[sample_id].keys() if f"{key}_aggregated" in _k]
                    for uncertainty_type in token_uncertainties:
                        uncertainty_words = []
                        tokens = decodings[sample_id][f"{token_key}_tokens"]
                        word_boundaries = np.where(np.array(tokens) == "")[0]
                        if len(word_boundaries) == 0:
                            uncertainty_tokens = torch.tensor(decodings[sample_id][uncertainty_type])
                            uncertainty_words.append(1 - (1 - uncertainty_tokens).prod().item())
                        else:
                            if word_boundaries[0] != 0:
                                uncertainty_tokens = torch.tensor(
                                    decodings[sample_id][uncertainty_type][0 : word_boundaries[0]]
                                )
                                # similar, but slightly worse results when taking min confidence/max uncertainty instead of prod
                                uncertainty_words.append(1 - (1 - uncertainty_tokens).prod().item())
                            for word_boundary_i in range(len(word_boundaries) - 1):
                                uncertainty_tokens = torch.tensor(
                                    decodings[sample_id][uncertainty_type][
                                        word_boundaries[word_boundary_i] : word_boundaries[word_boundary_i + 1]
                                    ]
                                )
                                # similar, but slightly worse results when taking min confidence/max uncertainty instead of prod
                                uncertainty_words.append(1 - (1 - uncertainty_tokens).prod().item())
                            if word_boundaries[-1] != len(tokens) - 1:
                                uncertainty_tokens = torch.tensor(
                                    decodings[sample_id][uncertainty_type][word_boundaries[-1] :]
                                )
                                # similar, but slightly worse results when taking min confidence/max uncertainty instead of prod
                                uncertainty_words.append(1 - (1 - uncertainty_tokens).prod().item())
                        decodings[sample_id][f"word_{uncertainty_type}"] = uncertainty_words
                        assert len(uncertainty_words) == len(
                            decodings[sample_id][f"word_local_error_aligned_to_decoding_{token_key}"]
                        )
    print(f"WER: {asr_brain.wer_metric.summarize()}")
    print(f"CER: {asr_brain.cer_metric.summarize()}")
    return decodings
