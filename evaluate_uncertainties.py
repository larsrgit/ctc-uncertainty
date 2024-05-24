"""
Author
* Lars Rumberg 2024
"""

# %%
import re
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from shapely.geometry import Polygon

from nemo_confidence_metrics import (
    auc_nt,
    auc_pr,
    auc_roc,
    auc_yc,
    ece,
    nce,
)
import matplotlib

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join(
            [
                r"\renewcommand{\cite}[1]{[#1]}",
                # r"\newcommand{\uymax}{U^t_{y_\mathrm{max}}}",
                # r"\newcommand{\uchange}{U_{\mathrm{change}}^t}",
                # r"\newcommand{\ufb}{U_{\mathrm{FB}}}",
                # r"\newcommand{\umcd}{U_{\mathrm{MCD}}}",
                # r"\newcommand{\utsallis}{U^t_{ts}}",
            ]
        ),
    }
)
# %%


# %%
def fix_citations_in_pgf_files(filenames):
    for _filename in filenames:
        with open(_filename, "r") as f:
            pgf_content = f.read()
        pgf_content = re.sub(r"\\_", "_", pgf_content)
        with open(_filename, "w") as f:
            f.write(pgf_content)


# %%
def evaluate_uncertainties(decodings, trial_id, ensemble_size, dataset):
    # %%
    figsize_factor = 1.3

    methods = [
        [
            "no_dropout_aggregated_frame_uncertainty_p_max",
            "no_dropout",
            r"$U^t_{y_\mathrm{max}} = 1 - \mathrm{max}_k(y^t_k)$",
            "-",
            "#800000",
        ],
        # ["no_dropout_aggregated_frame_uncertainty_tsallis_exp_03", "no_dropout", "tsallis 03 with blanks \cite{laptev_fast_2023}", "-", "#469990"],
        [
            "no_dropout_aggregated_frame_uncertainty_tsallis_exp_03_exclude_blanks",
            "no_dropout",
            r"Tsallis entropy $U^t_{ts}$, $\alpha=0.3$ \cite{laptev_fast_2023}",
            "-",
            "#42d4f4",
        ],
        [
            "no_dropout_aggregated_frame_uncertainty_tsallis_exp_05_exclude_blanks",
            "no_dropout",
            r"Tsallis entropy $U^t_{ts}$, $\alpha=0.5$ \cite{laptev_fast_2023}",
            "-",
            "#42d4f4",
        ],
        [
            "no_dropout_aggregated_frame_uncertainty_tsallis_exp_08_exclude_blanks",
            "no_dropout",
            r"Tsallis entropy $U^t_{ts}$, $\alpha=0.8$ \cite{laptev_fast_2023}",
            "-",
            "#42d4f4",
        ],
        [
            "no_dropout_aggregated_frame_uncertainty_sum_of_tokens_that_change",
            "no_dropout",
            r"$U_{\mathrm{change}}^t$ \cite{rumberg_uncertainty_2023}",
            "-",
            "#4363d8",
        ],
        [
            "dropout_merged_aggregated_frame_uncertainty_p_max",
            "no_dropout",
            r"MCD average of $U^t_{y_\mathrm{max}} = 1 - \mathrm{max}_k(y^t_k)$",
            "--",
            "#800000",
        ],
        # ["dropout_merged_aggregated_frame_uncertainty_tsallis_exp_03", "no_dropout", "MCD average of tsallis 03 with blanks \cite{laptev_fast_2023}", "--", "#469990"],
        [
            "dropout_merged_aggregated_frame_uncertainty_tsallis_exp_03_exclude_blanks",
            "no_dropout",
            r"MCD average of Tsallis entropy $U^t_{ts}$, $\alpha=0.3$ \cite{laptev_fast_2023}",
            "--",
            "#42d4f4",
        ],
        [
            "dropout_merged_aggregated_frame_uncertainty_tsallis_exp_05_exclude_blanks",
            "no_dropout",
            r"MCD average of Tsallis entropy $U^t_{ts}$, $\alpha=0.5$ \cite{laptev_fast_2023}",
            "--",
            "#42d4f4",
        ],
        [
            "dropout_merged_aggregated_frame_uncertainty_tsallis_exp_08_exclude_blanks",
            "no_dropout",
            r"MCD average of Tsallis entropy $U^t_{ts}$, $\alpha=0.8$ \cite{laptev_fast_2023}",
            "--",
            "#42d4f4",
        ],
        [
            "dropout_merged_aggregated_frame_uncertainty_sum_of_tokens_that_change",
            "no_dropout",
            r"MCD average of $U_{\mathrm{change}}^t$ \cite{rumberg_uncertainty_2023}",
            "--",
            "#4363d8",
        ],
        [
            "vyas_uncertainty",
            "no_dropout",
            r"MCD Disagreements $U_{\mathrm{MCD}}$ \cite{vyas_analyzing_2019}",
            "-",
            "#f58231",
        ],
        ["forward_uncertainty", "no_dropout", r"Forward-Backward $U_{\mathrm{FB}}$", "-", "#f032e6"],
        [
            "dropout_merged_forward_uncertainty",
            "no_dropout",
            r"MCD average of Forward-Backward $U_{\mathrm{FB}}$",
            "--",
            "#f032e6",
        ],
    ]
    pgf_files = []
    for split in ["test"]:  # "val",
        for level in ["", "word_"]:
            if level == "word_" and dataset == "timit":
                continue
            metrics_dict = dict()
            values_dict = dict()
            for _uncertainty_type, decoding_type, legend_label, line_style, color in methods:
                for aggregate in ("_min", "_max", "_mean", "_prod", ""):
                    uncertainty_type = f"{level}{_uncertainty_type}{aggregate}"

                    if uncertainty_type not in list(decodings.values())[0].keys():
                        continue

                    if uncertainty_type not in list(decodings.values())[-1].keys():
                        continue

                    # decoded_tokens = np.concatenate(
                    #     [
                    #         np.array(decodings[key][f"{decoding_type}_tokens"])
                    #         for key in decodings.keys()
                    #         if decodings[key]["split"] == split
                    #     ]
                    # )

                    uncertainty = np.concatenate(
                        [
                            np.array(decodings[key][uncertainty_type])
                            for key in decodings.keys()
                            if decodings[key]["split"] == split
                        ]
                    )
                    local_error = np.concatenate(
                        [
                            np.array(decodings[key][f"{level}local_error_aligned_to_decoding_{decoding_type}"])
                            for key in decodings.keys()
                            if decodings[key]["split"] == split
                        ]
                    )

                    # rejection curves. tokens are sorted by uncertainty and rejected to an oracle
                    uncertainty_argsort = np.argsort(uncertainty)[::-1]
                    orig_len = uncertainty.size
                    wer = []
                    percent_rejected = []
                    current_error = local_error.sum()
                    remaining = local_error.size
                    local_error_sum = local_error.sum()
                    for _id in uncertainty_argsort:
                        wer.append(current_error / local_error_sum)
                        percent_rejected.append(1 - (remaining / orig_len))
                        remaining -= 1
                        if local_error[_id]:
                            current_error -= 1

                    polygon_ar_uns = [[p_r, _wer] for p_r, _wer in zip(percent_rejected, wer)] + [[1, 0], [0, 1]]
                    polygon_ar_orc = [[0, 1], [local_error.sum() / orig_len, 0], [1, 0], [0, 1]]

                    PRR = Polygon(polygon_ar_uns).area / Polygon(polygon_ar_orc).area

                    dict_label = (
                        legend_label.replace("MCD average of ", ""),
                        "X" if any(_ in _uncertainty_type for _ in ["dropout_merged", "vyas"]) else "",
                        aggregate.replace("_", ""),
                    )

                    metrics_dict[dict_label] = dict(PRR=PRR)

                    # metrics from laptev2023 et al. using nemo code
                    y_true = np.logical_not(local_error)  # nemo code needs true predictions instead of errors
                    y_score = 1 - np.minimum(
                        np.maximum(uncertainty, 0), 1
                    )  # nemo code analyses confidence instead of uncertainty, min/max to catch values above 1 or below zero. Can happen due to rounding errors and when adding uncertainties from neighboring blanks to tokens
                    # following code from https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/utils/asr_confidence_benchmarking_utils.py
                    # output scheme: yc.mean(), yc.max(), yc.std() or yc.mean(), yc.max(), yc.std(), (thresholds, yc)
                    result_yc = auc_yc(y_true, y_score, return_std_maximum=True, return_curve=True)

                    # output scheme: ece or ece, (thresholds, ece_curve)
                    results_ece = ece(y_true, y_score, return_curve=True)

                    values_dict[dict_label] = dict(
                        local_error=local_error,
                        y_true=y_true,
                        y_score=y_score,
                        percent_rejected=percent_rejected,
                        wer=wer,
                        line_style=line_style,
                        color=color,
                        legend_label=legend_label + (f" ({aggregate.replace('_', '')})" if aggregate != "" else ""),
                    )
                    metrics_dict[dict_label].update(
                        {
                            "$AUC_{ROC}$": auc_roc(local_error, 1 - y_score),
                            "$AUC_{PR}$": auc_pr(y_true, y_score),
                            "$AUC_{NT}$": auc_nt(y_true, y_score),
                            "$NCE$": nce(y_true, y_score),
                            "$ECE$": results_ece if isinstance(results_ece, float) else results_ece[0],
                            "$AUC_{YC}$": result_yc[0],
                            "$STD_{YC}$": result_yc[1],
                            "$MAX_{YC}$": result_yc[2],
                        }
                    )

            df = pd.DataFrame.from_dict(metrics_dict, orient="index")
            df.index = pd.MultiIndex.from_tuples(df.index, names=["Method", "Ensemble", "Aggr."])
            # df = df.swaplevel("Ensemble", "Aggr.")

            # get methods without ensembles
            df_filtered = df.loc[pd.IndexSlice[:, "", :], :]
            # add vyas method
            if ensemble_size > 0:
                df_filtered = pd.concat(
                    [
                        df_filtered,
                        df.loc[
                            pd.IndexSlice[r"MCD Disagreements $U_{\mathrm{MCD}}$ \cite{vyas_analyzing_2019}", :, :], :
                        ],
                    ]
                )

            # for each method get the best Aggr.
            max_indices = df_filtered.groupby(["Method", "Ensemble"])["PRR"].idxmax()
            df_filtered = df.loc[max_indices]

            # another df with all methods using ensembles, use the same aggregations here as in the non ensemble case to avoid a cluttered table,
            # in some cases another aggregation is slightly better, but infsiginificant differences
            # (eg. $1-$max$(p)$ on mcv token level is best with prod aggregation on ensembles (AUCROC=0.857172), but min which is best without ensembles also achieves a AUCROC of 0.850589)
            if ensemble_size > 0:
                df_filtered_ensembles = pd.concat(
                    [df.loc[[(_method[0], "X", _method[2])]] for _method in df_filtered.index], axis=0
                )

            # only plot the best tsallis alpha
            tsallis_indexes = [_index for _index in df_filtered.index.values if "Tsallis" in _index[0]]
            best_tsallis_index = df_filtered.loc[tsallis_indexes]["PRR"].idxmax()
            plot_indexes = [
                _index
                for _index in df_filtered.index.values
                if "Tsallis" not in _index[0] or _index == best_tsallis_index
            ]
            plot_indexes = [
                plot_indexes[_] for _ in [1, 2, 0, 4, 3]
            ]  # change plot order to avoid overlap of curves with legend

            rejection_curve = plt.figure(figsize=(4.5 * figsize_factor, 3.5 * figsize_factor))
            rejection_curve_ax = rejection_curve.add_subplot(1, 1, 1)

            for dict_label in plot_indexes:
                rejection_curve_ax.plot(
                    values_dict[dict_label]["percent_rejected"],
                    values_dict[dict_label]["wer"],
                    values_dict[dict_label]["line_style"],
                    color=values_dict[dict_label]["color"],
                    label=values_dict[dict_label]["legend_label"],
                )
                print(
                    f'{dict_label[0]} rejects {values_dict[dict_label]["percent_rejected"][(np.array(values_dict[dict_label]["wer"])>0.2).sum()]} to detect 80% of errors'
                )

            rejection_curve_ax.plot([0, 1], [1, 0], "k--", label="expected random")
            rejection_curve_ax.plot([0, local_error.sum() / orig_len, 1], [1, 0, 0], "k:", label="oracle")
            rejection_curve_ax.set_xlabel("Part of tokens rejected to oracle")
            rejection_curve_ax.set_ylabel("Part of incorrect tokens remaining")
            rejection_curve_ax.legend(framealpha=0.0, markerfirst=False, fontsize=9)
            # rejection_curve_ax.set_title(f"Rejection curve {split} split")
            rejection_curve.show()
            _filename = f"outputs/{trial_id}_{level}rejection_curve_{split}_set_{dataset}_{ensemble_size}.pgf"
            rejection_curve.savefig(_filename, bbox_inches="tight")
            pgf_files.append(_filename)

            from sklearn.metrics import RocCurveDisplay

            roc_curve = plt.figure(figsize=(4.5 * figsize_factor, 3.5 * figsize_factor))
            roc_curve_ax = roc_curve.add_subplot(1, 1, 1)

            for _, dict_label in enumerate(plot_indexes):
                RocCurveDisplay.from_predictions(
                    values_dict[dict_label]["local_error"],
                    1 - values_dict[dict_label]["y_score"],
                    name=values_dict[dict_label]["legend_label"],
                    color=values_dict[dict_label]["color"],
                    linestyle=values_dict[dict_label]["line_style"],
                    ax=roc_curve_ax,
                    plot_chance_level=True if _ == len(plot_indexes) - 1 else False,
                )
            roc_curve_ax.set_xlabel("False Positive Rate")
            roc_curve_ax.set_ylabel("True Positive Rate")
            legend = roc_curve_ax.legend(framealpha=0.0, markerfirst=False, loc="lower right", fontsize=9)
            for _text in legend.get_texts():
                _text.set_text(re.sub(r"\(AUC = (\d+\.\d+)\)", "", _text.get_text()))
                _text.set_ha("right")
            # roc_curve_ax.set_title("ROC with correct as positive class")
            roc_curve.show()
            _filename = f"outputs/{trial_id}_{level}roc_curve_{split}_set_{dataset}_{ensemble_size}.pgf"
            roc_curve.savefig(_filename, bbox_inches="tight")
            pgf_files.append(_filename)

            youdens_curve = plt.figure(figsize=(4.5 * figsize_factor, 3.5 * figsize_factor))
            youdens_curve_ax = youdens_curve.add_subplot(1, 1, 1)

            for dict_label in plot_indexes:
                result_yc = auc_yc(
                    values_dict[dict_label]["y_true"],
                    values_dict[dict_label]["y_score"],
                    n_bins=1000,
                    return_std_maximum=True,
                    return_curve=True,
                )
                youdens_curve_ax.plot(
                    result_yc[3][0],
                    result_yc[3][1],
                    values_dict[dict_label]["line_style"],
                    color=values_dict[dict_label]["color"],
                    label=values_dict[dict_label]["legend_label"] + f"(AUC={result_yc[0]:.2f}, MAX={result_yc[2]:.2f})",
                )
            # to show how easy the youdens curve metric is "hackable" by just one parameter
            # (tsallis entropy from laptev et al. uses one hyperparam alpha) we just take our confidence to the power of a hyperparam.
            # similar results should be achivable by a more sensible calibration like temperature scaling
            dict_label = (r"Forward-Backward $U_{\mathrm{FB}}$", "", "")
            for hoch in [5, 10, 100]:
                result_yc = auc_yc(
                    values_dict[dict_label]["y_true"],
                    values_dict[dict_label]["y_score"] ** hoch,
                    n_bins=1000,
                    return_std_maximum=True,
                    return_curve=True,
                )
                youdens_curve_ax.plot(
                    result_yc[3][0],
                    result_yc[3][1],
                    "--",
                    color=values_dict[dict_label]["color"],
                    label=values_dict[dict_label]["legend_label"]
                    + f" to the power of {hoch} (AUC={result_yc[0]:.2f}, MAX={result_yc[2]:.2f})",
                )
            youdens_curve_ax.set_xlabel("Threshold")
            youdens_curve_ax.set_ylabel("$Y = $ TNR - FNR")
            legend = youdens_curve_ax.legend(framealpha=0.0, markerfirst=False, fontsize=9)
            youdens_curve.show()

            for _text in legend.get_texts():
                _text.set_text(re.sub(r"\(AUC = (\d+\.\d+)\)", "", _text.get_text()))
                _text.set_ha("right")
            # roc_curve_ax.set_title("ROC with correct as positive class")
            youdens_curve.show()
            _filename = f"outputs/{trial_id}_{level}youdens_curve_{split}_set_{dataset}_{ensemble_size}.pgf"
            youdens_curve.savefig(_filename, bbox_inches="tight")
            pgf_files.append(_filename)

            from sklearn.calibration import CalibrationDisplay

            cal_fig = plt.figure(figsize=(4.5 * figsize_factor, 3.5 * figsize_factor))
            cal_ax = cal_fig.add_subplot(1, 1, 1)

            for dict_label in plot_indexes:

                cal_curve = CalibrationDisplay.from_predictions(
                    # values_dict[dict_label]["y_true"],
                    values_dict[dict_label]["local_error"],
                    1 - values_dict[dict_label]["y_score"],
                    n_bins=10,  # 100,
                    strategy="uniform",  # "quantile",
                    # marker=None,
                    ax=cal_ax,
                    name=values_dict[dict_label]["legend_label"] + f" (ECE={df_filtered.loc[dict_label]['$ECE$']:.3f})",
                    color=values_dict[dict_label]["color"],
                    linestyle=values_dict[dict_label]["line_style"],
                )
            # dict_label = ("Forward-Backward", "", "")
            # for hoch in [5, 10, 100]:
            #     cal_curve = CalibrationDisplay.from_predictions(
            #         values_dict[dict_label]["y_true"],
            #         values_dict[dict_label]["y_score"]**hoch,
            #         n_bins=10,  # 100,
            #         strategy="uniform", # "quantile",
            #         # marker=None,
            #         ax=cal_ax,
            #         name=values_dict[dict_label]["legend_label"] + f" hoch {hoch}",
            #         color=values_dict[dict_label]["color"],
            #         linestyle="--",
            #     )
            legend = cal_ax.legend(framealpha=0.0, markerfirst=False, loc="lower right", fontsize=8)
            cal_ax.set_xlabel("Uncertainty")
            cal_ax.set_ylabel("Fraction of errors")
            for _text in legend.get_texts():
                _text.set_ha("right")
            cal_fig.show()
            _filename = f"outputs/{trial_id}_{level}calibration_curve_{split}_set_{dataset}_{ensemble_size}.pgf"
            cal_fig.savefig(_filename, bbox_inches="tight")
            pgf_files.append(_filename)

            conf_hist = plt.figure(figsize=(4.5 * figsize_factor, 3.5 * figsize_factor))
            conf_hist_ax = conf_hist.add_subplot(1, 1, 1)
            conf_hist_ax.hist(
                [values_dict[dict_label]["y_score"] for dict_label in plot_indexes],
                range=(0, 1),
                bins=10,
                histtype="bar",
                label=[values_dict[dict_label]["legend_label"] for dict_label in plot_indexes],
                color=[values_dict[dict_label]["color"] for dict_label in plot_indexes],
                linestyle=values_dict[dict_label]["line_style"],
            )
            num_correct = list(values_dict.values())[0]["y_true"].sum()
            num_incorrect = np.logical_not(list(values_dict.values())[0]["y_true"]).sum()
            conf_hist_ax.plot([0, 1], [num_correct, num_correct], "k-", label="correct predictions")
            conf_hist_ax.plot([0, 1], [num_incorrect, num_incorrect], "k--", label="incorrect predictions")
            conf_hist_ax.legend(framealpha=0.0, markerfirst=False, fontsize=9)
            conf_hist.show()

            df_filtered.to_pickle(f"outputs/{trial_id}_{level}dataframe_{split}_set_{dataset}_{ensemble_size}.pkl")
            if ensemble_size > 0:
                df_filtered_ensembles.to_pickle(
                    f"outputs/{trial_id}_{level}dataframe_ensembles_{split}_set_{dataset}_{ensemble_size}.pkl"
                )

            print("\n")
            print(
                df_filtered.sort_index(level=0)
                .style.format(precision=2)
                .highlight_max(
                    subset=[
                        "PRR",
                        "$AUC_{ROC}$",
                        "$AUC_{PR}$",
                        "$AUC_{NT}$",
                        "$NCE$",
                        "$AUC_{YC}$",
                        "$STD_{YC}$",
                        "$MAX_{YC}$",
                    ],
                    axis=0,
                    props="textbf:--rwrap;",
                )
                .highlight_min(subset=["$ECE$"], axis=0, props="textbf:--rwrap;")
                .hide(level=["Ensemble"])
                .hide(
                    subset=["PRR", "$AUC_{PR}$", "$ECE$", "$NCE$", "$AUC_{YC}$", "$STD_{YC}$", "$MAX_{YC}$"],
                    axis="columns",
                )
                .to_latex(
                    label=f"tab:{'word' if level=='word_' else 'token'}",
                    caption=f"{dataset}: {'Word' if level=='word_' else 'Token'} level uncertainty estimation without ensembles. For methods that need aggregation from frame to token level, the best aggregation method is shown.",
                    environment="table",
                    hrules=True,
                    # clines="skip-last;data",
                )
                .replace("\\\\\n\\multirow", "\\\\\\hline\n\\multirow")
            )

            if ensemble_size > 0:
                print("\n")
                print(
                    df_filtered_ensembles.sort_index(level=0)
                    .style.format(precision=2)
                    .highlight_max(
                        # subset=[
                        #     "PRR",
                        #     "$AUC_{ROC}$",
                        #     "$AUC_{PR}$",
                        #     "$AUC_{NT}$",
                        #     "$NCE$",
                        #     "$AUC_{YC}$",
                        #     "$STD_{YC}$",
                        #     "$MAX_{YC}$",
                        # ],
                        axis=0,
                        props="textbf:--rwrap;",
                    )
                    # .highlight_min(subset=["$ECE$"], axis=0, props="textbf:--rwrap;")
                    .hide(level=["Ensemble"])
                    .hide(
                        subset=["PRR", "$AUC_{PR}$", "$ECE$", "$NCE$", "$AUC_{YC}$", "$STD_{YC}$", "$MAX_{YC}$"],
                        axis="columns",
                    )
                    .to_latex(
                        label=f"tab:{'word' if level=='word_' else 'token'}_ensembles",
                        caption=f"{dataset}: {'Word' if level=='word_' else 'Token'} level uncertainty estimation with ensembles. For methods that need aggregation from frame to token and to word level, the best aggregation method from frame to token level is shown, aggregation from token to word level is done using product aggregation for these methods.",
                        environment="table",
                        hrules=True,
                        # clines="skip-last;data",
                    )
                    .replace("\\\\\n\\multirow", "\\\\\\hline\n\\multirow")
                )

    # matplotlib escapes underscores in the citation keys, replace them with regular underscores
    fix_citations_in_pgf_files(pgf_files)

    # %%


# %%


def analyse_framewise(decodings):
    # %%
    p_max_unc = np.concatenate(
        [np.array([frame["p_max"] for frame in sample["no_dropout_debug"]]) for sample in decodings.values()]
    )
    interspeech23 = np.concatenate(
        [
            np.array([frame["sum_of_tokens_that_change"] for frame in sample["no_dropout_debug"]])
            for sample in decodings.values()
        ]
    )
    nll = np.concatenate(
        [np.array([frame["nll"] for frame in sample["no_dropout_debug"]]) for sample in decodings.values()]
    )

    import math
    import torch

    def compute_tsallis(_nll):
        alpha = 0.8
        v = len(_nll)
        neg_entropy_alpha = (_nll * alpha).exp().sum()
        exp_neg_max_ent = math.exp((1 - math.pow(v, 1 - alpha)) / (1 - alpha))
        uncertainty = 1 - (((1 - neg_entropy_alpha) / (1 - alpha)).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)
        return uncertainty

    # %%
    nll_sorted = np.sort(nll, axis=1)[:, ::-1]

    bar_ind = np.arange(8)
    bar_width = 0.4

    figsize_factor = 1.1
    fig = plt.figure(figsize=(4.5 * figsize_factor, 3.5 * figsize_factor))
    ax = fig.add_subplot(1, 1, 1)

    # ax.bar(bar_ind, p_sorted_mean[:10], bar_width, label="all")

    # take the 1000 frames with the lowest p_max (highest p_max uncertainty):
    frames_high_p_max = np.argsort(p_max_unc)[::-1][:1000]  # int(len(p_max_unc) / 50)]

    # from these, take the 100 frames with the lowest interspeech23 uncertainty:
    frames_low_unc = np.argsort(interspeech23[frames_high_p_max])[:100]
    # and those 100 with the highest interspeech23 uncertainty
    frames_high_unc = np.argsort(interspeech23[frames_high_p_max])[::-1][:100]

    p_sorted_mean_interspeech_small_p_max_high = np.exp(nll_sorted[frames_high_p_max][frames_low_unc]).mean(axis=0)
    ax.bar(
        bar_ind - bar_width * 0.5,
        p_sorted_mean_interspeech_small_p_max_high[:8] * -np.log(p_sorted_mean_interspeech_small_p_max_high[:8]),
        bar_width,
        label="low uncertainty with \cite{rumberg_uncertainty_2023}",
    )
    print(
        f"mean tsallis for frame with p_max uncertainty high: {compute_tsallis(torch.tensor(p_sorted_mean_interspeech_small_p_max_high).log())}"
    )

    p_sorted_mean_interspeech_high_p_max_high = np.exp(nll_sorted[frames_high_p_max][frames_high_unc]).mean(axis=0)
    ax.bar(
        bar_ind + bar_width * 0.5,
        p_sorted_mean_interspeech_high_p_max_high[:8] * -np.log(p_sorted_mean_interspeech_high_p_max_high[:8]),
        bar_width,
        label="high uncertainty with \cite{rumberg_uncertainty_2023}",
    )
    print(
        f"mean tsallis for frame with both uncertainties high: {compute_tsallis(torch.tensor(p_sorted_mean_interspeech_high_p_max_high).log())}"
    )

    ax.set_ylabel("$-p*\mathrm{log}(p)$")
    ax.set_xlabel("index sorted by descending $p$")
    legend = ax.legend(framealpha=0.0, markerfirst=False, fontsize=9)
    for _text in legend.get_texts():
        _text.set_ha("right")
    fig.show()

    _filename = f"outputs/timit_fame_wise_analysis.pgf"
    fig.savefig(_filename, bbox_inches="tight")
    fix_citations_in_pgf_files([_filename])


# %%
