# %%
import pandas as pd
from itertools import product

# %%

if __name__ == "__main__":

    # %%
    ensemble_size = 50
    trial_id = dict(mcv="240207", timit=240207)

    key_table_header_map = dict(
        mcv_word_=("MCV", "Words", ""),
        mcv_word__ensemble=("MCV", "Words", "Ens."),
        mcv_=("MCV", "Tokens", ""),
        mcv__ensemble=("MCV", "Tokens", "Ens."),
        timit_=("TIMIT", "Tokens", ""),
        timit__ensemble=("TIMIT", "Tokens", "Ens."),
    )

    dfs = dict()
    dfs_ensemble = dict()
    for dataset in ("mcv", "timit"):
        for level in ("word_", ""):
            if dataset == "timit" and level == "word_":
                continue
            dfs[key_table_header_map[dataset + "_" + level]] = pd.read_pickle(
                f"outputs/{trial_id[dataset]}_{level}dataframe_test_set_{dataset}_{ensemble_size}.pkl"
            ).loc[pd.IndexSlice[:, "", :]]
            dfs[key_table_header_map[dataset + "_" + level + "_ensemble"]] = pd.read_pickle(
                f"outputs/{trial_id[dataset]}_{level}dataframe_ensembles_test_set_{dataset}_{ensemble_size}.pkl"
            ).loc[pd.IndexSlice[:, "X", :]]

    df = pd.concat([df for df in dfs.values()], axis=1, keys=[key for key in dfs.keys()])
    print(
        df.reorder_levels([3, 0, 1, 2], axis="columns")
        .sort_index(axis="columns", ascending=[False, True, True, True])
        .style.format(precision=2, na_rep="-")
        .highlight_max(
            axis=0,
            props="textbf:--rwrap;",
        )
        # .highlight_min(subset=product(dfs.keys(), ["$ECE$"]), axis=0, props="textbf:--rwrap;")
        .hide(
            subset=[
                tuple([metric]) + key
                for key in dfs.keys()
                for metric in [
                    "PRR",
                    "$AUC_{PR}$",
                    # "$AUC_{NT}$",
                    "$ECE$",
                    "$NCE$",
                    "$AUC_{YC}$",
                    "$STD_{YC}$",
                    "$MAX_{YC}$",
                ]
            ],
            axis="columns",
        )
        # .hide(level=3, axis="columns")
        .to_latex(
            label="tab:auc_roc",
            caption="$AUC_{ROC}$",
            column_format="ll|cc|cc|cc|cc|cc|cc",
            environment="table*",
            hrules=True,
            # clines="skip-last;data",
            multicol_align="c|",
        )
        .replace(r"\begin{tabular}{", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}")
        .replace(r"\end{tabular}", r"\end{tabular*}")
        .replace("\\\\\n\\multirow", "\\\\\\hline\n\\multirow")
    )

# %%
