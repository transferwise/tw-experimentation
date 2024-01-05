from tw_experimentation.utils import ExperimentDataset


from tw_experimentation.checker import (
    Monitoring,
    SegmentMonitoring,
    SequentialTest,
    NormalityChecks,
)

from tw_experimentation.statistical_tests import compute_frequentist_results, run_cuped
from tw_experimentation.segmentation_frequentist import run_segmentation_analysis


from tw_experimentation.bayes.bayes_test import BayesTest


from typing import List, Optional
import pandas as pd
import pickle as pkl


def generate_results(
    df: pd.DataFrame,
    variant: str,
    targets: List[str],
    date_created: Optional[str] = None,
    pre_experiment_cols: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
    remove_outliers=True,
):
    """Compute all results for an experiment

    Args:
        df (pd.DataFrame): dataframe with experiment data
        variant (str): variant name
        targets (List[str]): target names
        date_created (Optional[str], optional): timestamp column name. Defaults to None.
        pre_experiment_cols (Optional[List[str]], optional): pre-experiment outcome metrics. Defaults to None.
        segments (Optional[List[str]], optional): segments for monitoring and segmentation analysis.
            Defaults to None.
        remove_outliers (bool, optional): Whether to remove outliers from continuous variables.
            Defaults to True.

    Returns:
        Dict: Output plots, tables etc.
    """
    output = dict()

    # Data model setup
    ed = ExperimentDataset(
        data=df,
        variant=variant,
        targets=targets,
        date=date_created,
        pre_experiment_cols=pre_experiment_cols,
        segments=segments,
    )
    ed.preprocess_dataset(remove_outliers=remove_outliers)
    emd = ed.experiment_meta_data()
    output["emd"] = emd

    # Monitoring and integrity checks
    monitor = Monitoring(ed)
    output["monitor_results"] = monitor.create_tables_and_plots()

    if isinstance(segments, list) and len(segments) > 0:
        segment_monitor = SegmentMonitoring(ed, segments)
        CHI_SQUAERED_ALPHA = 0.05
        output["segment_monitor_results"] = segment_monitor.create_tables_and_plots(
            chi_squared_alpha=CHI_SQUAERED_ALPHA
        )
    else:
        output["segment_monitor_results"] = None

    nc = NormalityChecks(ed)
    ALPHA_NORMALITY_CHECK = 0.05
    output["nco"] = nc.create_results(alpha=ALPHA_NORMALITY_CHECK)

    sequential_test = SequentialTest(ed)
    _ = sequential_test.sequential_test_results()
    output["fig_sequential_test"] = sequential_test.fig_sequential_test()

    # Frequentist results
    output["frequentist_test_results"] = compute_frequentist_results(ed)
    if ed.pre_experiment_cols is not None and len(ed.pre_experiment_cols) > 0:
        output["cuped"] = run_cuped(ed)
    if isinstance(segments, list) and len(segments) > 0:
        output["segmentation_analysis"] = run_segmentation_analysis(
            ed, segments=ed.segments
        )

    # Bayes results
    bt = BayesTest(ed=ed)
    output["bayes_result"] = bt.compute_posterior()

    return output


def save_output(path: str, name: str, output: dict):
    """Save output to pickle file

    Args:
        path (str): path name
        name (str): file name
        output (dict): output dict to save
    """
    with open(path + name + ".pkl", "wb") as f:
        pkl.dump(output, f)


def load_output(path: str, name: str):
    with open(path + name + ".pkl", "rb") as file:
        # Load the object from the file
        output = pkl.load(file)

    return output
