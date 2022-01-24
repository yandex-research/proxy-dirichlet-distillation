import json
import os
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def parse_uncertainty_metrics(f, report_nbest=None):
    result = dict()
    for line in f:
        stripped_line = line.strip()
        group_start = re.search(r"N-BEST\s+(\d+)", stripped_line)
        if group_start:
            n = int(group_start.group(1))
        else:
            metric_res = re.search(r"using\s*(.*):\s*(-?\d+.\d+)", stripped_line)
            metric = metric_res.group(1)
            value = float(metric_res.group(2))
            if (
                not metric.endswith("EP")
                and not metric.endswith("ep_MKL")
                and not metric in ("var", "varcombo", "logcombo", "logvar")
                and (report_nbest is None or n in report_nbest)
            ):
                result[f"{n}_{metric}"] = value
    return result


def main(decode_path, reference_path, output_path):
    wer_stats = {
        "test-clean WER": float(np.loadtxt(decode_path / "wer-ltc", dtype=np.float64)),
        "test-other WER": float(np.loadtxt(decode_path / "wer-lto", dtype=np.float64)),
        "AMI WER": float(np.loadtxt(decode_path / "wer-ami", dtype=np.float64)),
    }

    scores_clean = np.loadtxt(
        reference_path / "test-clean" / "score.txt", dtype=np.float64
    )
    scores_other = np.loadtxt(
        reference_path / "test-other" / "score.txt", dtype=np.float64
    )
    scores_ami = np.loadtxt(reference_path / "ami" / "score.txt", dtype=np.float64)

    ood_metrics = dict()
    for testset in "test-other", "ami", "cv-fr":
        path_decode = decode_path / testset / "results_ood_bs.txt"
        path_reference = reference_path / testset / "results_ood_mc.txt"
        if os.path.exists(path_decode) and os.path.exists(path_reference):
            with open(decode_path / testset / "results_ood_bs.txt") as f:
                decode_bs_seq = parse_uncertainty_metrics(f, report_nbest=(20,))
            with open(decode_path / testset / "results_ood_mc.txt") as f:
                decode_mc_seq = parse_uncertainty_metrics(f, report_nbest=(20,))

            ood_metrics.update(
                {
                    **{
                        f"ood_{testset}_decode_bs_{key}": value
                        for key, value in decode_bs_seq.items()
                    },
                    **{
                        f"ood_{testset}_decode_mc_{key}": value
                        for key, value in decode_mc_seq.items()
                    },
                }
            )
        else:
            if not os.path.exists(path_decode):
                print(f"{path_decode} missing, skipping...")
            if not os.path.exists(path_reference):
                print(f"{path_reference} missing, skipping...")

    seq_error_metrics = dict()
    for testset in "test-clean", "test-other", "ami":
        with open(decode_path / testset / "results_seq_mc.txt") as f:
            decode_mc_seq = parse_uncertainty_metrics(f, report_nbest=(1,))

        seq_error_metrics.update(
            {
                **{
                    f"seq_error_{testset}_{key}": value
                    for key, value in decode_mc_seq.items()
                },
            }
        )

    result_dict = {
        **wer_stats,
        "test-clean NLL": scores_clean.mean(),
        "test-other NLL": scores_other.mean(),
        "ami NLL": scores_ami.mean(),
        **ood_metrics,
        **seq_error_metrics,
    }
    with open(output_path, "w+") as output_f:
        json.dump(result_dict, output_f, indent=2)
        output_f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--decode-path", required=True, type=Path)
    parser.add_argument("-r", "--reference-path", required=True, type=Path)
    parser.add_argument("-o", "--output-path", required=True, type=Path)
    args = parser.parse_args()
    main(args.decode_path, args.reference_path, args.output_path)
