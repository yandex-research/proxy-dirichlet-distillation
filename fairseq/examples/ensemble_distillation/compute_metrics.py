import os
import subprocess
from argparse import ArgumentParser


def main(
    input_dir,
    source_dir,
    output_dir,
    json_output_path,
    checkpoint_file,
    max_tokens,
    model_offset,
    no_distillation,
):
    os.makedirs(os.path.join(output_dir, "decode"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reference"), exist_ok=True)

    task_option = (
        ""
        if no_distillation
        else f"--task distillation --model-offset {model_offset} "
    )

    for dataset in (
        "test",
        "test5",
        "test6",
        "test9",
        "test12",
        "test14",
        "test_ens_pred",
    ):
        gen_cmds = (
            f"fairseq-generate "
            f"{input_dir}/data "
            f"--path {checkpoint_file} "
            f"--user-dir {source_dir}/examples/ensemble_distillation "
            f"--fp16 "
            f"{task_option}"
            f"--max-tokens {max_tokens} "
            f"--remove-bpe "
            f"--sacrebleu "
            f"--compute-uncertainty "
            f"--nbest 5 "
            f"--gen-subset {dataset} "
            f"> {os.path.join(output_dir, 'decode', f'results-{dataset}.txt')}"
        )
        subprocess.run(gen_cmds, shell=True, check=True)

        gen_cmds = (
            f"fairseq-generate "
            f"{input_dir}/data "
            f"--path {checkpoint_file} "
            f"--user-dir {source_dir}/examples/ensemble_distillation "
            f"--fp16 "
            f"{task_option}"
            f"--max-tokens {max_tokens} "
            f"--remove-bpe "
            f"--sacrebleu "
            f"--compute-uncertainty "
            f"--score-reference "
            f"--gen-subset {dataset} "
            f"> {os.path.join(output_dir, 'reference', f'results-{dataset}.txt')}"
        )
        subprocess.run(gen_cmds, shell=True, check=True)

    uncert_cmds = f"bash {os.path.abspath(source_dir)}/examples/ensemble_distillation/process_uncertainty.sh"
    subprocess.run(
        uncert_cmds,
        shell=True,
        check=True,
        cwd=output_dir,
        env=dict(
            os.environ,
            CODE_DIR=os.path.abspath(source_dir),
            TMP_OUTPUT_PATH=os.path.abspath(output_dir),
        ),
    )

    parse_cmds = (
        f"python3 {source_dir}/examples/ensemble_distillation/collect_json_metrics.py "
        f"--decode-path {os.path.join(output_dir, 'decode')} "
        f"--reference-path  {os.path.join(output_dir, 'reference')} "
        f"--output-path {json_output_path if json_output_path is not None else os.path.join(output_dir, 'json_output.json')} "
    )
    subprocess.run(parse_cmds, shell=True, check=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", "-i", required=True)
    parser.add_argument("--source-dir", "-s", required=True)
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--json-output", "-j")
    parser.add_argument("--checkpoint-file", "-c", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--model-offset", type=float, default=0)
    parser.add_argument("--no-distillation", action="store_true")
    args = parser.parse_args()
    main(
        args.input_dir,
        args.source_dir,
        args.output_dir,
        args.json_output,
        args.checkpoint_file,
        args.max_tokens,
        args.model_offset,
        args.no_distillation,
    )
