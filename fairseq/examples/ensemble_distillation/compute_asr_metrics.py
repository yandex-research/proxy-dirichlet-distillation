import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


def main(
    input_dir,
    source_dir,
    output_dir,
    json_output_path,
    checkpoint_file,
    max_tokens,
    max_sentences,
    model_offset,
):
    os.makedirs(os.path.join(output_dir, "decode"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(input_dir, "data", "data"), exist_ok=True)

    for dataset in "test-clean", "test-other", "ami", "cv-fr":
        src_path = Path(input_dir) / "data" / dataset
        tgt_path = Path(input_dir) / "data" / "data" / dataset
        # it is assumed that input_dir/data holds both json files and actual audio data
        if not tgt_path.exists():
            os.symlink(src_path, tgt_path)

        if dataset == "ami":
            cmd = f"--max-sentences {max_sentences} "
        else:
            cmd = f"--max-tokens {max_tokens} "
        cmds = (
            f"fairseq-generate "
            f"{input_dir}/data "
            f"--path {checkpoint_file} "
            f"--num-workers 8 "
            f"{cmd}"
            f"--user-dir {source_dir}/examples/ensemble_distillation "
            f"--task asr_distillation "
            f"--compute-uncertainty "
            f"--model-offset {model_offset} "
            f"--gen-subset {dataset} "
            f"--score-reference "
            f"> {output_dir}/reference/results-{dataset}.txt "
        )
        subprocess.run(
            cmds, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr
        )

        cmds = (
            f"fairseq-generate "
            f"{input_dir}/data "
            f"--path {checkpoint_file} "
            f"--num-workers 8 "
            f"{cmd}"
            f"--user-dir {source_dir}/examples/ensemble_distillation "
            f"--task asr_distillation "
            f"--compute-uncertainty "
            f"--model-offset {model_offset} "
            f"--gen-subset {dataset} "
            f"--beam 20 --nbest 20 "
            f"> {output_dir}/decode/results-{dataset}.txt "
        )
        subprocess.run(
            cmds, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr
        )

    uncert_cmds = f"bash {os.path.abspath(source_dir)}/examples/ensemble_distillation/process_uncertainty_asr.sh"
    subprocess.run(
        uncert_cmds,
        shell=True,
        check=True,
        cwd=output_dir,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=dict(
            os.environ,
            CODE_DIR=os.path.abspath(source_dir),
            TMP_OUTPUT_PATH=os.path.abspath(output_dir),
        ),
    )

    parse_cmds = (
        f"python3 {source_dir}/fairseq/examples/ensemble_distillation/collect_asr_json_metrics.py "
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
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-sentences", type=int, default=1)
    parser.add_argument("--model-offset", type=float, default=0)
    args = parser.parse_args()
    main(
        args.input_dir,
        args.source_dir,
        args.output_dir,
        args.json_output,
        args.checkpoint_file,
        args.max_tokens,
        args.max_sentences,
        args.model_offset,
    )
