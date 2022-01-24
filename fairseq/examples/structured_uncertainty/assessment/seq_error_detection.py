import argparse
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from sacrebleu import corpus_bleu, sentence_bleu
from sklearn.metrics import auc

from examples.structured_uncertainty.assessment.utils import load_uncertainties

sns.set()

parser = argparse.ArgumentParser(description='Assess ood detection performance')
parser.add_argument('path', type=str,
                    help='Path of directory containing in-domain uncertainties.')
parser.add_argument('--wer', action='store_true',
                    help='Whether to evaluate using WER instead of BLEU')
parser.add_argument('--beam_width', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--nbest', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--beam_search', action='store_true',
                    help='Path of directory where to save results.')
parser.add_argument('--calibrate', action='store_true',
                    help='Path of directory where to save results.')


def get_wer(i, path):
    subprocess.run(f"~/sclite -r {path}/sorted_refs.txt -h {path}/hypos_{i}.txt -i rm -o all", shell=True)
    return np.float(subprocess.run(
        f"grep 'Sum/Avg' {path}/hypos_{i}.txt.sys | sed 's/|//g' | sed 's/\+/ /g' | awk -F ' ' '{{print $8}}' ",
        capture_output=True, shell=True).stdout)


def get_sentence_wer(path, refs, hypos):
    if not os.path.exists(f"{path}/wers.txt"):
        print("Getting WERS...)")
        subprocess.run(f"~/sclite -r {path}/refs.txt -h {path}/hypos.txt -i rm -o all", shell=True)
    subprocess.run(
        f" grep -v '-'  {path}/hypos.txt.sys | egrep -v '[A-Z]' | egrep -v '=' | sed 's/|//g' | egrep '[0-9]+' | awk '{{print $8}}' > {path}/wers.txt",
        shell=True)

    return np.loadtxt(f"{path}/wers.txt", dtype=np.float32)


def reject_predictions_wer(refs, hypos, measure, path,
                           rev: bool = False,
                           corpus_wer: bool = False):  # , measure_name: str, save_path: str,  show=True):

    refs = np.asarray(refs)
    hypos = np.asarray(hypos)

    if rev:
        inds = np.argsort(measure)
    else:
        inds = np.argsort(measure)[::-1]

    assert measure[inds[0]] >= measure[inds[-1]]

    swers = get_sentence_wer(path, refs, hypos)
    sorted_swers = swers[inds]

    total_data = np.float(len(hypos))
    wers, percentages = [], []

    if corpus_wer:
        wers = Parallel(n_jobs=-1)(delayed(get_wer)(i, path) for i in range(len(hypos)))
    else:
        min_swers = np.zeros_like(sorted_swers)
        wers = [np.mean(np.concatenate((min_swers[:i], sorted_swers[i:]), axis=0)) for i in range(len(hypos))]

    for i in range(len(hypos)):
        percentages.append(float(i + 1) / total_data * 100.0)

    wers, percentages = np.asarray(wers), np.asarray(percentages)

    base_wer = np.mean(swers)
    n_items = wers.shape[0]

    random_wer = np.asarray([base_wer * (1.0 - float(i) / float(n_items)) for i in range(n_items)],
                            dtype=np.float32)

    sorted_swers = np.sort(swers)[::-1]
    perfect_wer = np.asarray(
        [np.mean(np.concatenate((min_swers[:i], sorted_swers[i:]), axis=0)) for i in range(len(hypos))])

    auc_rnd = 1.0 - auc(percentages / 100.0, random_wer / 100.0)
    auc_uns = 1.0 - auc(percentages / 100.0, wers / 100.0)
    auc_orc = 1.0 - auc(percentages / 100.0, perfect_wer / 100.0)
    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0
    return wers, random_wer, percentages, perfect_wer, rejection_ratio


def get_corpus_bleu(refs, hypos, i):
    tmp_hypo = np.concatenate((refs[:i], hypos[i:]), axis=0)
    return corpus_bleu(tmp_hypo, [refs]).score


def get_bleu(refs, hypos, i):
    tmp_hypo = np.concatenate((refs[:i], hypos[i:]), axis=0)
    return np.mean([sentence_bleu([tmp_hypo[i]], [[refs[i]]], smooth_value=1.0).score for i in range(hypos.shape[0])])


def reject_predictions_bleu(refs, hypos, measure, path,
                            rev: bool = False, corpus_bleu=False):  # , measure_name: str, save_path: str,  show=True):
    # Get predictions

    refs = np.asarray(refs)
    hypos = np.asarray(hypos)

    if rev:
        inds = np.argsort(measure)
    else:
        inds = np.argsort(measure)[::-1]

    assert measure[inds[0]] >= measure[inds[-1]]

    sorted_refs = refs[inds]
    sorted_hypos = hypos[inds]

    if not os.path.exists(f"{path}/sbleus.txt"):
        sbleus = np.asarray(
            [sentence_bleu([hypos[i]], [[refs[i]]], smooth_value=1.1).score for i in range(hypos.shape[0])])
        np.savetxt(f"{path}/sbleus.txt", sbleus)
    else:
        sbleus = np.loadtxt(f"{path}/sbleus.txt", dtype=np.float32)

    sorted_sbleus = sbleus[inds]

    total_data = np.float(len(hypos))
    bleus, percentages = [], []

    if corpus_bleu:
        bleus = Parallel(n_jobs=-1)(delayed(get_bleu)(sorted_refs, sorted_hypos, i) for i in range(len(hypos)))
    else:
        max_sbleus = 100 * np.ones_like(sorted_sbleus)
        bleus = [np.mean(np.concatenate((max_sbleus[:i], sorted_sbleus[i:]), axis=0)) for i in range(len(hypos))]

    for i in range(len(hypos)):
        percentages.append(float(i + 1) / total_data * 100.0)

    bleus, percentages = np.asarray(bleus), np.asarray(percentages)

    base_bleu = bleus[0]
    n_items = bleus.shape[0]

    random_bleu = np.asarray(
        [base_bleu * (1.0 - float(i) / float(n_items)) + 100.0 * (float(i) / float(n_items)) for i in range(n_items)],
        dtype=np.float32)

    sorted_sbleus = np.sort(sbleus)
    perfect_bleu = np.asarray(
        [np.mean(np.concatenate((max_sbleus[:i], sorted_sbleus[i:]), axis=0)) for i in range(len(hypos))])

    auc_rnd = 1.0 - auc(percentages / 100.0, random_bleu / 100.0)
    auc_uns = 1.0 - auc(percentages / 100.0, bleus / 100.0)
    auc_orc = 1.0 - auc(percentages / 100.0, perfect_bleu / 100.0)
    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0

    return bleus, random_bleu, percentages, perfect_bleu, rejection_ratio


def load_text(path, beam_width=5):
    refs, hypos = [], []
    with open(os.path.join(path, 'refs.txt'), 'r') as f:
        for line in f.readlines():
            refs.append(line[1:-1])

    with open(os.path.join(path, 'hypos.txt'), 'r') as f:
        count = 0
        for line in f.readlines():
            if count % beam_width == 0:
                hypos.append(line[1:-1])
            count += 1

    return refs, hypos


def main():
    args = parser.parse_args()
    if args.calibrate:
        for temp in np.arange(1.0, 51.0, 1.0):
            uncertainties = load_uncertainties(args.path, beam_width=args.beam_width, n_best=args.nbest,
                                               beam_search=True, temp=temp)
            out_dict = {}
            if args.wer:
                refs, hypos = load_text(args.path, beam_width=1)
                for key in uncertainties.keys():
                    wer, random_wer, percentage, perfect_wer, auc_rr = reject_predictions_wer(refs, hypos,
                                                                                              uncertainties[key],
                                                                                              args.path)
                    out_dict[key] = [wer, percentage, random_wer, perfect_wer, auc_rr]
                plt.plot(out_dict[key][1], out_dict[key][3], 'r--', lw=2)
                for key in out_dict.keys():
                    if key == 'SCR-PE':
                        plt.plot(out_dict[key][1], out_dict[key][0])
                plt.plot(out_dict[key][1], out_dict[key][2], 'k--', lw=2)
                plt.ylim(0.0, out_dict[key][2][0])
                plt.xlim(0.0, 100.0)
                plt.xlabel('Percentage Rejected')
                plt.ylabel('WER (%)')
                plt.legend(['Oracle', 'Joint-Seq TU', 'Expected Random'])
                plt.savefig(os.path.join(args.path, 'seq_reject_temp.png'), bbox_inches='tight', dpi=300)
                plt.close()

            else:
                refs, hypos = load_text(args.path, beam_width=args.beam_width)
                for key in uncertainties.keys():
                    bleus, random_bleu, percentage, perfect_bleu, auc_rr = reject_predictions_bleu(refs, hypos,
                                                                                                   uncertainties[key],
                                                                                                   args.path)
                    out_dict[key] = [bleus, percentage, random_bleu, perfect_bleu, auc_rr]
                plt.plot(out_dict[key][1], out_dict[key][3], 'r--', lw=2)
                for key in out_dict.keys():
                    if key == 'SCR-PE':
                        plt.plot(out_dict[key][1], out_dict[key][0])
                plt.plot(out_dict[key][1], out_dict[key][2], 'k--', lw=2)
                plt.ylim(out_dict[key][2][0], 100.0)
                plt.xlim(0.0, 100.0)
                plt.xlabel('Percentage Rejected')
                plt.ylabel('Bleu')
                plt.legend(['Oracle', 'Joint-Seq TU', 'Expected Random'])
                plt.savefig(os.path.join(args.path, 'seq_reject_temp.png'), bbox_inches='tight', dpi=300)
                plt.close()

            results = 'results_seq_calibrate.txt'
            with open(os.path.join(args.path, results), 'a') as f:
                f.write(f'--SEQ ERROR DETECT N-BEST  {args.nbest} BW: {args.beam_width}, BS:{True}, Temp:{temp}--\n')
                for key in out_dict.keys():
                    f.write('AUC-RR using ' + key + ": " + str(np.round(out_dict[key][-1], 1)) + '\n')
    else:
        uncertainties = load_uncertainties(args.path, beam_width=args.beam_width, n_best=args.nbest,
                                           beam_search=args.beam_search)
        out_dict = {}
        if args.wer:
            refs, hypos = load_text(args.path, beam_width=1)
            for key in uncertainties.keys():
                wer, random_wer, percentage, perfect_wer, auc_rr = reject_predictions_wer(refs, hypos,
                                                                                          uncertainties[key],
                                                                                          args.path)
                out_dict[key] = [wer, percentage, random_wer, perfect_wer, auc_rr]
            plt.plot(out_dict[key][1], out_dict[key][3], 'r--', lw=2)
            for key in out_dict.keys():
                if key == 'SCR-PE':
                    plt.plot(out_dict[key][1], out_dict[key][0])
            plt.plot(out_dict[key][1], out_dict[key][2], 'k--', lw=2)
            plt.ylim(0.0, out_dict[key][2][0])
            plt.xlim(0.0, 100.0)
            plt.xlabel('Percentage Rejected')
            plt.ylabel('WER (%)')
            plt.legend(['Oracle', 'Joint-Seq TU', 'Expected Random'])
            plt.savefig(os.path.join(args.path, 'seq_reject.png'), bbox_inches='tight', dpi=300)
            plt.close()

        else:
            refs, hypos = load_text(args.path, beam_width=args.beam_width)
            for key in uncertainties.keys():
                bleus, random_bleu, percentage, perfect_bleu, auc_rr = reject_predictions_bleu(refs, hypos,
                                                                                               uncertainties[key],
                                                                                               args.path)
                out_dict[key] = [bleus, percentage, random_bleu, perfect_bleu, auc_rr]
            plt.plot(out_dict[key][1], out_dict[key][3], 'r--', lw=2)
            for key in out_dict.keys():
                if key == 'SCR-PE':
                    plt.plot(out_dict[key][1], out_dict[key][0])
            plt.plot(out_dict[key][1], out_dict[key][2], 'k--', lw=2)
            plt.ylim(out_dict[key][2][0], 100.0)
            plt.xlim(0.0, 100.0)
            plt.xlabel('Percentage Rejected')
            plt.ylabel('Bleu')
            plt.legend(['Oracle', 'Joint-Seq TU', 'Expected Random'])
            plt.savefig(os.path.join(args.path, 'seq_reject.png'), bbox_inches='tight', dpi=300)
            plt.close()

        if args.beam_search:
            results = 'results_seq_bs.txt'
        else:
            results = 'results_seq_mc.txt'
        with open(os.path.join(args.path, results), 'a') as f:
            f.write(f'--SEQ ERROR DETECT N-BEST  {args.nbest} BW: {args.beam_width}, BS:{args.beam_search}--\n')
            for key in out_dict.keys():
                f.write('AUC-RR using ' + key + ": " + str(np.round(out_dict[key][-1], 1)) + '\n')


if __name__ == '__main__':
    main()
