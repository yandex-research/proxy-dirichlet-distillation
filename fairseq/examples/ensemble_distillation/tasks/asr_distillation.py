import os

import torch

from examples.ensemble_distillation.utils import (
    get_dirichlet_parameters,
)
from examples.speech_recognition.data.replabels import replabel_symbol
from examples.speech_recognition.tasks.speech_recognition import SpeechRecognitionTask
from fairseq import checkpoint_utils
from fairseq.data import Dictionary
from fairseq.data.data_utils import collate_tokens
from fairseq.tasks import register_task
from fairseq.uncertainty import (
    compute_token_dirichlet_uncertainties,
    compute_sequence_dirichlet_uncertainties,
)

EPS = 1e-10


@register_task("asr_distillation")
class ASRDistillationTask(SpeechRecognitionTask):
    @staticmethod
    def add_args(parser):
        SpeechRecognitionTask.add_args(parser)
        parser.add_argument(
            "--ensemble-paths", help="Paths to ensemble models for distillation"
        )
        parser.add_argument("--model-offset", default=0, type=float)
        parser.add_argument("--fp16-ensemble", action="store_true")

    def __init__(self, args, tgt_dict, models):
        super().__init__(args, tgt_dict)
        self.ensemble = models

        self.model_offset = args.model_offset

        self.criterion = args.criterion
        self.fp16_ensemble = args.fp16_ensemble
        self.compute_uncertainty = getattr(args, "compute_uncertainty", False)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dict_path = os.path.join(args.data, "dict.txt")
        if not os.path.isfile(dict_path):
            raise FileNotFoundError("Dict not found: {}".format(dict_path))
        tgt_dict = Dictionary.load(dict_path)

        if args.criterion == "ctc_loss":
            tgt_dict.add_symbol("<ctc_blank>")
        elif args.criterion == "asg_loss":
            for i in range(1, args.max_replabel + 1):
                tgt_dict.add_symbol(replabel_symbol(i))

        print("| dictionary: {} types".format(len(tgt_dict)))

        if args.ensemble_paths is not None:
            # Load ensemble
            print("| loading model(s) from {}".format(args.ensemble_paths))
            models, _model_args = checkpoint_utils.load_model_ensemble(
                args.ensemble_paths.split(","),
                task=SpeechRecognitionTask.setup_task(args, **kwargs),
            )
            use_cuda = torch.cuda.is_available() and not args.cpu
            # Optimize ensemble for generation (includes setting .eval())
            for model in models:
                model.make_generation_fast_(need_attn=False)
                if args.fp16_ensemble:
                    model.half()
                if use_cuda:
                    model.cuda()
        else:
            models = []

        return cls(args, tgt_dict, models)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        sample = self.compute_ensemble_logits(sample)
        return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def valid_step(self, sample, model, criterion):
        sample = self.compute_ensemble_logits(sample)
        return super().valid_step(sample, model, criterion)

    def build_generator(self, args):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_generator import (
                SequenceGenerator,
                SequenceGeneratorWithAlignment,
            )

            if getattr(args, "print_alignment", False):
                return SequenceGeneratorWithAlignment(
                    self.target_dictionary,
                    beam_size=getattr(args, "beam", 5),
                    max_len_a=getattr(args, "max_len_a", 0),
                    max_len_b=getattr(args, "max_len_b", 200),
                    min_len=getattr(args, "min_len", 1),
                    normalize_scores=(not getattr(args, "unnormalized", False)),
                    len_penalty=getattr(args, "lenpen", 1),
                    unk_penalty=getattr(args, "unkpen", 0),
                    sampling=getattr(args, "sampling", False),
                    sampling_topk=getattr(args, "sampling_topk", -1),
                    sampling_topp=getattr(args, "sampling_topp", -1.0),
                    temperature=getattr(args, "temperature", 1.0),
                    diverse_beam_groups=getattr(args, "diverse_beam_groups", -1),
                    diverse_beam_strength=getattr(args, "diverse_beam_strength", 0.5),
                    match_source_len=getattr(args, "match_source_len", False),
                    no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                )
            else:
                return SequenceGenerator(
                    self.target_dictionary,
                    beam_size=getattr(args, "beam", 5),
                    max_len_a=getattr(args, "max_len_a", 0),
                    max_len_b=getattr(args, "max_len_b", 200),
                    min_len=getattr(args, "min_len", 1),
                    normalize_scores=(not getattr(args, "unnormalized", False)),
                    len_penalty=getattr(args, "lenpen", 1),
                    unk_penalty=getattr(args, "unkpen", 0),
                    sampling=getattr(args, "sampling", False),
                    sampling_topk=getattr(args, "sampling_topk", -1),
                    sampling_topp=getattr(args, "sampling_topp", -1.0),
                    temperature=getattr(args, "temperature", 1.0),
                    diverse_beam_groups=getattr(args, "diverse_beam_groups", -1),
                    diverse_beam_strength=getattr(args, "diverse_beam_strength", 0.5),
                    match_source_len=getattr(args, "match_source_len", False),
                    no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                )

    @torch.no_grad()
    def inference_step(self, generator, models, sample, prefix_tokens=None):
        hypos_sample = generator.generate(models, sample, prefix_tokens=prefix_tokens)

        if self.compute_uncertainty:
            # compute uncertainties
            self.add_uncertainties(sample, hypos_sample, models)

        return hypos_sample

    def add_uncertainties(self, sample, hypos, models):
        if len(models) != 1:
            raise NotImplementedError(
                "Uncertainty estimation for ensembles of distilled models is not implemented"
            )
        model = models[0]

        tokens = collate_tokens(
            [out["tokens"] for sent in hypos for out in sent[: self.args.nbest]],
            eos_idx=self.tgt_dict.eos(),
            pad_idx=self.tgt_dict.pad(),
        )
        prev_output = collate_tokens(
            [out["tokens"] for sent in hypos for out in sent[: self.args.nbest]],
            eos_idx=self.tgt_dict.eos(),
            pad_idx=self.tgt_dict.pad(),
            move_eos_to_beginning=True,
        )

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        prev_tokens = sample["net_input"]["prev_output_tokens"]

        sample["net_input"]["src_tokens"] = torch.repeat_interleave(
            sample["net_input"]["src_tokens"], self.args.nbest, dim=0
        )
        sample["net_input"]["src_lengths"] = torch.repeat_interleave(
            sample["net_input"]["src_lengths"], self.args.nbest, dim=0
        )
        sample["net_input"]["prev_output_tokens"] = prev_output

        net_output = model(**sample["net_input"])

        sample["net_input"]["src_tokens"] = src_tokens
        sample["net_input"]["src_lengths"] = src_lengths
        sample["net_input"]["prev_output_tokens"] = prev_tokens

        dirichlet_params, concentrations = get_dirichlet_parameters(
            net_output,
            add_to_alphas=self.model_offset,
            dtype=torch.double,
        )
        # assert torch.isfinite(dirichlet_params).all()
        # assert torch.isfinite(concentrations).all()
        concentrations = concentrations.unsqueeze(2)

        normalized_probs = model.get_normalized_probs(net_output, log_probs=False)
        normalized_logprobs = normalized_probs.log()

        mask = tokens.eq(self.tgt_dict.pad())
        num_of_tokens = torch.sum(~mask, dim=1)

        (
            entropy_of_expected,
            expected_entropy,
            mutual_information,
            epkl,
            mkl,
        ) = compute_token_dirichlet_uncertainties(
            dirichlet_params, concentrations, normalized_probs
        )

        if mask.any():
            entropy_of_expected.masked_fill_(mask, 0)
            expected_entropy.masked_fill_(mask, 0)
            mutual_information.masked_fill_(mask, 0)
            epkl.masked_fill_(mask, 0)
            mkl.masked_fill_(mask, 0)

        (
            log_probs,
            scores,
            scores_mkl,
            token_log_probs,
            token_scores_mkl,
        ) = compute_sequence_dirichlet_uncertainties(
            dirichlet_params,
            concentrations,
            normalized_logprobs,
            tokens,
            mask,
            num_of_tokens,
        )

        for i, sent in enumerate(hypos):
            for j, hypo in enumerate(sent[: self.args.nbest]):
                ind = i * self.args.nbest + j

                zeros_tensor = torch.zeros_like(mkl[ind])
                pad_mask = ~mask[ind]

                hypo["token_uncertainties"] = {
                    "entropy_of_expected": entropy_of_expected[ind][pad_mask],
                    "expected_entropy": expected_entropy[ind][pad_mask],
                    "mutual_information": mutual_information[ind][pad_mask],
                    "EPKL": epkl[ind][pad_mask],
                    "MKL": mkl[ind][pad_mask],
                    "ep_entropy_of_expected": zeros_tensor[pad_mask],
                    "ep_mutual_information": zeros_tensor[pad_mask],
                    "ep_EPKL": zeros_tensor[pad_mask],
                    "ep_MKL": zeros_tensor[pad_mask],
                    "token_DU": zeros_tensor[pad_mask],
                    "token_ep_TU": zeros_tensor[pad_mask],
                    "token_pe_TU": -token_log_probs[ind][pad_mask],
                    "token_ep_MKL": zeros_tensor[pad_mask],
                    "token_pe_MKL": token_scores_mkl[ind][pad_mask],
                }

                zero_tensor = zeros_tensor.sum()

                hypo["sequence_uncertainties"] = {
                    "log-prob": log_probs[ind],
                    "pe_entropy_of_expected": entropy_of_expected[ind].sum()
                                              / num_of_tokens[ind],
                    "expected_entropy": expected_entropy[ind].sum()
                                        / num_of_tokens[ind],
                    "pe_mutual_information": mutual_information[ind].sum()
                                             / num_of_tokens[ind],
                    "pe_EPKL": epkl[ind].sum() / num_of_tokens[ind],
                    "pe_MKL": mkl[ind].sum() / num_of_tokens[ind],
                    "pe_sTU": scores[ind],
                    "pe_sMKL": scores_mkl[ind],
                    "ep_sTU": zero_tensor,
                    "sDU": zero_tensor,
                    "ep_sMKL": zero_tensor,
                    "ep_entropy_of_expected": zero_tensor,
                    "ep_mutual_information": zero_tensor,
                    "ep_EPKL": zero_tensor,
                    "ep_MKL": zero_tensor,
                    "var": zero_tensor,
                    "combo": zero_tensor,
                    "logvar": zero_tensor,
                    "logcombo": zero_tensor,
                }

    @torch.no_grad()
    def compute_ensemble_logits(self, sample):
        batch_size, num_tokens = sample["target"].size()
        ens_size, vocab_size = len(self.ensemble), len(self.tgt_dict)
        dtype = torch.half if self.args.fp16 else torch.float
        sample["ensemble_logits"] = torch.empty(
            (batch_size, num_tokens, ens_size, vocab_size),
            dtype=dtype,
            device="cpu" if self.args.cpu else "cuda",
        )

        for i, model in enumerate(self.ensemble):
            sample["ensemble_logits"][:, :, i] = model(**sample["net_input"])[0].type(
                dtype
            )
        return sample
