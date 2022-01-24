#!/bin/bash
set -euxo pipefail

CODE_DIR="${SOURCE_CODE_PATH:-${HOME}}"
OUTPUT_DIR="${TMP_OUTPUT_PATH:-${HOME}}"

(
  cd $OUTPUT_DIR/decode
  for i in test-clean test-other ami cv-fr; do
    rm -rf ${i}
    mkdir ${i}
    grep "^SU-*" results-${i}.txt >${i}/tmp
    grep "^H-[0-9]*\s" results-${i}.txt | sed -e "s/^H\-[0-9]*.*-[0-9]*\.[0-9]*//" >${i}/hypos.txt
    grep "^T-[0-9]*\s" results-${i}.txt | sed -e "s/^T\-[0-9]*//" >${i}/refs.txt
    awk '{print $2}' ${i}/tmp >${i}/entropy_expected.txt
    awk '{print $3}' ${i}/tmp >${i}/expected_entropy.txt
    awk '{print $4}' ${i}/tmp >${i}/mutual_information.txt
    awk '{print $5}' ${i}/tmp >${i}/epkl.txt
    awk '{print $6}' ${i}/tmp >${i}/score.txt
    awk '{print $7}' ${i}/tmp >${i}/aep_tu.txt
    awk '{print $8}' ${i}/tmp >${i}/aep_du.txt
    awk '{print $9}' ${i}/tmp >${i}/npmi.txt
    awk '{print $10}' ${i}/tmp >${i}/score_npmi.txt
    awk '{print $11}' ${i}/tmp >${i}/log_probs.txt
    awk '{print $12}' ${i}/tmp >${i}/ep_entropy_expected.txt
    awk '{print $13}' ${i}/tmp >${i}/ep_mutual_information.txt
    awk '{print $14}' ${i}/tmp >${i}/ep_epkl.txt
    awk '{print $15}' ${i}/tmp >${i}/ep_mkl.txt
    awk '{print $16}' ${i}/tmp >${i}/mkl.txt
    awk '{print $17}' ${i}/tmp >${i}/var.txt
    awk '{print $18}' ${i}/tmp >${i}/varcombo.txt
    awk '{print $19}' ${i}/tmp >${i}/logvar.txt
    awk '{print $20}' ${i}/tmp >${i}/logcombo.txt

    egrep "T\-[0-9]+" results-${i}.txt | sed -E "s/^T-[0-9]+\s+//" | sed -E "s/\s//g" | sed -E "s/▁/ /g" >${i}/refs.txt
    egrep "H\-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^H-[0-9]+\s+-*[0-9]*\.[0-9]+(e-[0-9]+)*\s+//" | sed -E "s/\s//g" | sed -E "s/▁/ /g" >${i}/hypos.txt
    egrep "T\-[0-9]+" results-${i}.txt | sed -E "s/^T-[0-9]+\s+//" >${i}/word_refs.txt
    egrep "H\-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^H-[0-9]+\s+-*[0-9]*\.[0-9]+(e-[0-9]+)*\s+//" >${i}/word_hypos.txt

    len=$(wc -l ${i}/word_hypos.txt | awk '{print $1}')
    for k in $(seq 1 ${len}); do echo "(${k}_1)" >>${i}/ids.txt; done
    mv ${i}/word_refs.txt ${i}/tmp
    paste -d " " ${i}/tmp ${i}/ids.txt >${i}/word_refs.txt
    mv ${i}/word_hypos.txt ${i}/tmp
    paste -d " " ${i}/tmp ${i}/ids.txt >${i}/word_hypos.txt
    mv ${i}/refs.txt ${i}/tmp
    paste -d " " ${i}/tmp ${i}/ids.txt >${i}/refs.txt
    mv ${i}/hypos.txt ${i}/tmp
    paste -d " " ${i}/tmp ${i}/ids.txt >${i}/hypos.txt

    egrep "P\-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^P-[0-9]+\s+//" | sed -E "s/\s+-*[0-9]+\.[0-9]+$//" >${i}/word_scores.txt
    egrep "T\-EOE-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-EOE-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_pe_eoe.txt
    egrep "T\-EXE-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-EXE-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_exe.txt
    egrep "T\-MI-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-MI-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_pe_mi.txt
    egrep "T\-EPKL-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-EPKL-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_pe_epkl.txt
    egrep "T\-MKL-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-MKL-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_pe_mkl.txt
    egrep "T\-EOE-EP-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-EOE-EP-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_ep_eoe.txt
    egrep "T\-MI-EP-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-MI-EP-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_ep_mi.txt
    egrep "T\-EPKL-EP-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-EPKL-EP-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_ep_epkl.txt
    egrep "T\-MKL-EP-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-MKL-EP-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_ep_mkl.txt
    egrep "T\-EP-TU-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-EP-TU-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_ep_sTU.txt
    egrep "T\-PE-TU-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-PE-TU-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_pe_sTU.txt
    egrep "T\-DU-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-DU-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_sDU.txt
    egrep "T\-EP-MKL-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-EP-MKL-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_ep_sMKL.txt
    egrep "T\-PE-MKL-[0-9]+" results-${i}.txt | awk 'NR % 20 == 1' | sed -E "s/^T-PE-MKL-[0-9]+\s+//" | sed -E "s/\s+0\.0000$//" >${i}/word_pe_sMKL.txt
    cd ${i}

    $CODE_DIR/sclite -r word_refs.txt -h word_hypos.txt -i rm -o all dtl sgml
    python3 $CODE_DIR/examples/structured_uncertainty/assessment/sgml-map.py -i word_hypos.txt.sgml -o error_labels.txt
    cd ..
  done

  rm -rf to_ami
  mkdir to_ami
  for n in 1 20; do
    for i in test-other ami cv-fr; do
      python3 $CODE_DIR/examples/structured_uncertainty/assessment/ood_detection.py test-clean ${i} ${i} --beam_width 20 --nbest ${n}
      python3 $CODE_DIR/examples/structured_uncertainty/assessment/ood_detection.py test-clean ${i} ${i} --beam_width 20 --nbest ${n} --beam_search
    done

    python3 $CODE_DIR/examples/structured_uncertainty/assessment/ood_detection.py test-other ami to_ami --beam_width 20 --nbest ${n}
    python3 $CODE_DIR/examples/structured_uncertainty/assessment/ood_detection.py test-other ami to_ami --beam_width 20 --nbest ${n} --beam_search

    for i in test-clean test-other ami; do
      python3 $CODE_DIR/examples/structured_uncertainty/assessment/seq_error_detection.py ${i} --nbest ${n} --beam_width 20 --wer
      python3 $CODE_DIR/examples/structured_uncertainty/assessment/seq_error_detection.py ${i} --nbest ${n} --beam_width 20 --wer --beam_search
    done
  done

  for i in test-clean test-other ami; do
    python3 $CODE_DIR/examples/structured_uncertainty/assessment/token_error_detection.py ${i}
  done

  cd $OUTPUT_DIR/reference
  for i in test-clean test-other ami cv-fr; do
    rm -rf ${i}
    mkdir ${i}
    grep "^SU-*" results-${i}.txt >${i}/tmp
    grep "^H-[0-9]*\s" results-${i}.txt | sed -e "s/^H\-[0-9]*.*-[0-9]*\.[0-9]*//" >${i}/hypos.txt
    grep "^T-[0-9]*\s" results-${i}.txt | sed -e "s/^T\-[0-9]*//" >${i}/refs.txt
    awk '{print $2}' ${i}/tmp >${i}/entropy_expected.txt
    awk '{print $3}' ${i}/tmp >${i}/expected_entropy.txt
    awk '{print $4}' ${i}/tmp >${i}/mutual_information.txt
    awk '{print $5}' ${i}/tmp >${i}/epkl.txt
    awk '{print $6}' ${i}/tmp >${i}/score.txt
    awk '{print $7}' ${i}/tmp >${i}/aep_tu.txt
    awk '{print $8}' ${i}/tmp >${i}/aep_du.txt
    awk '{print $9}' ${i}/tmp >${i}/npmi.txt
    awk '{print $10}' ${i}/tmp >${i}/score_npmi.txt
    awk '{print $11}' ${i}/tmp >${i}/log_probs.txt
    awk '{print $12}' ${i}/tmp >${i}/ep_entropy_expected.txt
    awk '{print $13}' ${i}/tmp >${i}/ep_mutual_information.txt
    awk '{print $14}' ${i}/tmp >${i}/ep_epkl.txt
    awk '{print $15}' ${i}/tmp >${i}/ep_mkl.txt
    awk '{print $16}' ${i}/tmp >${i}/mkl.txt
    awk '{print $17}' ${i}/tmp >${i}/var.txt
    awk '{print $18}' ${i}/tmp >${i}/varcombo.txt
    awk '{print $19}' ${i}/tmp >${i}/logvar.txt
    awk '{print $20}' ${i}/tmp >${i}/logcombo.txt
  done

  rm -rf to_ami
  mkdir to_ami
  for i in test-other ami cv-fr; do
    python3 $CODE_DIR/examples/structured_uncertainty/assessment/ood_detection.py test-clean ${i} ${i} --beam_width 1 --nbest 1
  done

  python3 $CODE_DIR/examples/structured_uncertainty/assessment/ood_detection.py test-other ami to_ami --beam_width 1 --nbest 1

  cd $OUTPUT_DIR

  grep "Sum/Avg" decode/test-clean/hypos.txt.sys | awk '{print $10}' >decode/wer-ltc
  grep "Sum/Avg" decode/test-other/hypos.txt.sys | awk '{print $10}' >decode/wer-lto
  grep "Sum/Avg" decode/ami/hypos.txt.sys | awk '{print $9}' >decode/wer-ami
)
