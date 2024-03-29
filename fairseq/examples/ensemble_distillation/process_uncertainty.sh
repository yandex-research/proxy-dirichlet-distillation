#!/bin/bash
set -euxo pipefail

CODE_DIR="${SOURCE_CODE_PATH:-${HOME}}"
OUTPUT_DIR="${TMP_OUTPUT_PATH:-${HOME}}"

(
  cd $OUTPUT_DIR/decode
  for i in test test5 test6 test9 test12 test14 test_ens_pred; do
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

  for i in test5 test6 test9 test12 test14; do
    for n in 1 5; do
      python3 $CODE_DIR/examples/structured_uncertainty/assessment/ood_detection.py test ${i} ${i} --nbest ${n} --beam_width 5
      python3 $CODE_DIR/examples/structured_uncertainty/assessment/ood_detection.py test ${i} ${i} --nbest ${n} --beam_width 5 --beam_search
    done
  done

  for n in 1 5; do
    for i in test test5; do
      python3 $CODE_DIR/examples/structured_uncertainty/assessment/seq_error_detection.py ./${i} --beam_width 5 --nbest ${n}
      python3 $CODE_DIR/examples/structured_uncertainty/assessment/seq_error_detection.py ./${i} --beam_width 5 --nbest ${n} --beam_search
    done
  done
)
(
  cd $OUTPUT_DIR/reference
  for i in test test5 test6 test9 test12 test14 test_ens_pred; do
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

  for i in test5 test6 test9 test12 test14; do
    python3 $CODE_DIR/examples/structured_uncertainty/assessment/ood_detection.py test ${i} ${i} --nbest 1 --beam_width 1
  done
)
