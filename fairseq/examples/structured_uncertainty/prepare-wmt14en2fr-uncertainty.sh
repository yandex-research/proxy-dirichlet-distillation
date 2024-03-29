#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
  "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
  "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
  "http://statmt.org/wmt13/training-parallel-un.tgz"
  "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
  "http://statmt.org/wmt10/training-giga-fren.tar"
  "http://statmt.org/wmt14/test-full.tgz"
  "http://www.statmt.org/wmt14/medical-task/khresmoi-summary-test-set.tgz"
  "http://www.statmt.org/wmt14/medical-task/khresmoi-query-test-set.tgz"
)
FILES=(
  "training-parallel-europarl-v7.tgz"
  "training-parallel-commoncrawl.tgz"
  "training-parallel-un.tgz"
  "training-parallel-nc-v9.tgz"
  "training-giga-fren.tar"
  "test-full.tgz"
  "khresmoi-summary-test-set.tgz"
  "khresmoi-query-test-set.tgz"
)
CORPORA=(
  "training/europarl-v7.fr-en"
  "commoncrawl.fr-en"
  "un/undoc.2000.fr-en"
  "training/news-commentary-v9.fr-en"
  "giga-fren.release2.fixed"
)

if [ ! -d "$SCRIPTS" ]; then
  echo "Please set SCRIPTS variable correctly to point to Moses scripts."
  exit
fi

src=en
tgt=fr
lang=en-fr
prep=wmt14_en_fr
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

cd $orig

for ((i = 0; i < ${#URLS[@]}; ++i)); do
  file=${FILES[i]}
  if [ -f $file ]; then
    echo "$file already exists, skipping download"
  else
    url=${URLS[i]}
    wget "$url"
    if [ -f $file ]; then
      echo "$url successfully downloaded."
    else
      echo "$url not successfully downloaded."
      exit -1
    fi
    if [ ${file: -4} == ".tgz" ]; then
      tar zxvf $file
    elif [ ${file: -4} == ".tar" ]; then
      tar xvf $file
    fi
  fi
done

gunzip giga-fren.release2.fixed.*.gz
cd ..

echo "pre-processing test data..."
for l in $src $tgt; do
  if [ "$l" == "$src" ]; then
    t="src"
  else
    t="ref"
  fi
  grep '<seg id' $orig/test-full/newstest2014-fren-$t.$l.sgm |
    sed -e 's/<seg id="[0-9]*">\s*//g' |
    sed -e 's/\s*<\/seg>\s*//g' |
    sed -e "s/\’/\'/g" |
    perl $TOKENIZER -threads 24 -a -l $l >$tmp/test.$l
  head -n 1500 $tmp/test.$l >$tmp/test-h1.$l
  tail -n +1501 $tmp/test.$l >$tmp/test-h2.$l
  echo ""
  grep '<seg id' $orig/khresmoi-summary-test-set/khresmoi-summary-dev.${l}.sgm |
    sed -e 's/<seg id="[0-9]*">\s*//g' |
    sed -e 's/\s*<\/seg>\s*//g' |
    sed -e "s/\’/\'/g" |
    perl $TOKENIZER -threads 24 -a -l $l >$tmp/bio-ks-dev.$l
  echo ""
  grep '<seg id' $orig/khresmoi-summary-test-set/khresmoi-summary-test.${l}.sgm |
    sed -e 's/<seg id="[0-9]*">\s*//g' |
    sed -e 's/\s*<\/seg>\s*//g' |
    sed -e "s/\’/\'/g" |
    perl $TOKENIZER -threads 24 -a -l $l >$tmp/bio-ks-test.$l
  echo ""
  cat $tmp/bio-ks-dev.$l $tmp/bio-ks-test.$l >$tmp/bio-ks.$l
  echo ""
  cat $tmp/bio-ks-dev.$l $tmp/bio-ks-test.$l >$tmp/bio-ks.$l
  cat $orig/librispeech/test-clean.txt |
    sed -e 's/[0-9]*\-[0-9]*\-[0-9]* //g' |
    sed 's/.*/\L&/' |
    sed -e 's/^\(.\)/\U\1/g' |
    sed -e "s/$/\./" |
    sed -e "s/\’/\'/g" |
    perl $TOKENIZER -threads 24 -a -l $l >$tmp/librispeech-tc.$l
  echo ""
  cat $orig/librispeech/test-other.txt |
    sed -e 's/[0-9]*\-[0-9]*\-[0-9]* //g' |
    sed 's/.*/\L&/' |
    sed -e 's/^\(.\)/\U\1/g' |
    sed -e "s/$/\./" |
    sed -e "s/\’/\'/g" |
    perl $TOKENIZER -threads 24 -a -l $l >$tmp/librispeech-tp.$l
  echo ""
done
echo ""
grep '<seg id' $orig/test-full/newstest2014-deen-ref.de.sgm |
  sed -e 's/<seg id="[0-9]*">\s*//g' |
  sed -e 's/\s*<\/seg>\s*//g' |
  sed -e "s/\’/\'/g" |
  perl $TOKENIZER -threads 24 -a -l de >$tmp/test.de
echo ""

TRAIN=$tmp/train.fr-en
BPE_CODE=$prep/code

for L in $src $tgt; do
  for f in test.$L test-h1.$L test-h2.$L bio-ks-dev.$L bio-ks-test.$L bio-ks.$L librispeech-tc.$L librispeech-tp.$L; do
    echo "apply_bpe.py to ${f}..."
    python $BPEROOT/apply_bpe.py -c $BPE_CODE <$tmp/$f >$tmp/bpe.$f
  done
done
python $BPEROOT/apply_bpe.py -c $BPE_CODE <$tmp/test.de >$tmp/bpe.test.de

for L in $src $tgt; do
  cp $tmp/bpe.test.$L $prep/test.$L
  cp $tmp/bpe.test-h1.$L $prep/test-h1.$L
  cp $tmp/bpe.test-h2.$L $prep/test-h2.$L
  cp $tmp/bpe.bio-ks-dev.$L $prep/bio-ks-dev.$L
  cp $tmp/bpe.bio-ks-test.$L $prep/bio-ks-test.$L
  cp $tmp/bpe.bio-ks.$L $prep/bio-ks.$L
  cp $tmp/bpe.librispeech-tc.$L $prep/librispeech-tc.$L
  cp $tmp/bpe.librispeech-tp.$L $prep/librispeech-tp.$L
  cat $prep/test.$L | python permute_sentence.py >$prep/test-perm.$L
done
cp $tmp/bpe.test.de $prep/test.de

cd $prep
#Make language-switched forms of the data
cp test.fr test-fren.en
cp test.en test-fren.fr
cp test.en test-enen.fr
cp test.en test-enen.en
cp test.fr test-frfr.en
cp test.fr test-frfr.fr
cp test.fr test-enfr.en
cp test.en test-enfr.fr

#Make BPE-permuted forms of the data.
cp test-perm.fr test-perm-enfr.fr
cp test-perm.en test-perm-enfr.en
cp test.en test-perm-fr.en
cp test-perm.fr test-perm-fr.fr
cp test-perm.en test-perm-en.en
cp test.fr test-perm-en.fr
cd ../
