#!/usr/bin/env bash
# ./data_loader.sh locate toy research
DB_ID=$1
GCP_EXTENSION=$2
run_mode=$3

sudo chmod -R 777 ./data
per_dir=./data/permanent

if [[ -d "$per_dir" ]]
then
    echo "Directory $per_dir exists."
else
    echo "Create $per_dir directory"
    sudo mkdir ${per_dir}
fi
sudo chmod -R 777 ${per_dir}

# Download pre-trained emb if it does not exist
embfile=${per_dir}/glove.42B.300d.txt
if [[ -f "$embfile" ]]; then
    echo "File $embfile exist"
else
    echo "Download $embfile file"
    sudo wget -P ${per_dir} http://nlp.stanford.edu/data/glove.42B.300d.zip
    sudo unzip ${per_dir}/glove.42B.300d.zip -d ${per_dir}
    sudo rm ${per_dir}/glove.42B.300d.zip
fi

# Download conceptNet if it does not exist
KBisa=${per_dir}/english_IsA.pkl
if [[ -f "$KBisa" ]]
then
    echo "File $KBisa exists."
else
    echo "Download $KBisa file"
    gsutil -m cp gs://chata-caas-npx/${DB_ID}/1/model/neural_parser/conceptNet/english_IsA.pkl ${KBisa}
fi

KBrel=$per_dir/english_RelatedTo.pkl
if [[ -f "$KBrel" ]]
then
    echo "File $KBrel exists."
else
    echo "Download $KBrel file"
    gsutil -m cp gs://chata-caas-npx/${DB_ID}/1/model/neural_parser/conceptNet/english_RelatedTo.pkl ${KBrel}
fi


vol_dir=./data/data_${DB_ID}_${GCP_EXTENSION}
if [[ -d "$vol_dir" ]]
then
    echo "Directory $vol_dir exists."
else
    echo "Create $vol_dir directory"
    sudo mkdir ${vol_dir}
fi
sudo chmod -R 777 ${vol_dir}

# Download schema file if it does not exist
TBDIR=${vol_dir}/schema
if [[ -d "$TBDIR" ]]
then
    echo "Directory $TBDIR exists."
else
    echo "Create $TBDIR directory"
    sudo mkdir ${TBDIR}
fi
sudo chmod -R 777 ${TBDIR}

TBFILE=${TBDIR}/json_tables_full.json
if [[ -f "$TBFILE" ]]
then
    echo "File $TBFILE exists."
else
    echo "Download $TBFILE file"
    gsutil -m cp gs://chata-caas-npx/${DB_ID}/1/model/neural_parser/data/json_tables_full.json ${TBFILE}
fi

# Download csv corpus file if it does not exist
CSVDIR=${vol_dir}/csv
if [[ -d "$CSVDIR" ]]
then
    echo "Directory $CSVDIR exists."
else
    echo "Create $CSVDIR directory"
    sudo mkdir ${CSVDIR}
fi
sudo chmod -R 777 ${CSVDIR}
#gsutil cp -r gs://chata-caas-npx/${DB_ID}/1/model/neural_parser/datasets/dataset_${GCP_EXTENSION}/* ${CSVDIR}

trainFILE=${CSVDIR}/generated_corpus_stored_train_human.csv
if [[ -f "$trainFILE" ]]
then
    echo "File $trainFILE exists."
else
    echo "Download $trainFILE file"
    gsutil -m cp gs://chata-caas-npx/${DB_ID}/1/model/neural_parser/datasets/dataset_${GCP_EXTENSION}/generated_corpus_stored_train_human.csv ${trainFILE}
fi

if [[ "$run_mode" == "research" ]]
then
  echo "Split data"
  python create_dataset.py --corpus_file ${trainFILE} --train_file ${CSVDIR}/train.csv --val_file ${CSVDIR}/dev.csv --test_file ${CSVDIR}/test.csv --firstline --tr_ratio 0.7 --val_ratio 0.85

else
  echo "Keep data"
fi


#devFILE=${CSVDIR}/generated_corpus_stored_dev_human.csv
#if [[ -f "$devFILE" ]]
#then
#    echo "File $devFILE exists."
#else
#    echo "Download $devFILE file"
#    gsutil cp -r gs://chata-caas-npx/${DB_ID}/1/model/neural_parser/datasets/dataset_${GCP_EXTENSION}/generated_corpus_stored_dev_human.csv ${devFILE}
#fi
#
#testFILE=${CSVDIR}/generated_corpus_stored_test_human.csv
#if [[ -f "$testFILE" ]]
#then
#    echo "File $testFILE exists."
#else
#    echo "Download $testFILE file"
#    gsutil cp -r gs://chata-caas-npx/${DB_ID}/1/model/neural_parser/datasets/dataset_${GCP_EXTENSION}/generated_corpus_stored_test_human.csv ${testFILE}
#fi

