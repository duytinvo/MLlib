#!/usr/bin/env bash
# ./standard_corpus.sh purefacts ./data/standard_corpus/purefacts/full/purefacts_1_model_neural_parser_training_team_corpus_20191130_purefacts_full_corpus_20191130.csv
DB_ID=$1
full_file=$2
parentdir="$(dirname "${full_file}")"
savedir="$(dirname "${parentdir}")"
rm ${savedir}/${DB_ID}*.csv

python create_dataset.py --corpus_file ${full_file} --train_file ${parentdir}/${DB_ID}_corpus.csv --val_file ${savedir}/${DB_ID}_final_test2.csv --test_file ${savedir}/${DB_ID}_empty.csv --firstline --tr_ratio 0.75 --val_ratio 1.0

python create_dataset.py --corpus_file ${parentdir}/${DB_ID}_corpus.csv --train_file ${savedir}/${DB_ID}_1_train.csv --val_file ${savedir}/${DB_ID}_1_dev.csv --test_file ${savedir}/${DB_ID}_1_test.csv --firstline --tr_ratio 0.9 --val_ratio 0.95

python create_dataset.py --corpus_file ${savedir}/${DB_ID}_1_train.csv --train_file ${savedir}/${DB_ID}_2_train.csv --val_file ${savedir}/${DB_ID}_2_dev.csv --test_file ${savedir}/${DB_ID}_2_test.csv --firstline --tr_ratio 0.8 --val_ratio 0.9
tail -n +2 ${savedir}/${DB_ID}_1_dev.csv >> ${savedir}/${DB_ID}_2_dev.csv
tail -n +2 ${savedir}/${DB_ID}_1_test.csv >> ${savedir}/${DB_ID}_2_test.csv

python create_dataset.py --corpus_file ${savedir}/${DB_ID}_2_train.csv --train_file ${savedir}/${DB_ID}_3_train.csv --val_file ${savedir}/${DB_ID}_3_dev.csv --test_file ${savedir}/${DB_ID}_3_test.csv --firstline --tr_ratio 0.7 --val_ratio 0.85
tail -n +2 ${savedir}/${DB_ID}_2_dev.csv >> ${savedir}/${DB_ID}_3_dev.csv
tail -n +2 ${savedir}/${DB_ID}_2_test.csv >> ${savedir}/${DB_ID}_3_test.csv