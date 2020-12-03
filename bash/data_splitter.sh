#!/usr/bin/env bash
#./data_splitter.sh ../data/reviews/processed_csv/res.csv 1 0.5 0.25

FileName=$1
FirstLine=$2
TrainRatio=$3
DevRatio=$4

DirName=$(dirname "${FileName}")
BaseName=$(basename "${FileName}")
count=$(wc -l "${FileName}" | cut -f1 -d' ')

tmpFile="${FileName}".tmp
if [[ "${FirstLine}" == "0" ]]
then
  echo "${count} lines, NO HEADER"
#  echo "${TrainLines}, ${DevLines}"
  echo "read and shuffle file ${FileName} of ${count} lines"
  tail -n+1 "${FileName}" | awk -F '\t' '{print $1, $NF}' | shuf > "${tmpFile}"

else
  count=$(("${count}" - 1))
  echo "${count} lines, WITH HEADER"
#  echo "${TrainLines}, ${DevLines}"
  echo "read and shuffle file ${FileName} of ${count} lines"
  tail -n+2 "${FileName}" | awk -F '\t' '{print $1, $NF}' | shuf > "${tmpFile}"
fi

TrainLines=$(echo "scale=2;${count}*${TrainRatio}" | bc | awk '{print ($0-int($0)<0.499)?int($0):int($0)+1}')
DevLines=$(echo "scale=2;${count}*${DevRatio}" | bc | awk '{print ($0-int($0)<0.499)?int($0):int($0)+1}')
TestLines=$(echo "scale=2;${count}-${TrainLines}-${DevLines}" | bc | awk '{print ($0-int($0)<0.499)?int($0):int($0)+1}')

TrainFile="${DirName}"/train_"${BaseName}"
if [[ -f "${TrainFile}" ]]; then
    echo "File ${TrainFile} exist; Removing..."
    rm "${TrainFile}"
fi
echo "write train file of ${TrainLines} lines"
head -n "${TrainLines}" "${tmpFile}" > "${TrainFile}"

DevFile="${DirName}"/dev_"${BaseName}"
if [[ -f "${DevFile}" ]]; then
    echo "File ${DevFile} exist; Removing..."
    rm "${DevFile}"
fi
echo "write dev file of ${DevLines} lines"
tail -n "+$((${TrainLines}+1))" "${tmpFile}" | head -n "$((${DevLines}+1))" > "${DevFile}"

TestFile="${DirName}"/test_"${BaseName}"
if [[ -f "${TestFile}" ]]; then
    echo "File ${TestFile} exist; Removing..."
    rm "${TestFile}"
fi
echo "Write test file of ${TestLines} lines"
tail -n "${TestLines}" "${tmpFile}" > "${TestFile}"
rm "${tmpFile}"
