#!/usr/bin/env bash
DirName=$1
FileName=$2

WriteFile="${DirName}"/"${FileName}"
if [[ -f "${WriteFile}" ]]; then
    echo "File ${WriteFile} exist; Removing..."
    rm "${WriteFile}"
fi
head -n -1 "${DirName}"/star1/"${FileName}" > "${WriteFile}"
echo "," >> "${WriteFile}"

tail -n +2 "${DirName}"/star2/"${FileName}" | head -n -1 >> "${WriteFile}"
echo "," >> "${WriteFile}"

tail -n +2 "${DirName}"/star3/"${FileName}" | head -n -1 >> "${WriteFile}"
echo "," >> "${WriteFile}"

tail -n +2 "${DirName}"/star4/"${FileName}" | head -n -1 >> "${WriteFile}"
echo "," >> "${WriteFile}"

tail -n +2 "${DirName}"/star5/"${FileName}" | head -n -0 >> "${WriteFile}"
