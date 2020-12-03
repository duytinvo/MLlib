#!/usr/bin/env bash

#git clone https://github.com/facebookresearch/fastText.git
#cd fastText
#mkdir build && cd build && cmake ..
#make && make install

wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
unzip v0.1.0.zip
cd fastText-0.1.0
make

cd ..

if [ ! -d build/ ]; then
    mkdir build
fi

if [ ! -d build/fasttext/ ]; then
    mkdir build/fasttext
fi

cp fastText-0.1.0/fasttext build/fasttext/
rm -rf fastText-0.1.0 v0.1.0.zip