#!/usr/bin/env bash
git clone http://github.com/stanfordnlp/glove
cd glove && make

cd ..

if [ ! -d build/ ]; then
    mkdir build
fi

if [ ! -d build/glove/ ]; then
    mkdir build/glove
fi

cp -rf glove/build/* build/glove

rm -rf glove/
