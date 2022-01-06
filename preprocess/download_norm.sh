#!/bin/bash
# Copyright 2018-present, Data Mining and Information Systems (DMIS)
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

cd ..

if [ ! -d "normalization" ]; then
    mkdir normalization
fi

cd normalization

if [ -f "data.zip" ]; then
    echo "Found data.zip."
else
    echo "Not found data.zip. Downloading.."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NqgG3zJzopG2IqG-0g1o6fH0xVpO4PPN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NqgG3zJzopG2IqG-0g1o6fH0xVpO4PPN" -O data.zip && rm -rf /tmp/cookies.txt
fi

# Get dictionary data
unzip "data.zip"
rm "data.zip"

if [ -f "resources.zip" ]; then
    echo "Found resources.zip."
else
    echo "Not found resources.zip. Downloading.."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uU1U6UORqr3l_YYQ5TXeazpLrpeg_OcP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uU1U6UORqr3l_YYQ5TXeazpLrpeg_OcP" -O resources.zip && rm -rf /tmp/cookies.txt
fi

# Get Normalization resources
unzip "resources.zip"
rm "resources.zip"

chmod 764 resources/normalizers/gene/Ab3P
chmod 764 resources/normalizers/gene/CRF/crf_test


if [ -d "jsons" ]; then
    echo "Found jsons/"
else
    mkdir "jsons/"
fi

if [ -d "jsons_normalized" ]; then
    echo "Found jsons_normalized/"
else
    mkdir "jsons_normalized/"
fi


cd ..
if [ ! -d "logs" ]; then
    mkdir logs
fi

echo "BERN normalizer settings done!"
