# BERN2

We present **BERN2** (Advanced **B**iomedical **E**ntity **R**ecognition and **N**ormalization), a tool that improves the previous neural network-based NER tool by employing a multi-task NER model and neural network-based NEN models to achieve much faster and more accurate inference. This repository provides a way to host your own BERN2 server. Currently, BERN2 is running on a hosting server with 64-core CPU, 512GB Memory, and 12GB GPU. See our [paper](https://arxiv.org/abs/2201.02080) for more details.

***** **Try BERN2 at [http://bern2.korea.ac.kr](http://bern2.korea.ac.kr)** ***** 

### Updates
* \[**Jun 26, 2022**\] We updated our resource file ([resources_v1.1.b.tar.gz](http://nlp.dmis.korea.edu/projects/bern2-sung-et-al-2022/resources_v1.1.b.tar.gz)) to address the issue regarding CRF++. (issue https://github.com/dmis-lab/BERN2/issues/17). 
* \[**Apr 14, 2022**\] We updated our resource file ([resources_v1.1.a.tar.gz](http://nlp.dmis.korea.edu/projects/bern2-sung-et-al-2022/resources_v1.1.a.tar.gz)) to address the issue where BERN2 is not working on Windows (issue https://github.com/dmis-lab/BERN2/issues/4). 
* \[**Apr 14, 2022**\] We increased the API limit of our [web service](http://bern2.korea.ac.kr) from '100 reqeusts per 100 seconds' to '**300** requests per 100 seconds' per user.
* \[**Mar 18, 2022**\] On the [web service](http://bern2.korea.ac.kr), we set the API limit of 100 requests per 100 seconds per user. For bulk requests, we highly recommend you to use the local installation.
* \[**Mar 17, 2022**\] [BERN2 v1.1](https://github.com/dmis-lab/BERN2/releases/tag/v1.1.0) has been released. Please see the release page for more information on what's new in this version.
* \[**Feb 15, 2022**\] [Bioregistry](https://bioregistry.io/) is used to standardize prefixes for normalized entity identifiers.

| old | new |
| :---:   | :-: |
| MESH:D009369  | [mesh:D009369](https://bioregistry.io/mesh:D009369)  |
| OMIM:608627 | [mim:608627](https://bioregistry.io/mim:608627) |
| CL_0000021 | [CL:0000021](https://bioregistry.io/CL:0000021) |
| CVCL_J260 | [cellosaurus:CVCL_J260](https://bioregistry.io/cellosaurus:CVCL_J260) |
| NCBI:txid10095 | [NCBITaxon:10095](https://bioregistry.io/NCBITaxon:10095) |
| EntrezGene:10533 |[NCBIGene:10533](https://bioregistry.io/NCBIGene:10533) |

* \[**Jan 07, 2022**\] [BERN2 v1.0](https://github.com/dmis-lab/BERN2/releases/tag/v1.0.0) has been released.

## Installing BERN2

You first need to install BERN2 and its dependencies.

```bash
# Install torch with conda (please check your CUDA version)
conda create -n bern2 python=3.7
conda activate bern2
conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch
conda install faiss-gpu libfaiss-avx2 -c conda-forge

# Check if cuda is available
python -c "import torch;print(torch.cuda.is_available())"

# Install BERN2
git clone git@github.com:dmis-lab/BERN2.git
cd BERN2
pip install -r requirements.txt

```

(Optional) If you want to use mongodb as a caching database, you need to install and run it.
```
# https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/#install-mongodb-community-edition-using-deb-packages
sudo systemctl start mongod
sudo systemctl status mongod
```

Then, you need to download resources (e.g., external modules or dictionaries) for running BERN2. Note that you will need 70GB of free disk space. You can also download the resource file from [google drive](https://drive.google.com/file/d/147b3OhU4IdQi121ZBUSqO1XKdKoXE5DK/view?usp=sharing).

```
wget http://nlp.dmis.korea.edu/projects/bern2-sung-et-al-2022/resources_v1.1.b.tar.gz
tar -zxvf resources_v1.1.b.tar.gz
md5sum resources_v1.1.b.tar.gz
# make sure the md5sum is 'c0db4e303d1ccf6bf56b42eda2fe05d0'
rm -rf resources_v1.1.b.tar.gz

# (For Linux/MacOS Users) install CRF 
cd resources/GNormPlusJava
tar -zxvf CRF++-0.58.tar.gz
mv CRF++-0.58 CRF
cd CRF
./configure --prefix="$HOME"
make
make install
cd ../../..

# (For Windows Users) install CRF 
cd resources/GNormPlusJava
unzip -zxvf CRF++-0.58.zip
mv CRF++-0.58 CRF
cd ../..
```

## Running BERN2

The minimum memory requirement for running BERN2 on GPU is 63.5GB of RAM & 5.05GB of GPU. 
The following command runs BERN2.
```
export CUDA_VISIBLE_DEVICES=0
cd scripts

# For Linux and MacOS
bash run_bern2.sh

# For Windows
bash run_bern2_windows.sh
```

(Optional) To restart BERN2, you need to run the following commands.
```
export CUDA_VISIBLE_DEVICES=0
cd scripts
bash stop_bern2.sh
bash run_bern2.sh
```

## Using BERN2

After successfully running BERN2 in your local environment, you can access it via RESTful API. 
If you want to use BERN2 without installing it locally, please see [here](http://bern2.korea.ac.kr/documentation#api_content) for instructions on how to use the web service.

### Plain Text as Input
```
import requests

def query_plain(text, url="http://localhost:8888/plain"):
    return requests.post(url, json={'text': text}).json()

if __name__ == '__main__':
    text = "Autophagy maintains tumour growth through circulating arginine."
    print(query_plain(text))
```

### PubMed ID (PMID) as Input
```
import requests

def query_pmid(pmids, url="http://localhost:8888/pubmed"):
    return requests.get(url + "/" + ",".join(pmids)).json()

if __name__ == '__main__':
    pmids = ["30429607", "29446767"]
    print(query_pmid(pmids))
```

## Annotations

```
wget http://nlp.dmis.korea.edu/projects/bern2-sung-et-al-2022/annotation_v1.1.tar.gz
```

NER and normalization for 33.4+ millions of PubMed articles (pubmed22n0001 ~ pubmed22n1114 (2021.12.12)) generated by [BERN2 v1.1](https://github.com/dmis-lab/BERN2/releases/tag/v1.1.0) (Compressed, 22 GB). The data provided by BERN2 is post-processed and may differ from the most current/accurate data available from [U.S. National Library of Medicine (NLM)](https://www.nlm.nih.gov/).

## Citation
```bibtex
@article{sung2022bern2,
    title={BERN2: an advanced neural biomedical namedentity recognition and normalization tool}, 
    author={Sung, Mujeen and Jeong, Minbyul and Choi, Yonghwa and Kim, Donghyeon and Lee, Jinhyuk and Kang, Jaewoo},
    year={2022},
    eprint={2201.02080},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Contact Information
For help or issues using BERN2, please submit a GitHub issue. Please contact Mujeen Sung (`mujeensung (at) korea.ac.kr`), or Minbyul Jeong (`minbyuljeong (at) korea.ac.kr`) for communication related to BERN2.
