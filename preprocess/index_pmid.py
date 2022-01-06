import json
import requests
import pprint
from tqdm import tqdm
# import pymongo
# from pymongo import MongoClient

import pubmed_parser as pp

# client = MongoClient('163.152.163.44:27017')
# db = client.bern
# collection = db.pmid

# get input text
def generate_input(path=None):
    path = '/hdd1/minbyul/dataset/pubmed/'
    for idx in range(839, 840):
        pubmed_dict = pp.parse_medline_xml(path + "pubmed21n%04g"%idx + ".xml.gz")
        for ent_idx, ent in enumerate(pubmed_dict):
            if ent_idx > 12209:
                yield idx, ent_idx, ent


def query_raw(text, url="http://163.152.20.133:8898"):
    body_data ={"param":
        json.dumps({"text":text})
    }
    # return requests.post(url, data=body_data).json()
    return requests.post(url, data=body_data)


def main():
    input_generator = generate_input()

    for data_idx, input_idx, input in tqdm(input_generator):
        pmid = input['pmid']
        title = input['title']
        abstract = input['abstract']

        text = title + "\n" + abstract

        result = query_raw(text)
        try:
            json_result = result.json()
        
            json_result['_id'] = pmid
        except:
            continue

        if result.status_code != 200:
            with open('error_pmid.txt', 'a') as out_:
                out_.write("{}\n".format(pmid))

        else:
            # success
            try:
                with open("/hdd1/minbyul/dataset/pubmed_preprocess/pubmed21n%4g.json"%(data_idx), 'a') as fp:
                    json.dump(json_result, fp)
                # collection.insert_one(json_result)
            # except pymongo.errors.DuplicateKeyError:
            #     print(f"duplicated pmid ={pmid}")
            except:
                with open('error_pmid.txt', 'a') as out_:
                    out_.write("{}\n".format(pmid))

    # retrieve
    # input_generator = generate_input()
    # for input in input_generator:
    #     pmid = input['pmid'] + '1'
    #     output = collection.find_one({"_id": pmid})
    #     import pdb ; pdb.set_trace()


if __name__ == "__main__":
    main()

