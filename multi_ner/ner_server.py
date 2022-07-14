import os
import json
import socket
import struct
import argparse

from datetime import datetime
from main import MTNER
from ops import filter_entities, pubtator2dict_list

def count_entities(data):
    num_entities = 0
    for d in data:
        if 'entities' not in d:
            continue
        for ent_type, entities in d['entities'].items():
            num_entities += len(entities)

    return num_entities

def mtner_recognize(model, dict_path, base_name, args):
    input_mt_ner = os.path.join(args.mtner_home, 'input',
                                f'{dict_path[2:]}.PubTator')
    output_mt_ner = os.path.join(args.mtner_home, 'output',
                                f'{dict_path[2:]}.json')
    
    dict_list = pubtator2dict_list(input_mt_ner)

    res = model.recognize(
        input_dl=dict_list,
        base_name=base_name
    )

    if res is None:
        return None, 0

    num_filtered_species_per_doc = filter_entities(res)
    for n_f_spcs in num_filtered_species_per_doc:
        if n_f_spcs[1] > 0:
            print(datetime.now().strftime(args.time_format),
                  '[{}] Filtered {} species'
                  .format(base_name, n_f_spcs[1]))
    num_entities = count_entities(res)

    res[0]['num_entities'] = num_entities
    # Write output str to a .PubTator format file
    with open(output_mt_ner, 'w', encoding='utf-8') as f:
        json.dump(res[0], f)

def run_server(model, args):
    host = args.mtner_host
    port = args.mtner_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        while True:
            conn, addr = s.accept()
            dict_path = conn.recv(512).decode('utf-8')
            base_name = dict_path.split('.')[0]
            # hotfix
            base_name = base_name.replace("\x00A","")
            
            mtner_recognize(model, dict_path, base_name, args)
            
            output_stream = struct.pack('>H', len(dict_path)) + dict_path.encode(
                'utf-8')

            conn.send(output_stream)
            conn.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, help='random seed for initialization',
                            default=1)
    argparser.add_argument('--model_name_or_path', default='dmis-lab/bern2-ner')
    argparser.add_argument('--max_seq_length', type=int, help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.',
                            default=128)
    argparser.add_argument('--mtner_home',
                           help='biomedical language model home')         
    argparser.add_argument('--mtner_host',
                           help='biomedical language model host', default='localhost')
    argparser.add_argument('--mtner_port', type=int, 
                           help='biomedical language model port', default=18894)
    argparser.add_argument('--time_format',
                            help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')    
    argparser.add_argument('--no_cuda', action="store_true", help="Avoid using CUDA when available")
    args = argparser.parse_args()
    mt_ner = MTNER(args)
    
    run_server(mt_ner, args)