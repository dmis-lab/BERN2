from datetime import datetime
import os
import time
import socket
import threading

from normalizers.chemical_normalizer import ChemicalNormalizer
from normalizers.species_normalizer import SpeciesNormalizer
from normalizers.cellline_normalizer import CellLineNormalizer
from normalizers.celltype_normalizer import CellTypeNormalizer
from normalizers.neural_normalizer import NeuralNormalizer

time_format = '[%d/%b/%Y %H:%M:%S.%f]'


class Normalizer:
    def __init__(self, use_neural_normalizer, gene_port=18888, disease_port=18892, no_cuda=False):
        # Normalizer paths
        self.BASE_DIR = 'resources/normalization/'
        self.NORM_INPUT_DIR = {
            'disease': os.path.join(self.BASE_DIR, 'inputs/disease'),
            'gene': os.path.join(self.BASE_DIR, 'inputs/gene'),
        }
        self.NORM_OUTPUT_DIR = {
            'disease': os.path.join(self.BASE_DIR, 'outputs/disease'),
            'gene': os.path.join(self.BASE_DIR, 'outputs/gene'),
        }
        self.NORM_DICT_PATH = {
            'drug': os.path.join(self.BASE_DIR,
                                'dictionary/dict_ChemicalCompound_20210630.txt'),
            'gene': 'setup.txt',
            'species': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_Species.txt'),
            'cell_line': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_CellLine_20210520.txt'),
            'cell_type': os.path.join(self.BASE_DIR,
                                    'dictionary/dict_CellType_20210810.txt'),
        }

        # checkpoint on huggingface hub
        self.NEURAL_NORM_MODEL_PATH = {
            'disease':'dmis-lab/biosyn-sapbert-bc5cdr-disease',
            'drug':'dmis-lab/biosyn-sapbert-bc5cdr-chemical',
            'gene':'dmis-lab/biosyn-sapbert-bc2gn',
        }
        self.NEURAL_NORM_CACHE_PATH = {
            'disease':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_Disease_20210630.txt.pk'),
            'drug':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_ChemicalCompound_20210630.txt.pk'),
            'gene':os.path.join(self.BASE_DIR,
                    'normalizers/neural_norm_caches/dict_Gene.txt.pk'),
        }
        
        self.NORM_MODEL_VERSION = 'dmis ne norm v.20220226'

        self.HOST = '127.0.0.1'

        # normalizer port
        self.GENE_PORT = gene_port
        self.DISEASE_PORT = disease_port

        self.NO_ENTITY_ID = 'CUI-less'

        self.chemical_normalizer = ChemicalNormalizer(self.NORM_DICT_PATH['drug'])
        self.species_normalizer = SpeciesNormalizer(self.NORM_DICT_PATH['species'])
        self.cellline_normalizer = CellLineNormalizer(self.NORM_DICT_PATH['cell_line'])
        self.celltype_normalizer = CellTypeNormalizer(self.NORM_DICT_PATH['cell_type'])
        
        # neural normalizer
        self.neural_disease_normalizer = None
        self.neural_chemical_normalizer = None
        self.neural_gene_normalizer = None
        self.use_neural_normalizer = use_neural_normalizer

        if self.use_neural_normalizer:
            print("start loading neural normalizer..")
            self.neural_disease_normalizer = NeuralNormalizer(
                model_name_or_path=self.NEURAL_NORM_MODEL_PATH['disease'],
                cache_path=self.NEURAL_NORM_CACHE_PATH['disease'],
                no_cuda=no_cuda,
            )
            print(f"neural_disease_normalizer is loaded.. model={self.NEURAL_NORM_MODEL_PATH['disease']} , dictionary={self.NEURAL_NORM_CACHE_PATH['disease']}")

            self.neural_chemical_normalizer = NeuralNormalizer(
                model_name_or_path=self.NEURAL_NORM_MODEL_PATH['drug'],
                cache_path=self.NEURAL_NORM_CACHE_PATH['drug'],
                no_cuda=no_cuda,
            )
            print(f"neural_chemical_normalizer is loaded.. model={self.NEURAL_NORM_MODEL_PATH['drug']} , dictionary={self.NEURAL_NORM_CACHE_PATH['drug']}")

            self.neural_gene_normalizer = NeuralNormalizer(
                model_name_or_path=self.NEURAL_NORM_MODEL_PATH['gene'],
                cache_path=self.NEURAL_NORM_CACHE_PATH['gene'],
                no_cuda=no_cuda,
            )
            print(f"neural_gene_normalizer is loaded.. model={self.NEURAL_NORM_MODEL_PATH['gene']} , dictionary={self.NEURAL_NORM_CACHE_PATH['gene']}")


    def normalize(self, base_name, doc_dict_list):
        start_time = time.time()

        names = dict()
        saved_items = list()
        ent_cnt = 0
        abs_cnt = 0

        for item in doc_dict_list:
            # Get json values
            content = item['abstract']
            # pmid = item['pmid']
            entities = item['entities']

            abs_cnt += 1

            # Iterate entities per abstract
            for ent_type, locs in entities.items():
                ent_cnt += len(locs)
                for loc in locs:

                    loc['end'] += 1

                    if ent_type == 'mutation':
                        name = loc['normalizedName']

                        if ';' in name:
                            name = name.split(';')[0]
                    else:
                        name = content[loc['start']:loc['end']]

                    if ent_type in names:
                        names[ent_type].append([name, len(saved_items)])
                    else:
                        names[ent_type] = [[name, len(saved_items)]]

            # Work as pointer
            item['norm_model'] = self.NORM_MODEL_VERSION
            saved_items.append(item)

        # For each entity,
        # 1. Write as input files to normalizers
        # 2. Run normalizers
        # 3. Read output files of normalizers
        # 4. Remove files
        # 5. Return oids

        # Threading
        results = list()
        threads = list()
        for ent_type in names.keys():
            t = threading.Thread(target=self.run_normalizers_wrap,
                                 args=(ent_type, base_name, names, saved_items, results))
            t.daemon = True
            t.start()
            threads.append(t)

        # block until all tasks are done
        for t in threads:
            t.join()

        # Save oids
        for ent_type, type_oids in results:
            oid_cnt = 0
            for saved_item in saved_items:
                for loc in saved_item['entities'][ent_type]:

                    # Put oid
                    loc['id'] = type_oids[oid_cnt]
                    loc['is_neural_normalized'] = False
                    oid_cnt += 1

        print(datetime.now().strftime(time_format),
              '[{}] Rule-based normalization '
              '{:.3f} sec ({} article(s), {} entity type(s))'
              .format(base_name, time.time() - start_time, abs_cnt,
                      len(names.keys())))

        return saved_items

    # normalize using neural model
    def neural_normalize(self, ent_type, tagged_docs):
        abstract = tagged_docs[0]['abstract']
        entities = tagged_docs[0]['entities'][ent_type]
        entity_names = [abstract[e['start']:e['end']] for e in entities]
        cuiless_entity_names = []
        for entity, entity_name in zip(entities, entity_names):
            if entity['id'] == self.NO_ENTITY_ID:
                cuiless_entity_names.append(entity_name)
        cuiless_entity_names = list(set(cuiless_entity_names))
        
        if len(cuiless_entity_names) == 0:
            return tagged_docs
        print(f"# cui-less in {ent_type}={len(cuiless_entity_names)}")
        if ent_type == 'disease':
            norm_entities = self.neural_disease_normalizer.normalize(
                names=cuiless_entity_names, 
            )
        elif ent_type == 'drug':
            norm_entities = self.neural_chemical_normalizer.normalize(
                names=cuiless_entity_names, 
            )
        elif ent_type == 'gene':
            norm_entities = self.neural_gene_normalizer.normalize(
                names=cuiless_entity_names, 
            )
        
        cuiless_entity2norm_entities = {c:n for c, n in zip(cuiless_entity_names,norm_entities)}
        for entity, entity_name in zip(entities, entity_names):
            if entity_name in cuiless_entity2norm_entities:
                cui = cuiless_entity2norm_entities[entity_name][0]
                entity['id'] = cui if cui != -1 else self.NO_ENTITY_ID
                entity['is_neural_normalized'] = True
            else:
                entity['is_neural_normalized'] = False
        
        return tagged_docs

    def run_normalizers_wrap(self, ent_type, base_name, names, saved_items, results):
        results.append((ent_type,
                        self.run_normalizer(ent_type, base_name, names, saved_items)))

    def run_normalizer(self, ent_type, base_name, names, saved_items):
        start_time = time.time()
        name_ptr = names[ent_type]
        oids = list()
        bufsize = 4

        base_thread_name = base_name
        input_filename = base_thread_name + '.concept'
        output_filename = base_thread_name + '.oid'

        print(f'ent_type = {ent_type}')

        # call sieve normalizer
        if ent_type == 'disease':
            # 1. Write as input files to normalizers
            norm_inp_path = os.path.join(self.NORM_INPUT_DIR[ent_type],
                                         input_filename)
            norm_abs_path = os.path.join(self.NORM_INPUT_DIR[ent_type],
                                         base_thread_name + '.txt')
            with open(norm_inp_path, 'w') as norm_inp_f:
                for name, _ in name_ptr:
                    norm_inp_f.write(name + '\n')
            # created for disease normalizer
            with open(norm_abs_path, 'w') as _:
                pass

            # 2. Run normalizers
            s = socket.socket()
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                s.connect((self.HOST, self.DISEASE_PORT))
                s.send('{}'.format(base_thread_name).encode('utf-8'))
                s.recv(bufsize)
            except ConnectionRefusedError as cre:
                print('Check Sieve jar', cre)
                os.remove(norm_inp_path)
                os.remove(norm_abs_path)
                s.close()
                return oids
            s.close()

            # 3. Read output files of normalizers
            norm_out_path = os.path.join(self.NORM_OUTPUT_DIR[ent_type],
                                         output_filename)
            if os.path.exists(norm_out_path):
                with open(norm_out_path, 'r') as norm_out_f:
                    for line in norm_out_f:
                        oid = line[:-1]
                        if oid != self.NO_ENTITY_ID:
                            oids.append(oid)
                        else:
                            oids.append(self.NO_ENTITY_ID)
                os.remove(norm_out_path)
            else:
                print('Not found!!!', norm_out_path)

                # Sad error handling
                for _ in range(len(name_ptr)):
                    oids.append(self.NO_ENTITY_ID)

        elif ent_type == 'drug':
            names = [ptr[0] for ptr in name_ptr]
            preds = self.chemical_normalizer.normalize(names)
            for pred in preds:
                oids.append(pred)            

        elif ent_type == 'mutation':
            # pass because tmVar does mutation normalization
            pass

        elif ent_type == 'species':
            names = [ptr[0] for ptr in name_ptr]
            preds = self.species_normalizer.normalize(names)
            for pred in preds:
                if pred != self.NO_ENTITY_ID:
                    pred = int(pred) // 100
                    # https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=10095
                    # "... please use NCBI:txid10095 ..."
                    oids.append('NCBI:txid{}'.format(pred))
                else:
                    oids.append(self.NO_ENTITY_ID)
        
        elif ent_type == 'cell_line':
            names = [ptr[0] for ptr in name_ptr]
            preds = self.cellline_normalizer.normalize(names)
            for pred in preds:
                if pred != self.NO_ENTITY_ID:
                    oids.append(pred)
                else:
                    oids.append(self.NO_ENTITY_ID)

        elif ent_type == 'cell_type':
            names = [ptr[0] for ptr in name_ptr]
            preds = self.celltype_normalizer.normalize(names)
            for pred in preds:
                if pred != self.NO_ENTITY_ID:
                    oids.append(pred)
                else:
                    oids.append(self.NO_ENTITY_ID)

        # call GNormPlus
        elif ent_type == 'gene':
            # create socket
            # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s = socket.socket()
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                s.connect((self.HOST, self.GENE_PORT))
            except ConnectionRefusedError as cre:
                print('Check GNormPlus jar', cre)
                s.close()
                return oids

            # 1. Write as input files to normalizers
            norm_inp_path = os.path.join(self.NORM_INPUT_DIR[ent_type],
                                         input_filename)
            norm_abs_path = os.path.join(self.NORM_INPUT_DIR[ent_type],
                                         base_thread_name + '.txt')

            space_type = ' ' + ent_type
            with open(norm_inp_path, 'w') as norm_inp_f:
                with open(norm_abs_path, 'w') as norm_abs_f:
                    for saved_item in saved_items:
                        entities = saved_item['entities'][ent_type]
                        if len(entities) == 0:
                            continue

                        abstract_title = saved_item['abstract']

                        ent_names = list()
                        for loc in entities:
                            e_name = abstract_title[loc['start']:loc['end']]
                            if len(e_name) > len(space_type) \
                                    and space_type \
                                    in e_name.lower()[-len(space_type):]:
                                # print('Replace', e_name,
                                #       'w/', e_name[:-len(space_type)])
                                e_name = e_name[:-len(space_type)]

                            ent_names.append(e_name)
                        norm_abs_f.write(saved_item['pmid'] + '||' +
                                         abstract_title + '\n')
                        norm_inp_f.write('||'.join(ent_names) + '\n')

            # 2. Run normalizers
            gene_input_dir = os.path.abspath(
                os.path.join(self.NORM_INPUT_DIR[ent_type]))
            gene_output_dir = os.path.abspath(
                os.path.join(self.NORM_OUTPUT_DIR[ent_type]))
            setup_dir = self.NORM_DICT_PATH[ent_type] # setup.txt

            # start jar
            jar_args = '\t'.join(
                [gene_input_dir, gene_output_dir, setup_dir, '9606',  # human
                 base_thread_name]) + '\n'
            s.send(jar_args.encode('utf-8'))
            # input_stream = struct.pack('>H', len(jar_args)) + jar_args.encode('utf-8')
            # s.send(input_stream)
            s.recv(bufsize)
            s.close()

            # 3. Read output files of normalizers
            norm_out_path = os.path.join(gene_output_dir, output_filename)
            if os.path.exists(norm_out_path):
                with open(norm_out_path, 'r') as norm_out_f, \
                        open(norm_inp_path, 'r') as norm_in_f:
                    for line, input_l in zip(norm_out_f, norm_in_f):
                        gene_ids, gene_mentions = line[:-1].split('||'), \
                                                  input_l[:-1].split('||')
                        for gene_id, gene_mention in zip(gene_ids,
                                                         gene_mentions):
                            eid = None
                            if gene_id.lower() == 'cui-less':
                                eid = self.NO_ENTITY_ID
                            else:
                                bar_idx = gene_id.find('-')
                                if bar_idx > -1:
                                    gene_id = gene_id[:bar_idx]
                                eid = gene_id
                                eid = "EntrezGene:" + eid
                            
                            oids.append(eid)

                # 5. Remove output files
                os.remove(norm_out_path)
            else:
                print('Not found!!!', norm_out_path)

                # Sad error handling
                for _ in range(len(name_ptr)):
                    oids.append(self.NO_ENTITY_ID)

            # 4. Remove input files
            os.remove(norm_inp_path)
            os.remove(norm_abs_path)

        else:
            print(f"WARN! {ent_type} is not supported yet")
            names = [ptr[0] for ptr in name_ptr]
            for name in names:
                oids.append(self.NO_ENTITY_ID)

        # 5. Return oids
        assert len(oids) == len(name_ptr), '{} vs {} in {}'.format(
            len(oids), len(name_ptr), ent_type)

        # double checking
        if 0 == len(oids):
            return oids

        cui_less_count = 0
        for oid in oids:
            if self.NO_ENTITY_ID == oid:
                cui_less_count += 1

        print(datetime.now().strftime(time_format),
              '[{}] [{}] {:.3f} sec, CUI-less: {:.1f}% ({}/{})'.format(
                  base_name, ent_type, time.time() - start_time,
                  cui_less_count * 100. / len(oids),
                  cui_less_count, len(oids)))
        return oids
