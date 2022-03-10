import copy
from datetime import datetime, timezone
import json
from operator import itemgetter
import xml.etree.ElementTree as ElTree
# from download import query_pubtator2biocxml


def pubtator2dict_list(pubtator_file_path):
    dict_list = list()

    title_pmid = ''
    # abstract_pmid = ''
    title = ''
    abstract_text = ''
    doc_line_num = 0
    mutations = list()

    with open(pubtator_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                if len(mutations) > 0:
                    if len(mutations) > 1:
                        mutations = sorted(mutations,
                                           key=itemgetter('start'))

                    mutations = get_bestplus_spans(mutations, abstract_text)
                    
                doc_dict = {
                    'pmid': title_pmid,
                    'mutation_model': 'tmVar 2.0',
                    'entities': {'mutation': copy.deepcopy(mutations)}
                }
                doc_dict['abstract'] = abstract_text

                dict_list.append(doc_dict)

                doc_line_num = 0
                mutations.clear()
                continue

            if doc_line_num == 0:
                title_cols = line.split('|t|')

                if len(title_cols) != 2:
                    return '{"error": "wrong #title_cols {}"}'\
                        .format(len(title_cols))

                title_pmid = title_cols[0]

                if '- No text -' == title_cols[1]:
                    # make tmvar2 results empty
                    title = ''
                else:
                    title = title_cols[1]
            elif doc_line_num == 1:
                abstract_cols = line.split('|a|')

                if len(abstract_cols) != 2:
                    if len(abstract_cols) > 2:
                        abstract_text = "|a|".join(abstract_cols[1:])
                    else:
                        return '{"error": "wrong #abstract_cols {}"}' \
                            .format(len(abstract_cols))
                else:
                    if '- No text -' == abstract_cols[1]:
                        # make tmvar2 results empty
                        abstract_text = ''
                    else:
                        abstract_text = abstract_cols[1]
            elif doc_line_num > 1:
                mutation_cols = line.split('\t')

                if len(mutation_cols) != 6:
                    return '{"error": "wrong #mutation_cols {}"}' \
                        .format(len(mutation_cols))

                mutations.append({'start': int(mutation_cols[1]),
                                  'end': int(mutation_cols[2]),
                                  'mention': mutation_cols[3],
                                  'mutationType': mutation_cols[4],
                                  'normalizedName': mutation_cols[5]})

            doc_line_num += 1
    return dict_list

def preprocess(text):
    text = text.replace('\r ', ' ')

    text = text.replace('\u2028', ' ')
    text = text.replace('\u2029', ' ')

    # HAIR SPACE
    # https://www.fileformat.info/info/unicode/char/200a/index.htm
    text = text.replace('\u200A', ' ')

    # THIN SPACE
    # https://www.fileformat.info/info/unicode/char/2009/index.htm
    text = text.replace('\u2009', ' ')
    text = text.replace('\u2008', ' ')

    # FOUR-PER-EM SPACE
    # https://www.fileformat.info/info/unicode/char/2005/index.htm
    text = text.replace('\u2005', ' ')
    text = text.replace('\u2004', ' ')
    text = text.replace('\u2003', ' ')

    # EN SPACE
    # https://www.fileformat.info/info/unicode/char/2002/index.htm
    text = text.replace('\u2002', ' ')

    # NO-BREAK SPACE
    # https://www.fileformat.info/info/unicode/char/00a0/index.htm
    text = text.replace('\u00A0', ' ')

    # https://www.fileformat.info/info/unicode/char/f8ff/index.htm
    text = text.replace('\uF8FF', ' ')

    # https://www.fileformat.info/info/unicode/char/202f/index.htm
    text = text.replace('\u202F', ' ')

    text = text.replace('\uFEFF', ' ')
    text = text.replace('\uF044', ' ')
    text = text.replace('\uF02D', ' ')
    text = text.replace('\uF0BB', ' ')

    text = text.replace('\uF048', 'Η')
    text = text.replace('\uF0B0', '°')

    # MIDLINE HORIZONTAL ELLIPSIS: ⋯
    # https://www.fileformat.info/info/unicode/char/22ef/index.htm
    # text = text.replace('\u22EF', '...')

    return text


def pubtator2pubannotation(pubtator):
    dict_list = list()

    title_pmid = ''
    abstract_pmid = ''
    title = ''
    abstract_text = ''
    doc_line_num = 0
    entities = list()

    for line in pubtator.splitlines():
        if len(line) == 0:

            if title_pmid != abstract_pmid:
                return '{"error": "pmid disagreement"}'

            doc_dict = {
                'text': title + ' ' + abstract_text,
                'project': 'BERN',
                'sourcedb': 'PubMed',
                'sourceid': title_pmid,
                'annotations': copy.deepcopy(entities),
            }

            dict_list.append(doc_dict)

            doc_line_num = 0
            entities.clear()
            continue

        if doc_line_num == 0:
            title_cols = line.split('|t|')

            if len(title_cols) != 2:
                return '{"error": "wrong #title_cols=%d", "line": "%s"}' \
                       % (len(title_cols), line)

            title_pmid = title_cols[0]

            if '- No text -' == title_cols[1]:
                # make tmvar2 results empty
                title = ''
            else:
                title = title_cols[1]
        elif doc_line_num == 1:
            abstract_cols = line.split('|a|')

            if len(abstract_cols) != 2:
                return \
                    '{"error": "wrong #abstract_cols %d"}' % len(abstract_cols)

            abstract_pmid = abstract_cols[0]

            if '- No text -' == abstract_cols[1] \
                    or '-NoAbstract-' == abstract_cols[1]:
                # make tmvar2 results empty
                abstract_text = ''
            else:
                abstract_text = abstract_cols[1]
        elif doc_line_num > 1:
            entity_cols = line.split('\t')

            if len(entity_cols) != 6:
                return '{"error": "wrong #mutation_cols %d"}' % len(entity_cols)
                
            if entity_cols[4] in entity_cols:
                entities.append({'obj': entity_cols[4],
                                 'id': entity_cols[5].split('|'),
                                 'span': {
                                     'begin': int(entity_cols[1]),
                                     'end': int(entity_cols[2])
                                 },
                                 })

        doc_line_num += 1
    return dict_list


def get_bestplus_spans(mutations, title_space_abstract):
    adjusted_mutations = list()

    mention_count_dict = dict()
    for m in mutations:

        if 'No text' in m['mention']:
            continue

        # (20220113) hotfix
        if 'text ' in m['mention']:
            continue

        if m['mention'] in mention_count_dict:
            mention_count_dict[m['mention']] += 1
        else:
            mention_count_dict[m['mention']] = 1

        count = mention_count_dict[m['mention']]

        start = -1
        found = 0
        try:
            while found < count:
                start = title_space_abstract.index(m['mention'], start + 1)
                assert start > -1
                found += 1
        except ValueError:
            # hotfix for tmvar wrong mention
            continue
            
        end = start + len(m['mention']) - 1  # 2018.8.29 @chanho feedback

        assert m['mention'] == title_space_abstract[start: end + 1]

        adjusted_mutations.append({'start': start,
                                   'end': end,
                                   'mention': m['mention'],
                                   'mutationType': m['mutationType'],
                                   'normalizedName': m['normalizedName']})

    return adjusted_mutations


# Ref.
# http://pubannotation.org/docs/sourcedb/PubMed/sourceid/10022882/spans/606-710/annotations.json
# http://www.pubannotation.org/docs/annotation-format/
def get_pub_annotation(bern_dict):
    sourceid = bern_dict['pmid']

    sourcedb = ''
    text = bern_dict['abstract']

    pa_dict = {
        'project': 'BERN',
        'sourcedb': sourcedb,
        'sourceid': sourceid,
        'text': text,
        'annotations': bern2pub_annotation(bern_dict['entities'], bern_dict, text),
        'timestamp': datetime.now(tz=timezone.utc).strftime(
            '%a %b %d %H:%M:%S %z %Y')
    }
      
    return pa_dict


def bern2pub_annotation(entity_dict, bern_dict, text):
    entity_list = list()
    for etype in entity_dict:
        for entity_idx, entity in enumerate(entity_dict[etype]):

            # TODO prevention in the previous step
            if 'id' not in entity:
                entity['id'] = 'CUI-less'

            assert 'id' in entity, \
                '{}, entity={}, entity_dict={}'.format(
                    etype, entity, entity_dict)
            assert 'start' in entity and 'end' in entity, \
                '{}, entity={}, entity_dict={}'.format(
                    etype, entity, entity_dict)

            if '\t' in entity['id']:
                eid = entity['id'].split('\t')
            else:
                eid = [entity['id']]
            
            entity_pa_dict = {}
            if 'mutation' == etype:
                assert 'mutationType' in entity \
                       and 'normalizedName' in entity, \
                    '{}, entity={}, entity_dict={}'.format(
                        etype, entity, entity_dict)

                entity_pa_dict['mutationType'] = entity['mutationType']
                entity_pa_dict['normalizedName'] = entity['normalizedName']
                entity['end'] += 1 # tmvar2 end span makes one character shift in mention

            entity_pa_dict.update({
                'id': eid,
                'span': {
                    'begin': entity['start'],
                    'end': entity['end']
                },
                'obj': etype,
                'mention': text[entity['start']:entity['end']],
                'prob':bern_dict['prob'][etype][entity_idx][1] if etype in bern_dict['prob'] and bern_dict['prob'][etype][entity_idx][1] else None,
                'is_neural_normalized': entity['is_neural_normalized'] if 'is_neural_normalized' in entity else False
            })

            entity_list.append(entity_pa_dict)

    # sort by span begin
    def get_item_key1(item):
        return item['span']['begin']

    def get_item_key2(item):
        return item['obj']

    return sorted(sorted(entity_list, key=get_item_key2), key=get_item_key1)


def get_pubtator(bern_dict_list):
    result = ''
    for bd in bern_dict_list:
        text = bd['title'] + ' ' + bd['abstract']

        main = bd['pmid'] + '|t|' + bd['title'] + '\n' + \
            bd['pmid'] + '|a|' + bd['abstract']

        # sort by start
        sorted_entities = list()

        for etype in bd['entities']:
            for entity in bd['entities'][etype]:
                mention = text[entity['start']: entity['end']]
                sorted_entities.append(
                    [entity['start'], entity['end'], mention, etype,
                     '|'.join(entity['id'].split('\t'))])

        sorted_entities = sorted(sorted_entities, key=itemgetter(0))

        entities = ''
        for e in sorted_entities:
            entities += '{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                bd['pmid'], e[0], e[1], e[2], e[3], e[4])

        result += main + '\n' + entities + '\n'

    return result


def tmtooljson2bern(tmtool_res):
    tmtool_dicts = json.loads(tmtool_res)

    bern_dicts = list()

    for td in tmtool_dicts:
        mutations = list()
        for d in td['annotations']:
            mention = td['text'][d['span']['begin']: d['span']['end']]
            d['span']['end'] += 1
            mutations.append({
                'start': d['span']['start'],
                'end': d['span']['end'],
                'mention': mention,
                'normalizedName': d['obj'].replace('Mutation:', '')
            })

        doc_dict = {
            'pmid': td['sourceid'],
            'text': td['text'],
            'entities': {'mutation': mutations}
        }

        bern_dicts.append(doc_dict)

    return bern_dicts

