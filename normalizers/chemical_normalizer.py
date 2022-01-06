import string

class ChemicalNormalizer(object):
    def __init__(self, dict_path):
        self.NO_ENTITY_ID = 'CUI-less'

        # Create dictionary for exact match
        self.chem2oid = dict()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                oid, names = line[:-1].split('||')
                names = names.split('|')
                for name in names:
                    # a part of tmChem normalization
                    self.chem2oid[self.get_tmchem_name(name)] = oid

    def normalize(self, names):
        oids = list()
        for name in names:
            # a part of tmChem normalization
            normalized_name = self.get_tmchem_name(name)
             
            if normalized_name in self.chem2oid:
                oids.append(self.chem2oid[normalized_name])
            else:
                # oids.append('CUI-less')
                oids.append(self.NO_ENTITY_ID)
        
        return oids
    
    def get_tmchem_name(self, name):
        # 1. lowercase, 2. removes all whitespace and punctuation
        # https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S3
        normalized_name = ''
        for nc in name.lower():
            if nc == ' ' or nc in string.punctuation:
                continue
            normalized_name += nc
        return normalized_name
