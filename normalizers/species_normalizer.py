class SpeciesNormalizer(object):
    def __init__(self, dict_path):
        self.NO_ENTITY_ID = 'CUI-less'

        # Create dictionary for exact match
        self.species2oid = dict()
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                oid, names = line[:-1].split('||')
                names = names.split('|')
                for name in names:
                    # a part of tmChem normalization
                    self.species2oid[name] = oid

    def normalize(self, names):
        oids = list()
        for name in names:
            if name in self.species2oid:
                oids.append(self.species2oid[name])
            elif name.lower() in self.species2oid:
                oids.append(self.species2oid[name.lower()])
            else:
                oids.append(self.NO_ENTITY_ID)
        
        return oids