"""
input: dictionary with bern id
output: dictionary with official id (MESH, CHEBI)
"""

bern_dict_path = './normalization/resources/dictionary/best_dict_Disease.txt'
meta_dict_path = './normalization/resources/meta/disease_meta_190310.tsv'
save_path = './normalization/resources/dictionary_bern2/dict_Disease.txt'

BID2NAMES= {}
# 1) generate BID2NAMES = {}
with open(bern_dict_path) as f:
    for line in f:
        line = line.strip()
        bid, names = line.split("||")
        assert bid not in BID2NAMES

        BID2NAMES[bid] = names

OID2BID = {}
# 2) generate OID2BID = {}
with open(meta_dict_path) as f:
    for line in f:
        line = line.strip()
        bid, oid = line.split("\t")
        if oid in OID2BID:
            OID2BID[oid] += "|" + bid
        else:
            OID2BID[oid] = bid

OID2NAMES = {}
# 3) create OID2NAMES 
for oid in OID2BID:
    bids = OID2BID[oid].split("|")
    names = []
    for bid in bids:
        local_names = BID2NAMES[bid]
        names += local_names.split("|")
    names = list(dict.fromkeys(names))
    names = '|'.join(names)
    OID2NAMES[oid] = names

# 4) save 
with open(save_path, 'w') as f:
    for oid in OID2NAMES:
        names = OID2NAMES[oid]
        line = oid + "||" + names
        f.write(line)
        f.write("\n")