# bern_dict_path = '/home/bern/bern2/bern2_backend/normalization/resources/dictionary/best_dict_Disease.txt'
# new_dict_path = "/home/bern/bern2/bern2_backend/normalization/resources/rawdata/CTD_diseases_20210630.tsv"
# save_path = '/home/bern/bern2/bern2_backend/normalization/resources/dictionary/best_dict_Disease_20210630_tmp.txt'
# entity_type = "disease"
bern_dict_path = '/home/bern/bern2/bern2_backend/normalization/resources/dictionary/best_dict_ChemicalCompound.txt'
ctd_dict_path = "/home/bern/bern2/bern2_backend/normalization/resources/rawdata/CTD_chemicals_20210630.tsv"
chebi_dict_path = "/home/bern/bern2/bern2_backend/normalization/resources/rawdata/Chebi_names_20210801.tsv"
save_path = '/home/bern/bern2/bern2_backend/normalization/resources/dictionary/best_dict_ChemicalCompound20210630_tmp.txt'
entity_type = "chemical"
use_only_mesh = False

def load_new_dict(entity_type, ctd_path=None, chebi_path=None):
    new_dict_cuis = []
    new_cui2name = {}

    if entity_type == 'chemical':
        ctd_dict_cuis, ctd_cui2name = load_ctd(entity_type, ctd_path)
        chebi_dict_cuis, chebi_cui2name = load_chebi(chebi_path)
        new_dict_cuis = ctd_dict_cuis + chebi_dict_cuis
        new_cui2name = {**ctd_cui2name, **chebi_cui2name}

    return new_dict_cuis, new_cui2name

def load_chebi(chebi_path):
    new_dict_cuis = []
    new_cui2name = {}

    with open(chebi_path) as f:
        for line in f:
            line = line.strip()
            _, cui, _, _, name, _, lan = line.split("\t")

            if cui == 'COMPOUND_ID':
                continue

            if lan != 'en':
                continue
            
            if cui not in new_cui2name:
                new_cui2name[cui] = name
                new_dict_cuis.append(cui)
            else:
                new_cui2name[cui] += "|" + name
                new_cui2name[cui] = "|".join(list(dict.fromkeys(new_cui2name[cui].split("|"))))
    
    return new_dict_cuis, new_cui2name

def load_ctd(entity_type, ctd_path):
    new_dict_cuis = []
    new_cui2name = {}
    with open(ctd_path) as f:
        for line in f:
            if line.startswith("#"):
                continue

            line = line.strip()

            if entity_type == "disease":
                components = line.split("\t")
                name = components[0]
                cui = components[1]
                alt_cuis = components[2]
                if len(components)>7:
                    synonyms = components[7]
                    name = name + "|" + synonyms
                # if alt_cuis != '':
                if False:
                    alt_cuis = alt_cuis.split("|")
                    cuis = [cui] + alt_cuis
                else:
                    cuis = [cui]

                if use_only_mesh:
                    cuis = [cui for cui in cuis if "MESH" in cui]
                
                for cui in cuis:
                    new_dict_cuis.append(cui)
                    new_cui2name[cui] = name

            elif entity_type == "chemical":
                components = line.split("\t")
                name = components[0]
                if len(components)>7:
                    synonyms = components[7]
                    name = name + "|" + synonyms
                cui = components[1]
                cuis = [cui]
                if use_only_mesh:
                    cuis = [cui for cui in cuis if "MESH" in cui]

                for cui in cuis:
                    new_dict_cuis.append(cui)
                    new_cui2name[cui] = name
                
            else:
                import pdb ; pdb.set_trace()
                pass

    return new_dict_cuis, new_cui2name

# bern_dict_cuis = []
# with open(bern_dict_path) as f:
#     for line in f:
#         line = line.strip()
#         cuis, names = line.split("||")
#         cuis = cuis.split(",")
#         if use_only_mesh:
#             cuis = [cui for cui in cuis if "MESH" in cui]

#         for cui in cuis:
#             bern_dict_cuis.append(cui)

new_dict_cuis, new_cui2name = load_new_dict(entity_type, ctd_dict_path, chebi_dict_path)

# new_dict_cuis = []
# new_cui2name = {}
# with open(new_dict_path) as f:
#     for line in f:
#         if line.startswith("#"):
#             continue

#         line = line.strip()

#         if entity_type == "disease":
#             components = line.split("\t")
#             name = components[0]
#             cui = components[1]
#             alt_cuis = components[2]
#             if len(components)>7:
#                 synonyms = components[7]
#                 name = name + "|" + synonyms
#             # if alt_cuis != '':
#             if False:
#                 alt_cuis = alt_cuis.split("|")
#                 cuis = [cui] + alt_cuis
#             else:
#                 cuis = [cui]

#             meshs = [cui for cui in cuis if "MESH" in cui]
#             for mesh in meshs:
#                 new_dict_cuis.append(mesh)
#                 new_cui2name[mesh] = name
#         elif entity_type == "chemical":
#             components = line.split("\t")
#             name = components[0]
#             if len(components)>7:
#                 synonyms = components[7]
#                 name = name + "|" + synonyms
#             cui = components[1]
#             cuis = [cui]
#             meshs = [cui for cui in cuis if "MESH" in cui]
#             for mesh in meshs:
#                 new_dict_cuis.append(mesh)
#                 new_cui2name[mesh] = name
            
#         else:
#             import pdb ; pdb.set_trace()
#             pass
        
bern_dict_cuis = set(bern_dict_cuis)
new_dict_cuis = set(new_dict_cuis)


# for a in (new_dict_cuis-bern_dict_cuis):
#     print(f"{a}||{new_cui2name[a]}")


print(f"len(bern_dict_cuis)={len(bern_dict_cuis)}")
print(f"len(new_dict_cuis)={len(new_dict_cuis)}")
print(f"len(bern_dict_cuis-new_dict_cuis)={len(bern_dict_cuis-new_dict_cuis)}")
print(f"len(new_dict_cuis-bern_dict_cuis)={len(new_dict_cuis-bern_dict_cuis)}")
print(f"len(new_dict_cuis^bern_dict_cuis)={len(new_dict_cuis.intersection(bern_dict_cuis))}")
print(f"len(new_dict_cuis+bern_dict_cuis)={len(new_dict_cuis.union(bern_dict_cuis))}")

new_cuis = new_dict_cuis-bern_dict_cuis

with open(bern_dict_path) as fi, open(save_path, 'w') as fo:
    for line in fi:
        fo.write(line)

    for cui in new_cuis:
        fo.write(cui + "||" + new_cui2name[cui])
        fo.write("\n")