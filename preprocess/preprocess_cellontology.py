import csv

input_path = "../resources/normalization/resources/rawdata/CL_20210810.csv"
output_path = "../resources/normalization/resources/dictionary/best_dict_CellType_20210810.txt"

cui2names = {}
with open(input_path) as f:
    rdr = csv.reader(f)
    for line in rdr:
        class_id = line[0]
        # only consider CL
        if not class_id.split("/")[-1].startswith("CL"):
            continue
        cui = class_id.split("/")[-1]
        name = line[1]
        synonyms = line[2].split("|")

        if line[2].strip() != '':
            cui2names[cui] = '|'.join([name] + synonyms)
        else:
            cui2names[cui] = name
# save
with open(output_path, 'w') as f:
    for cui, names in cui2names.items():
        f.write(cui + "||" + names)
        f.write("\n")   

