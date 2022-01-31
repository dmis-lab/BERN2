input_path = "../resources/normalization/resources/rawdata/cellosaurus_20210520.txt"
output_path = "../resources/normalization/resources/dictionary/best_dict_CellLine_20210520.txt"

cui2names = {}
with open(input_path) as f:
    for line in f:
        if line.startswith(" "):
            continue
        line = line.strip()
        if line.startswith("ID"):
            name = ' '.join(line.split()[1:])
        elif line.startswith("AC"):
            cui = ' '.join(line.split()[1:])
            
            assert cui not in cui2names
            assert name != ''

            cui2names[cui] = name
            name = ''
        
        # synonyms
        elif line.startswith("SY"):
            synonyms = ' '.join(line.split()[1:]).split(";")
            synonyms = [sy.strip() for sy in synonyms]

            cui2names[cui] = '|'.join([cui2names[cui]] + synonyms)

# save
with open(output_path, 'w') as f:
    for cui, names in cui2names.items():
        f.write(cui + "||" + names)
        f.write("\n")   

