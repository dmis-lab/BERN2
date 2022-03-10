from typing import Optional

import bioregistry

COLOR_DICT = {
    'disease': (228, 26, 28),
    'mutation': (55, 126, 184),
    'drug': (77, 175, 74),
    'gene': (152, 78, 163),
    'species': (255, 127, 0),
    'DNA': (255, 255, 51),
    'RNA': (166, 86, 40),
    'cell_line': (247, 129, 191),
    'cell_type': (153, 153, 153)
}

def id2url(_id):
    """Generate a URL for the given compact URI (CURIE), if possible.

    :param curie: A compact URI (CURIE) in the form of `prefix:identifier`
    :returns: A URL string if the Bioregistry can construct one, otherwise None.

    >>> id2url("mesh:D009369")
    'https://bioregistry.io/mesh:D009369'
    >>> id2url("NCBIGene:10533")
    'https://bioregistry.io/NCBIGene:10533'
    """

    return f"https://bioregistry.io/{_id}"

class Denotation:
    def __init__(self, obj_id=None, point=None, offset=None, key=None, info=None, type=None, mention=None):
        self.obj_id = obj_id
        self.point = point
        self.offset = offset
        self.key = key
        self.info = info
        self.type = type
        self.mention = mention

        self.ids = self.info['id']
        if isinstance(self.ids, str):
            self.ids = [self.ids]
        
        if self.info['is_neural_normalized']:
            self.mark = "*"
        else:
            self.mark = ""
        
        self.anchor_text = self.id2anchor()

        self.className = "span-{}".format(self.key)
        if self.key == "mutation":
            self.toolTip = "Mention: {}\nEntity type: {}\nID: {}{}\nMutation type: {}\nNormalized name: {}".format(
                self.mention,
                self.type,
                self.info['normalizedName'],
                self.mark,
                self.info['mutationType'],
                self.info['normalizedName']
            )
            self.htmlToolTip = "<div class='entity-box py-2'> \
                <span class='entity-header'> \
                    Mention \
                </span><span class='entity-body'> \
                    : {} \
                </span><br><span class='entity-header'> \
                    Entity type \
                </span><span class='entity-body'> \
                    : {} \
                </span><br><span class='entity-header'> \
                    ID \
                </span><span class='entity-body'> \
                    : {} \
                </span><br><span class='entity-header'> \
                    Mutation type \
                </span><span class='entity-body'> \
                    : {} \
                </span><br><span class='entity-header'> \
                    Normalized name \
                </span><span class='entity-body'> \
                    : {} \
                </span></div>".format(self.mention, self.type, self.anchor_text, self.info['mutationType'], self.info['normalizedName'])

        else:
            self.toolTip = "Mention: {}\nEntity type: {}\nID: {}{}".format(
                self.mention,
                self.type,
                ",".join(self.ids),
                self.mark
            )
            self.htmlToolTip = "<div class='entity-box py-2'> \
                <span class='entity-header'> \
                    Mention \
                </span><span class='entity-body'> \
                    : {} \
                </span><br><span class='entity-header'> \
                    Entity type \
                </span><span class='entity-body'> \
                    : {} \
                </span><br><span class='entity-header'> \
                    ID \
                </span><span class='entity-body'> \
                    : {} \
                </span></div>".format(self.mention, self.type, self.anchor_text)
    
    def id2anchor(self):
        if self.key == "mutation":
            anchor_text = "{}{}".format(self.info['normalizedName'], self.mark)
        else:
            anchor_texts = []
            for _id in self.ids:
                _url = id2url(_id)
                if _url == "":
                    _a_text = _id
                else:
                    _a_text = "<a href='{}' target='_blank'>{}</a>".format(_url, _id)
                anchor_texts.append(_a_text)
            anchor_text = "{}{}".format(",".join(anchor_texts), self.mark)
        
        return anchor_text


    def to_span_text(self, tagname="span"):
        if self.point == 'end':
            return "</{}>".format(tagname)
        else:
            return "<{} data-tooltip-position='top' data-tooltip=\"{}\">".format(
                tagname,
                self.toolTip
            )

class DenotationStack:
    def __init__(self, result_id="text"):
        self.result_id=result_id
        self.ids = []
        self.dict = {}
    
    def _length(self):
        return len(self.ids)
    
    def _add(self, d_item):
        self.ids.append(d_item.obj_id)
        self.dict[d_item.obj_id] = d_item
    
    def _pop(self, d_item):
        if d_item.obj_id in self.ids:
            self.ids.pop(self.ids.index(d_item.obj_id))
            self.dict.pop(d_item.obj_id, None)
    
    def _contains(self, d_item):
        return d_item.obj_id in self.ids
    
    def merge_colors(self):
        merge_colors = [COLOR_DICT[d_item.key] for d_id, d_item in self.dict.items()]
        c_r = int(sum([c[0] for c in merge_colors])/len(merge_colors))
        c_g = int(sum([c[1] for c in merge_colors])/len(merge_colors))
        c_b = int(sum([c[2] for c in merge_colors])/len(merge_colors))
        
        # return (int(c_r/len(merge_colors)), int(c_g/len(merge_colors)), int(c_b/len(merge_colors)))
        return "{},{},{}".format(c_r, c_g, c_b)
    
    def to_span_text(self):
        if self._length() == 0:
            return "</span>"
        return "<{} data-tooltip-position='top' data-tooltip=\"{}\" style='background-color: rgba({}, 0.3);'>".format(
            "span",
            # " ".join([d_item.className for d_id, d_item in self.dict.items()]),
            "\n\n".join([d_item.toolTip for d_id, d_item in self.dict.items()]),
            self.merge_colors()
        )
    
    def to_span_div_text(self, offset="0"):
        span_text = "</span>"
        div_text = ""
        if self._length() != 0:
            span_text = "<{} id='{}' class='stack-spans {}' style='background-color: rgba({}, 0.3);'>".format(
                "span",
                "{}_{}_span".format(self.result_id, offset),
                " ".join("obj_{}_{}".format(self.result_id, d_item.obj_id) for d_id, d_item in self.dict.items()),
                self.merge_colors()
            )
            # div_text = " ".join(d_item.htmlToolTip for d_id, d_item in self.dict.items())
            div_text = "<div id='{}' class='stack-tooltips' role='tooltip'>{}<div class='arrow' data-popper-arrow></div></div>".format(
                "{}_{}_tooltip".format(self.result_id, offset),
                " ".join(d_item.htmlToolTip for d_id, d_item in self.dict.items())
            )
        
        return span_text, div_text
        

class ResultParser:
    def __init__(self):
        self.entity_type_dict = {
            'disease': 'Disease',
            'gene': 'Gene/Protein',
            'drug': 'Drug/Chemical',
            'species': 'Species',
            'mutation': 'Mutation',
            'DNA': 'DNA',
            'RNA': 'RNA',
            'cell_line': 'Cell line',
            'cell_type': 'Cell type'
        }
    
    def parse_result(self, result_dict, draw_keys, result_id="text"):
        _keys = draw_keys
        parsed_annotations = {}

        org_text = result_dict['text']

        keys_in_dict = []

        if 'annotations' not in result_dict.keys():
            # TODO: handle with error_code stuff
            # Error case
            if 'pmid' in result_dict.keys():
                return "error", "{} is not a valid PMID".format(result_dict['pmid']), {}
            else:
                return "error", "Empty text as an input", {}
            # result_dict['annotations'] = []
        
        if "error_code" in result_dict.keys():
            if int(result_dict["error_code"]) != 0:
                return "error", result_dict["error_message"], {}
        
        for d_idx, d_info in enumerate(result_dict['annotations']):
            # if not is_neural_normalized:
            #     if d_info['is_neural_normalized']:
            #         continue
            if d_info['obj'] not in keys_in_dict:
                keys_in_dict.append(d_info['obj'])
            s_offset = int(d_info['span']['begin'])
            e_offset = int(d_info['span']['end'])

            mention = org_text[s_offset:e_offset]

            d_s_item = Denotation(obj_id = str(d_idx), point = 'start', offset = s_offset, key = d_info['obj'], info = d_info, type = self.entity_type_dict[d_info['obj']], mention=mention)
            d_e_item = Denotation(obj_id = str(d_idx), point = 'end', offset = e_offset, key = d_info['obj'], info = d_info, type = self.entity_type_dict[d_info['obj']], mention=mention)

            if s_offset in parsed_annotations.keys():
                parsed_annotations[s_offset].append(d_s_item)
            else:
                parsed_annotations[s_offset] = [d_s_item]
            
            if e_offset in parsed_annotations.keys():
                parsed_annotations[e_offset].append(d_e_item)
            else:
                parsed_annotations[e_offset] = [d_e_item]
        
        split_text = []
        prev_offset = 0

        d_stack = DenotationStack(result_id=result_id)

        _offsets = list(parsed_annotations.keys())
        _offsets = sorted(_offsets)
        for _offset in _offsets:
            _token = org_text[prev_offset:_offset]
            split_text.append(_token)

            if d_stack._length() > 0:
                split_text.append("</span>")

            for d_item in parsed_annotations[_offset]:
                if d_item.point == 'end':
                    d_stack._pop(d_item)
                else:
                    d_stack._add(d_item)
            
            if d_stack._length() > 0:
                span_text, div_text = d_stack.to_span_div_text(offset=str(_offset))
                split_text.append(div_text + span_text)
                # split_text.append(d_stack.to_span_text())
            
            prev_offset = _offset
        
        split_text.append(org_text[prev_offset:])

        return "success", ''.join(split_text), keys_in_dict