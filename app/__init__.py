import os
import json

from flask import Flask, render_template, request, Response

try:
    from .result_parser import ResultParser
except ImportError:
    from result_parser import ResultParser

# Import Engine
import bern2

import time
    
def del_keys_from_dict(_dict, keys):
    for _key in keys:
        _dict.pop(_key, None)
    return _dict

def create_app(args):
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_mapping(
        SECRET_KEY="@#$%^BERN2%^FLASK@#$%^"
    )

    print(app.root_path)
    
    # LOAD MODEL
    if args.front_dev:
        model = None
    else:
        model = bern2.BERN2(
            mtner_home=args.mtner_home,
            mtner_port=args.mtner_port,
            gnormplus_home=args.gnormplus_home,
            gnormplus_port=args.gnormplus_port,
            tmvar2_home=args.tmvar2_home,
            tmvar2_port=args.tmvar2_port,
            gene_norm_port=args.gene_norm_port,
            disease_norm_port=args.disease_norm_port,
            cache_host=args.cache_host,
            cache_port=args.cache_port,
            use_neural_normalizer=args.use_neural_normalizer,
            keep_files=args.keep_files,
            no_cuda=args.no_cuda,
        )
    
    r_parser = ResultParser()

    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html', debug=False)
    
    @app.route('/documentation', methods=['GET'])
    def doc_view():
        return render_template('documentation.html')
    
    @app.route('/debug', methods=['GET'])
    def debug():
        return render_template('index.html', debug=True)
    
    @app.route('/pubmed/<pmids>', methods=['GET'])
    def pubmed_api(pmids):
        pmids = [pmid.strip() for pmid in pmids.split(",")]
        if len(pmids) == 0:
            return "[]"
        
        result_dicts = [model.annotate_pmid(pmid=pmid) for pmid in pmids]
        for r in result_dicts:
            if "error_code" in r.keys():
                if int(r["error_code"]) != 0:
                    return Response(json.dumps({"error_message": r["error_message"]}), status=404, content_type='application/json')
            del_keys_from_dict(r, ["sourcedb", "sourceid", "project", "elapse_time", "error_code", "error_message"])
        return json.dumps(result_dicts, sort_keys=True)
    
    @app.route('/plain', methods=['POST'])
    def plain_api():
        params = request.get_json()
        sample_text = params['text']

        # annotate input
        result_dict = model.annotate_text(text=sample_text)
        if "error_code" in result_dict.keys():
            if int(result_dict["error_code"]) != 0:
                return Response(json.dumps({"error_message": result_dict["error_message"]}), status=404, content_type='application/json')
        del_keys_from_dict(result_dict, ["sourcedb", "sourceid", "project", "elapse_time", "error_code", "error_message"])

        return json.dumps(result_dict, sort_keys=True)

    
    
    @app.route('/senddata', methods=['POST'])
    def send_data():
        start = time.time()

        res_items = []
        draw_keys = json.loads(request.form['draw_keys'])
        req_type = request.form['req_type']
        # is_neural_normalized = (request.form['use_neural'] == 'true')
        # print(is_neural_normalized)
        _debug = False
        if 'debug' in request.form:
            if request.form['debug'] == 'True':
                _debug = True
        
        # print("DEBUG:", _debug)

        if req_type == "text":
            sample_text = request.form['sample_text']
            # parse from BERN2 Model
            if not args.front_dev:
                result_dict = model.annotate_text(text=sample_text)
            else:
                dummy_path = os.path.join(app.root_path, "temp/dummy1_20211129.json")
                with open(dummy_path, 'r') as rf:
                    result_dict = json.load(rf)
            _code, parse_res, keys_in_dict = r_parser.parse_result(result_dict, draw_keys, result_id="text")

            if not _debug:
                del_keys_from_dict(result_dict, ["sourcedb", "sourceid", "project", "elapse_time", "error_code", "error_message"])

            latency = time.time() - start

            res_items.append({
                'parsed_response': parse_res,
                'keys': {k: r_parser.entity_type_dict[k] for k in keys_in_dict}
            })

            return render_template('result_text.html', result_items=res_items, latency = f'{latency*1000:9.2f}', result_str=json.dumps(result_dict, sort_keys=True, indent=4))
        elif req_type == "pmid":
            sample_pmid = request.form['sample_text']
            _pmids = list(map(str.strip, sample_pmid.split(",")))

            if not args.front_dev:
                result_dicts = [model.annotate_pmid(pmid=pmid) for pmid in _pmids]
            else:
                dummy_path = os.path.join(app.root_path, "temp/dummy2_20111129.json")
                with open(dummy_path, 'r') as rf:
                    result_dicts = json.load(rf)
                    
            pmid2result_dicts = {f"{result_dict['pmid']}_{i}":result_dict for i, result_dict in enumerate(result_dicts)}

            for _pmid, result_dict in pmid2result_dicts.items():
                _code, parse_res, keys_in_dict = r_parser.parse_result(result_dict, draw_keys, result_id=_pmid)

                if _code == "error":
                    # TODO: logging ERROR case
                    print("ERROR PMID:", _pmid)
                
                if not _debug:
                    del_keys_from_dict(result_dict, ["sourcedb", "sourceid", "project", "elapse_time", "error_code", "error_message"])

                legend_items = {k: r_parser.entity_type_dict[k] for k in keys_in_dict}

                res_item = {
                    'parsed_response': parse_res,
                    'keys': legend_items
                }
                if 'pmid' in result_dict.keys():
                    res_item['title'] = result_dict['pmid']

                res_items.append(res_item)
            
            
            latency = time.time() - start

            return render_template('result_text.html', result_items=res_items, latency = f'{latency*1000:9.2f}', result_str=json.dumps(result_dicts, sort_keys=True, indent=4))

    return app
