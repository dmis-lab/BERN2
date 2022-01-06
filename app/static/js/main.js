
$(document).ready(function(){
    $.ajaxSetup({
        timeout: 5000,
        error: function(xhr){
            var textarea = document.getElementById("test_textarea");
            $(textarea).animate({'height': '2.4rem'}, 100);
            // document.getElementById("demo_result").innerHTML = res;
            document.getElementById("demo_result").innerHTML = "ERROR: " + xhr.status + " " + xhr.statusText;
            initTooltips();
            $(".loader-box").hide();
        }
    });
    $("input[type=radio][name=select_input_type]").change(function(){
        var r_type = $(this).val();
        var textarea = document.getElementById("test_textarea");
        if(r_type == "text"){
            // textarea.placeholder = "Autophagy maintains tumour growth through circulating arginine. Autophagy captures intracellular components and delivers them to lysosomes, where they are degraded and recycled to sustain metabolism and to enable survival during starvation1-5. Acute, whole-body deletion of the essential autophagy gene Atg7 in adult mice causes a systemic metabolic defect that manifests as starvation intolerance and gradual loss of white adipose tissue, liver glycogen and muscle mass1. Cancer cells also benefit from autophagy. Deletion of essential autophagy genes impairs the metabolism, proliferation, survival and malignancy of spontaneous tumours in models of autochthonous cancer6,7. Acute, systemic deletion of Atg7 or acute, systemic expression of a dominant-negative ATG4b in mice induces greater regression of KRAS-driven cancers than does tumour-specific autophagy deletion, which suggests that host autophagy promotes tumour growth1,8. Here we show that host-specific deletion of Atg7 impairs the growth of multiple allografted tumours, although not all tumour lines were sensitive to host autophagy status. Loss of autophagy in the host was associated with a reduction in circulating arginine, and the sensitive tumour cell lines were arginine auxotrophs owing to the lack of expression of the enzyme argininosuccinate synthase 1. Serum proteomic analysis identified the arginine-degrading enzyme arginase I (ARG1) in the circulation of Atg7-deficient hosts, and in vivo arginine metabolic tracing demonstrated that serum arginine was degraded to ornithine. ARG1 is predominantly expressed in the liver and can be released from hepatocytes into the circulation. Liver-specific deletion of Atg7 produced circulating ARG1, and reduced both serum arginine and tumour growth. Deletion of Atg5 in the host similarly regulated [corrected] circulating arginine and suppressed tumorigenesis, which demonstrates that this phenotype is specific to autophagy function rather than to deletion of Atg7. Dietary supplementation of Atg7-deficient hosts with arginine partially restored levels of circulating arginine and tumour growth. Thus, defective autophagy in the host leads to the release of ARG1 from the liver and the degradation of circulating arginine, which is essential for tumour growth; this identifies a metabolic vulnerability of cancer. (PMID:30429607)";
            textarea.value = "Autophagy maintains tumour growth through circulating arginine. Autophagy captures intracellular components and delivers them to lysosomes, where they are degraded and recycled to sustain metabolism and to enable survival during starvation1-5. Acute, whole-body deletion of the essential autophagy gene Atg7 in adult mice causes a systemic metabolic defect that manifests as starvation intolerance and gradual loss of white adipose tissue, liver glycogen and muscle mass1. Cancer cells also benefit from autophagy.";
            $(textarea).animate({'height': '7rem'}, 100);
        }else{
            // textarea.placeholder = "29446767";
            textarea.value = "30429607,29446767";
            $(textarea).animate({'height': '2.4rem'}, 100);
        }
        checkTextCount($("#test_textarea"));
    });

    checkTextCount($("#test_textarea"));

    $("#test_textarea").on("input", function(){
        checkTextCount(this);
    });

    $("#test_textarea").keypress(function (e) {
        if(e.which === 13 && !e.shiftKey) {
            e.preventDefault();
            submit_sample();
        }
    });
});

var check_parents = function(obj, target_id){
    var obj_id = obj.getAttribute('id');
    if(obj_id != null){
        if(obj_id.includes('tooltip')){
            return true;
        }
    }
    while(obj.parentNode && obj.parentNode.nodeName.toLowerCase() != 'body'){
        obj = obj.parentNode;
        obj_id = obj.getAttribute('id');
        if(obj_id != null){
            if(obj_id.includes('tooltip')){
                return true;
            }
        }
    }
    return false;
}

var initUnderlines = function(){
    $("span.stack-spans").hover(function(){
        var classList = $(this).attr('class').split(/\s+/);
        $.each(classList, function(index, item){
            if(item != "stack-spans" && item != "text-decoration-underline"){
                $("." + item).addClass("text-decoration-underline");
            }
        });
    }, function(){
        var classList = $(this).attr('class').split(/\s+/);
        $.each(classList, function(index, item){
            if(item != "stack-spans" && item != "text-decoration-underline"){
                $("." + item).removeClass("text-decoration-underline");
            }
        });
    });
}

var initTooltips = function(){
    var spans = $('.stack-spans');
    var tooltips = $('.stack-tooltips');

    $(document).on('click touchend', function(e){
        var target = $(e.target);
        if(check_parents(e.target)){
            // Contains tooltip or is tooltip itself:
            // Do nothing
            return;
        }else{
            // Non-tooltip object
            if(target.attr('id') == null){
                // NULL
                // Close all the other tooltips
                tooltips.hide();
            }else{
                var target_id = target.attr('id');
                if(target_id.includes('span')){
                    // SPAN
                    // Close all the other tooltips
                    tooltips.hide();
                    // Show the tooltip of target
                    var tooltip_id = target_id.replace("span", "tooltip");
                    var target_span = document.getElementById(target_id);
                    var target_tooltip = document.getElementById(tooltip_id);
                    Popper.createPopper(target_span, target_tooltip, {
                        placement: 'top'
                    });
                    $(target_tooltip).show();

                }else{
                    // Close all the other tooltips
                    tooltips.hide();
                }
            }
        }
    });
    for(var i=0; i<spans.length; i++){
        Popper.createPopper(spans[i], tooltips[i], {
            placement: 'top',
        });
    }
}

var checkTextCount = function(obj){
    var req_types = document.getElementsByName("select_input_type");
    for(var i=0; i<req_types.length; i++){
        if(req_types[i].checked){
            req_type = req_types[i].value;
            break;
        }
    }

    span_class = ""

    if(req_type == "text"){
        var max_length = $(obj).attr("maxlength");
        var cur_text = $(obj).val();
        var cur_length = cur_text.length;

        if(cur_length >= max_length){
            cur_text = cur_text.substring(0, max_length);
            cur_length = cur_text.length;
            $(obj).val(cur_text);
        }

        document.getElementById("text_count").innerHTML = "<span class='" + span_class + "'>" + cur_length + "</span>/" + max_length + " characters";
    }else{
        var max_length = 5;
        var cur_text = $(obj).val();
        var pmids = cur_text.split(',');
        if(pmids.length > max_length){
            span_class = "text-danger";
        }
        document.getElementById("text_count").innerHTML = "<span class='" + span_class + "'>" + pmids.length + "</span>/" + max_length + " PMIDs (comma separated)";
    }
}

var onFocusArea = function(){
    var textarea = document.getElementById("test_textarea");
    var r_type = $("input[type=radio][name=select_input_type]:checked").val();
    // console.log(r_type);
    if(r_type == "text"){
        $(textarea).animate({'height': '7rem'}, 100);
    }else{
        $(textarea).animate({'height': '2.4rem'}, 100);
    }
    checkTextCount($("#test_textarea"));
}

var copy_json_to_clipboard = function(){
    var json_pre = document.getElementById("results_in_json");
    var json_pre_text = json_pre.textContent;

    var el = document.createElement("textarea");
    el.value = json_pre_text;
    el.setAttribute('readonly', '');
    el.style.position='absolute';
    el.style.left='-9999px';
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
}

var submit_sample = function(){
    var req_type = "text";
    var req_types = document.getElementsByName("select_input_type");
    for(var i=0; i<req_types.length; i++){
        if(req_types[i].checked){
            req_type = req_types[i].value;
            break;
        }
    }
    
    var textarea = document.getElementById("test_textarea");
    var txt = textarea.value;
    if(txt == ""){
        if(req_type == "text"){
            // txt = "Autophagy maintains tumour growth through circulating arginine. Autophagy captures intracellular components and delivers them to lysosomes, where they are degraded and recycled to sustain metabolism and to enable survival during starvation1-5. Acute, whole-body deletion of the essential autophagy gene Atg7 in adult mice causes a systemic metabolic defect that manifests as starvation intolerance and gradual loss of white adipose tissue, liver glycogen and muscle mass1. Cancer cells also benefit from autophagy. Deletion of essential autophagy genes impairs the metabolism, proliferation, survival and malignancy of spontaneous tumours in models of autochthonous cancer6,7. Acute, systemic deletion of Atg7 or acute, systemic expression of a dominant-negative ATG4b in mice induces greater regression of KRAS-driven cancers than does tumour-specific autophagy deletion, which suggests that host autophagy promotes tumour growth1,8. Here we show that host-specific deletion of Atg7 impairs the growth of multiple allografted tumours, although not all tumour lines were sensitive to host autophagy status. Loss of autophagy in the host was associated with a reduction in circulating arginine, and the sensitive tumour cell lines were arginine auxotrophs owing to the lack of expression of the enzyme argininosuccinate synthase 1. Serum proteomic analysis identified the arginine-degrading enzyme arginase I (ARG1) in the circulation of Atg7-deficient hosts, and in vivo arginine metabolic tracing demonstrated that serum arginine was degraded to ornithine. ARG1 is predominantly expressed in the liver and can be released from hepatocytes into the circulation. Liver-specific deletion of Atg7 produced circulating ARG1, and reduced both serum arginine and tumour growth. Deletion of Atg5 in the host similarly regulated [corrected] circulating arginine and suppressed tumorigenesis, which demonstrates that this phenotype is specific to autophagy function rather than to deletion of Atg7. Dietary supplementation of Atg7-deficient hosts with arginine partially restored levels of circulating arginine and tumour growth. Thus, defective autophagy in the host leads to the release of ARG1 from the liver and the degradation of circulating arginine, which is essential for tumour growth; this identifies a metabolic vulnerability of cancer. (PMID:30429607)";
            txt = "Autophagy maintains tumour growth through circulating arginine. Autophagy captures intracellular components and delivers them to lysosomes, where they are degraded and recycled to sustain metabolism and to enable survival during starvation1-5. Acute, whole-body deletion of the essential autophagy gene Atg7 in adult mice causes a systemic metabolic defect that manifests as starvation intolerance and gradual loss of white adipose tissue, liver glycogen and muscle mass1. Cancer cells also benefit from autophagy.";
        }else{
            txt = "30429607,29446767";
        }
        document.getElementById("test_textarea").value = txt;
    }

    var draw_keys = ['disease', 'mutation', 'gene', 'drug', 'species', 'DNA', 'RNA', 'cell_line', 'cell_type'];

    if(req_type == "pmid"){
        var pmids = txt.split(',');
        if(pmids.length > 5){
            return;
        }
    }
    
    document.getElementById("demo_result").innerHTML = "";
    $("#test_textarea").trigger('focusout');
    $("#test_textarea").trigger('blur');
    $(".loader-box").show();

    $.post('./senddata', {
        'sample_text': txt,
        'draw_keys': JSON.stringify(draw_keys),
        'req_type': req_type,
        'debug': "{{ debug }}"
    }, function(res){
        $(textarea).animate({'height': '2.4rem'}, 100);
        document.getElementById("demo_result").innerHTML = res;
        initTooltips();
        initUnderlines();
        $(".loader-box").hide();
    });
}