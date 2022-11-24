import pandas as pd

from bern2.run_bern2 import get_initialized_bern, run_bern2_annotation

if __name__ == '__main__':

    tagging_file_path = "/Volumes/GoogleDrive/Shared drives/Navina/Data Science/Diagnoses Entity Recognition/" \
                        "tagging_results/summary_both_2703_after_fixing.csv"

    bern2 = get_initialized_bern()
    gt_df = pd.read_csv(tagging_file_path)
    text = []
    for id in gt_df.doc_id.unique():
        doc = gt_df.query(f"doc_id == '{id}'")
        text_series = doc.fillna('').apply(lambda row: row.text + (' ' * row.ws), axis=1)
        cur_text = ''.join(text_series).replace('\ub200', '')
        text.append(cur_text)

    results = bern2.annotate_text(text[:2])

    # results = []
    # for cur_text in text:
    #     result = bern2.annotate_text(cur_text)
    #     results.append(result)
    res_df = pd.DataFrame(results)
    print(results)
