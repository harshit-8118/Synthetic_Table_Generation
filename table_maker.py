import sdv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sdv.evaluation.single_table import (
    QualityReport,
    DiagnosticReport,
    evaluate_quality,
    get_column_pair_plot,
    get_column_plot,
    run_diagnostic,
)
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
)
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

real_data = None
synthetic_generated_data = None
synthetic_metadata = None
selected_model = None
model_dict = {
    "clf1": "GaussianCopulaSynthesizer",
    "clf2": "CopulaGanSynthesizer",
    "clf3": "CTGanSynthesizer",
    "clf4": "TVAESynthesizer",
}


def parse_real_table(table_path):
    global real_data
    data = pd.read_csv(table_path)
    data = data.iloc[:, 1:]
    real_data = data
    data = data.iloc[:5, :]
    data = data.to_dict(orient="records")
    return data


def parse_real_tables(table_path, tables):
    global real_data
    data = {}
    for table in tables:
        table_content = parse_real_table(os.path.join(table_path, table))
        data[table] = table_content

    real_data = data
    return data


# show demo table
def get_demo_table(table_type, table_name):
    if table_type == "single_table":
        table_path = f"SDV_single_table_demos\{table_name}.csv"
        return parse_real_table(table_path)
    else:
        table_path = f"SDV_multi_table_demos\{table_name}"
        tables = os.listdir(table_path)
        return parse_real_tables(table_path, tables)


def parse_synthetic_table(syn_data):
    syn_data = syn_data.iloc[:5, :]
    syn_data = syn_data.to_dict(orient="records")
    return syn_data


# show synthetic generated table
def get_synthetic_table(table_type, table_name, num_rows=10000):
    global synthetic_generated_data, synthetic_metadata, selected_model

    if table_type == "single_table":
        trf_Gaussian_path = (
            f"SDV_trained_file_single_table\{table_name}\{table_name}_Gaussian.pkl"
        )
        trf_CopulaGan_path = (
            f"SDV_trained_file_single_table\{table_name}\{table_name}_CopulaGan.pkl"
        )
        trf_CTGan_path = (
            f"SDV_trained_file_single_table\{table_name}\{table_name}_CTGan.pkl"
        )
        trf_TVAE_path = (
            f"SDV_trained_file_single_table\{table_name}\{table_name}_TVAE.pkl"
        )

        data = {}
        logs = []

        try:
            clf1 = GaussianCopulaSynthesizer.load(filepath=trf_Gaussian_path)
            clf1_data = clf1.sample(num_rows)
            data["clf1"] = {"metadata": clf1.get_metadata(), "data": clf1_data}
        except:
            pass

        try:
            clf2 = CopulaGANSynthesizer.load(trf_CopulaGan_path)
            clf2_data = clf2.sample(num_rows)
            data["clf2"] = {"metadata": clf2.get_metadata(), "data": clf2_data}
        except:
            pass

        try:
            clf3 = CTGANSynthesizer.load(filepath=trf_CTGan_path)
            clf3_data = clf3.sample(num_rows)
            data["clf3"] = {"metadata": clf3.get_metadata(), "data": clf3_data}
        except:
            pass

        try:
            clf4 = TVAESynthesizer.load(trf_TVAE_path)
            clf4_data = clf4.sample(num_rows)
            data["clf4"] = {"metadata": clf4.get_metadata(), "data": clf4_data}
        except:
            pass
        
        score = 0.0
        for k, v in data.items():
            print(k)
            try:
                quality_report = evaluate_quality(
                    real_data=real_data,
                    synthetic_data=v["data"],
                    metadata=v["metadata"],
                    verbose=False
                )
                details = quality_report.get_details(property_name="Column Shapes")
                details_dict = details.to_dict(orient="records")
                details_dict.append(
                    {"Score": quality_report.get_score(), "Synthesizer": model_dict[k]}
                )
                logs.append(details_dict)

                if score < quality_report.get_score():
                    score = quality_report.get_score()
                    synthetic_generated_data = v["data"]
                    synthetic_metadata = v["metadata"]
                    selected_model = k
            except:
                pass
                
        clf_data = parse_synthetic_table(synthetic_generated_data)
        eval_report = get_eval_reports(table_type, table_name, data)
        return clf_data, synthetic_metadata, selected_model, logs, eval_report


# show diagnostic reports
def get_diag_reports(table_type, table_name):
    if table_type == "single_table":
        diagnostic_report = run_diagnostic(
            real_data, synthetic_generated_data, synthetic_metadata
        )
        score = diagnostic_report.get_score() * 100
        diagnostic_report = diagnostic_report.get_info()
        diagnostic_report["score"] = score
        return diagnostic_report
    else:
        pass


# show evaluation reports
def get_eval_reports(table_type, table_name, data):
    if table_type == "single_table":
        quality_report = evaluate_quality(real_data=real_data, synthetic_data=synthetic_generated_data, metadata=synthetic_metadata)
        details = quality_report.get_details(property_name="Column Shapes")

        return details.to_dict(orient="records")
    else:
        pass
