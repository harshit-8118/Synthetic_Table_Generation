import pandas as pd
from sdv.evaluation.single_table import (
    # QualityReport,
    # DiagnosticReport,
    # get_column_pair_plot,
    evaluate_quality,
    get_column_plot,
    run_diagnostic,
)
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
)
from sdv.multi_table import HMASynthesizer
import os
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

"""
parsing real single table
function->parse_real_table(table_path)
table: table_name.csv
"""


def parse_real_table(table_path):
    global real_data
    data = pd.read_csv(table_path)
    data = data.iloc[:, 1:]
    real_data = data
    data = data.iloc[:5, :]
    data = data.to_dict(orient="records")
    return data


"""
parsing multitable
function->parse_real_tables(table_path, tables)
table: table_name.csv
tables: [table_name1, table_name2 ... table_name3]
"""


def parse_real_tables(table_path, tables):
    global real_data
    data = {}
    org_data = {}
    for table in tables:
        tab_path = os.path.join(table_path, table)
        df = pd.read_csv(tab_path)
        df = df.iloc[:, 1:]
        org_tab = df
        df = df.iloc[:5, :]
        df = df.to_dict(orient="records")
        org_data[table] = org_tab
        data[table] = df

    real_data = org_data
    return data


"""
get_demo_table(table_type, table_name)
table_type: single_table | multi_table
table_name: table_name
"""


def get_demo_table(table_type, table_name):
    if table_type == "single_table":
        table_path = f"SDV_single_table_demos\{table_name}.csv"
        return parse_real_table(table_path)
    else:
        table_path = f"SDV_multi_table_demos\{table_name}"
        tables = os.listdir(table_path)
        return parse_real_tables(table_path, tables)


"""
function->parse_synthetic_table(syn_data)
slicing top five synthetic records to show
"""


def parse_synthetic_table(syn_data):
    syn_data = syn_data.iloc[:5, :]
    syn_data = syn_data.to_dict(orient="records")
    return syn_data


"""
function->parse_synthetic_multi_table(syn_data)
slicing top five synthetic records from multiple table to show
"""


def parse_synthetic_multi_table(syn_data):
    for k, v in syn_data.items():
        dv = v.iloc[:5, :]
        syn_data[k] = dv.to_dict(orient="records")
    return syn_data


"""
get_synthetic_table(table_type, table_name, num_rows=1000):
table_type: single_table | multi_table
table_name: table_name
num_rows: amount of records to be obtained
working: 
    for single_table:
        read        
    for multi_table: 
        read
"""


def get_synthetic_table(table_type, table_name, num_rows=1000):
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
        for k, v in data.items():  # evaluating report for different model
            print(k)
            try:
                quality_report = evaluate_quality(
                    real_data=real_data,
                    synthetic_data=v["data"],
                    metadata=v["metadata"],
                    verbose=False,
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
        eval_report = get_eval_reports(table_type, table_name)
        return clf_data, synthetic_metadata, selected_model, logs, eval_report
    else:
        trf_HMA_path = f"SDV_trained_file_multi_table\{table_name}.pkl"

        data = {}

        try:
            clf_hma = HMASynthesizer.load(trf_HMA_path)
            clf_data = clf_hma.sample(scale=0.6)
            data["clf_hma"] = {"metadata": clf_hma.get_metadata(), "data": clf_data}
        except:
            pass

        print("clf_hma")
        synthetic_generated_data = data["clf_hma"]["data"]
        synthetic_metadata = data["clf_hma"]["metadata"]

        clf_data = parse_synthetic_multi_table(synthetic_generated_data)
        return clf_data, synthetic_metadata


# show diagnostic reports
def get_diag_reports(table_type, table_name):
    if table_type == "single_table":
        diagnostic_report = run_diagnostic(
            real_data, synthetic_generated_data, synthetic_metadata, verbose=False
        )
        diagnostic_report = diagnostic_report.get_info()
        return diagnostic_report
    else:
        pass


# show evaluation reports
def get_eval_reports(table_type, table_name):
    global real_data, synthetic_generated_data, synthetic_metadata
    if table_type == "single_table":
        quality_report = evaluate_quality(
            real_data=real_data,
            synthetic_data=synthetic_generated_data,
            metadata=synthetic_metadata,
            verbose=False,
        )
        details = quality_report.get_details(property_name="Column Shapes")
        score = quality_report.get_score()

        return details.to_dict(orient="records")
    else:
        pass


def get_evaluation_graphs(table_type, table_name):
    categories = ["boolean", "categorical", "datetime", "numerical"]
    graph_path = []
    if table_name in ["student_placements", "student_placements_pii"]:
        return None

    for k, v in synthetic_metadata.columns.items():
        if v["sdtype"] in categories:
            fig = get_column_plot(
                real_data=real_data,
                synthetic_data=synthetic_generated_data,
                metadata=synthetic_metadata,
                column_name=k,
            )
            # fig.show()
            graph_path.append(fig)

    return graph_path
