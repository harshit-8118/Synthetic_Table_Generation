import sdv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sdv.evaluation.single_table import QualityReport, DiagnosticReport, evaluate_quality, get_column_pair_plot, get_column_plot, run_diagnostic
from sdv.single_table import GaussianCopulaSynthesizer, CopulaGANSynthesizer, CTGANSynthesizer, TVAESynthesizer
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

real_data = ""
synthetic_generated_data = ""
synthetic_metadata = ""

def parse_real_table(table_path):
    global real_data
    data = pd.read_csv(table_path)
    data = data.iloc[:1000, 1:]
    real_data = data
    data = data.to_dict(orient='records')
    return data[:5]


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
    if table_type == 'single_table':
        table_path = f'SDV_single_table_demos\{table_name}.csv'
        return parse_real_table(table_path)
    else:
        table_path = f'SDV_multi_table_demos\{table_name}'
        tables = os.listdir(table_path)
        return parse_real_tables(table_path, tables)



def parse_synthetic_table(syn_data):
    syn_data = syn_data.iloc[:5, :]
    syn_data = syn_data.to_dict(orient='records')
    return syn_data

# show synthetic generated table
def get_synthetic_table(table_type, table_name, num_rows=1000):
    global synthetic_generated_data, synthetic_metadata

    if table_type == 'single_table':
        trf_Gaussian_path = f'SDV_trained_file_single_table\{table_name}\{table_name}_Gaussian.pkl'
        trf_CopulaGan_path = f'SDV_trained_file_single_table\{table_name}\{table_name}_CopulaGan.pkl'
        trf_CTGan_path = f'SDV_trained_file_single_table\{table_name}\{table_name}_CTGan.pkl'
        trf_TVAE_path = f'SDV_trained_file_single_table\{table_name}\{table_name}_TVAE.pkl'

        clf1 = GaussianCopulaSynthesizer.load(filepath=trf_Gaussian_path)
        clf1_data = clf1.sample(num_rows)
        
        clf2 = CopulaGANSynthesizer.load(trf_CopulaGan_path)
        clf2_data = clf2.sample(num_rows)

        clf3 = GaussianCopulaSynthesizer.load(filepath=trf_CTGan_path)
        clf3_data = clf3.sample(num_rows)
        
        clf4 = CopulaGANSynthesizer.load(trf_TVAE_path)
        clf4_data = clf4.sample(num_rows)

        data = {
            'clf1': {
                'metadata': clf1.get_metadata(),
                'data' : clf1_data          
            },
            'clf2': {
                'metadata': clf2.get_metadata(),
                'data' : clf2_data          
            },
            'clf3': {
                'metadata': clf3.get_metadata(),
                'data' : clf3_data          
            },
            'clf4': {
                'metadata': clf4.get_metadata(),
                'data' : clf4_data          
            },
        }
        score = 0.0
        for k, v in data.items():
            diag_report = run_diagnostic(real_data=real_data, synthetic_data=v['data'], metadata=v['metadata'])
            if score < diag_report.get_score():
                score = diag_report.get_score() * 100
                synthetic_generated_data = v['data']
                synthetic_metadata = v['metadata']


        clf_data = parse_synthetic_table(synthetic_generated_data)
        return clf_data, synthetic_metadata

# show diagnostic reports
def get_diag_reports(table_type, table_name):
    if table_type == 'single_table':
        diagnostic_report = run_diagnostic(real_data, synthetic_generated_data, synthetic_metadata)
        score = diagnostic_report.get_score() * 100
        diagnostic_report = diagnostic_report.get_info()
        diagnostic_report['score'] = score
        return diagnostic_report
    else:
        pass

# show evaluation reports
def get_eval_reports(table_type, table_name):
    if table_type == 'single_table':
        eval_report_Gaussian = f'SDV_reports_single_table\{table_name}\evaluation\{table_name}_Gaussian_eval.pkl'
        eval_report_CopulaGan = f'SDV_reports_single_table\{table_name}\evaluation\{table_name}_CopulaGan_eval.pkl'
        eval_report_CTGan = f'SDV_reports_single_table\{table_name}\evaluation\{table_name}_CTGan_eval.pkl'
        eval_report_TVAE = f'SDV_reports_single_table\{table_name}\evaluation\{table_name}_TVAE_eval.pkl'

        quality_report = QualityReport.load(eval_report_Gaussian)
        quality_report = quality_report.get_details(property_name='Column Shapes')
        return quality_report.to_dict(orient='records')
    else:
        pass

