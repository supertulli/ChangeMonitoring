import os
from pyhealth.datasets import OMOPDataset

data_root_path = "./data/mimic-iv-demo-data-in-the-omop-common-data-model-0.9/1_omop_data_csv"
omop_csv_file_list = [ f.rstrip('.csv') for f in os.listdir(data_root_path) if os.path.isfile(os.path.join(data_root_path,f))]

omop_ds=OMOPDataset(
    root=data_root_path,
    tables=[],#["person","visit_occurrence","death"], # omop_table_list,
    code_mapping={}
)