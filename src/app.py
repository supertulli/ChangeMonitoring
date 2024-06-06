import io
import json
from operator import index
import os

from datetime import date

import streamlit as st
import streamlit_ext as ste
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from icdmappings import Mapper, Validator

from inout.load_omop import OMOP_data
from data_handler import period_freq_tables
from igt import get_dist_matrix, get_IGT_embeddings, get_2D_IGT_plot, get_3D_IGT_plot, multiprocessing_get_dist_matrix

#load .env variables
CSV_FOLDER_PATH = os.getenv('CSV_FOLDER_PATH')
TABLE_STRUCTURE_JSON = os.getenv('TABLE_STRUCTURE_JSON')
ICD10_CHAPTER_DESCRIPTION = os.getenv('ICD10_CHAPTER_DESCRIPTION')
HLUZ_DATA = os.getenv('HLUZ_data')
ICD_VERSION = os.getenv('ICD_VERSION') # either 'icd9' or 'icd10'
CCR_VERSION = os.getenv('CCR_VERSION') # either 'ccs' or 'ccsr'
ATC_STRUCTURED_CODES = os.getenv('ATC_STRUCTURED_CODES')
ATC_CODES_DESCRIPTIONS = os.getenv('ATC_CODES_DESCRIPTIONS')

HLuz_data = (HLUZ_DATA == 'True')

@st.cache_data(show_spinner="Loading csv data to memory pandas database...")
def load_pandas_db(csv_folder_path:str = CSV_FOLDER_PATH, tables_structure_json :dict = TABLE_STRUCTURE_JSON) -> OMOP_data:
     with open(tables_structure_json, 'r') as f:
          tables_structure = json.load(f)
     omop_db = OMOP_data(csv_data_path=csv_folder_path, tables_structure=tables_structure)
     return omop_db

@st.cache_data
def get_visit_conditions_df(omop_db:OMOP_data) -> pd.DataFrame:
     visit_condition_df = omop_db.clinical_tables.condition_occurrence[['condition_source_value','visit_occurrence_id']]\
          .join(
               omop_db.clinical_tables.visit_occurrence[['visit_occurrence_id','visit_start_date']]\
          .set_index('visit_occurrence_id'),
               on='visit_occurrence_id',
               rsuffix='_visit'
          )
          
     ### HLUZ data adaptation
     def mutate_icd_code(original_code, validator=validator):
          code = original_code
          code = original_code.replace('.','')
          if validator.validate(code, expects='icd9_diagnostic'):
               return code
          
          for suffix in ['00','0','1']:
               suff_code = code + suffix
               if validator.validate(suff_code, expects='icd9_diagnostic'):
                    return suff_code
          
          return original_code

     if HLuz_data:
          visit_condition_df['condition_source_value']=visit_condition_df['condition_source_value'].apply(mutate_icd_code)
     
     visit_condition_df[f'{CCR_VERSION}_code'] = visit_condition_df['condition_source_value'].apply(lambda x: mapper.map(x.strip(), 
                                                                                                              source=ICD_VERSION,# 'icd10', 
                                                                                                              target=CCR_VERSION,# 'ccsr'
                                                                                                                   )
                                                                                                         ) # type: ignore
     visit_condition_df[f'{ICD_VERSION}_chapters'] = visit_condition_df['condition_source_value'].apply(lambda x: mapper.map(x.strip(), 
                                                                                                                   source=ICD_VERSION, #'icd10', 
                                                                                                                   target='chapter'
                                                                                                                   )
                                                                                                    ) # type: ignore

     return visit_condition_df[visit_condition_df[f'{CCR_VERSION}_code'].notnull() & visit_condition_df[f'{ICD_VERSION}_chapters'].notnull()]

### END : HLUZ data adaptation ###################

@st.cache_data
def get_chapter_df(visit_condition_df:pd.DataFrame, selected_chapter_code:str) -> pd.DataFrame:
     return visit_condition_df[visit_condition_df[f'{ICD_VERSION}_chapters'] == selected_chapter_code].drop([f'{ICD_VERSION}_chapters'], axis=1)

@st.cache_data
def get_visit_drugs_df(omop_db:OMOP_data) -> pd.DataFrame:
     visit_drug_df = omop_db.clinical_tables.drug_exposure[['drug_source_value','visit_occurrence_id']]\
          .join(
               omop_db.clinical_tables.visit_occurrence[['visit_occurrence_id','visit_start_date']]\
          .set_index('visit_occurrence_id'),
               on='visit_occurrence_id',
               rsuffix='_visit'
          )
     
     return visit_drug_df[visit_drug_df['drug_source_value'].notnull()]

@st.cache_data
def get_atc_level_df(visit_drug_df:pd.DataFrame,atc_parent_code:str|None, atc_child_level:int) -> pd.DataFrame:
     slice_dict = {
          1: slice(0,1), 2: slice(0,3), 3: slice(0,4), 4: slice(0,5), 5: slice(0,7)
     }
     if atc_parent_code:
          visit_drug_df['ATC_parent_level'] = visit_drug_df['drug_source_value'].apply(lambda x: x[slice_dict[atc_child_level-1]])
     visit_drug_df['ATC_current_level'] = visit_drug_df['drug_source_value'].apply(lambda x: x[slice_dict[atc_child_level]])
     
     return visit_drug_df[visit_drug_df['ATC_parent_level'] == atc_parent_code].drop(['ATC_parent_level'], axis=1) if atc_parent_code else visit_drug_df
          
     

@st.cache_data
def get_diagnoses_period_freq_tables(chapter_df:pd.DataFrame, period_string:str) -> tuple[pd.DataFrame, pd.DataFrame]:
     return period_freq_tables(chapter_df, f'{CCR_VERSION}_code', period_string)

def get_prescriptions_period_freq_tables(chapter_df:pd.DataFrame, period_string:str) -> tuple[pd.DataFrame, pd.DataFrame]:
     return period_freq_tables(chapter_df, 'ATC_current_level', period_string)

# @st.cache_data(show_spinner="Calculating IGT embeddings...")
@st.cache_data
def igt_embbedings(source_df:pd.DataFrame, multiprocessing:bool = False) -> np.ndarray:
     
     dist_matrix = multiprocessing_get_dist_matrix(source_df) if multiprocessing else get_dist_matrix(source_df)
     igt_embeddings = get_IGT_embeddings(dist_matrix)
     return igt_embeddings
     
st.set_page_config(page_title="Temporal Change Detection", layout="wide")

st.sidebar.title("Temporal Change Detection")

omop_db = load_pandas_db()

data_domain = st.sidebar.radio('Select analysis domain:',
                         ("Diagnostics", "Prescriptions")
                              )
if data_domain == "Diagnostics":
     mapper = Mapper()
     validator = Validator() 
     visit_condition_df = get_visit_conditions_df(omop_db)
     # with open(ICD10_CHAPTER_DESCRIPTION, 'r') as fp:
     with open(f'./data/mappings/{ICD_VERSION}_chapter_description.json', 'r') as fp:
          ICD_chapters_mapping = json.load(fp)
          
     chapter_selection = st.sidebar.selectbox(f'Select {ICD_VERSION.upper()} chapter',
                                             [f'{key}: {value}' for key, value in ICD_chapters_mapping.items() if key in visit_condition_df[f'{ICD_VERSION}_chapters'].unique()])
     selected_chapter_code = chapter_selection.split()[0][:-1]
     structured_df = get_chapter_df(visit_condition_df, selected_chapter_code)

if data_domain == "Prescriptions":
     visit_drugs_df = get_visit_drugs_df(omop_db)
     with open(ATC_STRUCTURED_CODES, 'r') as fp:
          ATC_structured_codes = json.load(fp)
     with open(ATC_CODES_DESCRIPTIONS, 'r') as fp:
          ATC_codes_descriptions = json.load(fp)
     structured_df = get_atc_level_df(visit_drugs_df, None, 1)
     parent_atc_code = None
     level_1_atc = st.sidebar.selectbox('Select level 1 ATC code', [None]+[f'{key}: {value}' 
                                                                           for key, value in ATC_codes_descriptions.get('level_1').items() 
                                                                           if key in structured_df['ATC_current_level'].unique()])
     atc_child_level = 'level_1'
     if level_1_atc:
          parent_atc_code = level_1_atc.split()[0][:-1]
          structured_df = get_atc_level_df(visit_drugs_df, parent_atc_code, 2)
          children_codes = ATC_structured_codes['level_1'].get(parent_atc_code)
          level_2_atc = st.sidebar.selectbox('Select level 2 ATC code', 
                                             [None]+[f'{key}: {value}' 
                                                       for key, value in ATC_codes_descriptions.get('level_2').items() 
                                                       if key in structured_df['ATC_current_level'].unique() and key in children_codes
                                                       ]
                                             )
          atc_child_level = 'level_2'
          if level_2_atc:
               parent_atc_code = level_2_atc.split()[0][:-1]
               structured_df = get_atc_level_df(visit_drugs_df, parent_atc_code, 3)
               children_codes = ATC_structured_codes['level_2'].get(parent_atc_code)
               level_3_atc = st.sidebar.selectbox('Select level 3 ATC code',
                                                  [None]+[f'{key}: {value}' 
                                                            for key, value in ATC_codes_descriptions.get('level_3').items() 
                                                            if key in structured_df['ATC_current_level'].unique() and key in children_codes
                                                            ]
                                                  )
               atc_child_level = 'level_3'
               if level_3_atc:
                    parent_atc_code = level_3_atc.split()[0][:-1]
                    structured_df = get_atc_level_df(visit_drugs_df, parent_atc_code, 4)
                    children_codes = ATC_structured_codes['level_3'].get(parent_atc_code)
                    level_4_atc = st.sidebar.selectbox('Select level 4 ATC code',
                                                       [None]+[f'{key}: {value}'
                                                                 for key, value in ATC_codes_descriptions.get('level_4').items()
                                                                 if key in structured_df['ATC_current_level'].unique() and key in children_codes
                                                                 ]
                                                       )
                    atc_child_level = 'level_4'
                    if level_4_atc:
                         parent_atc_code = level_4_atc.split()[0][:-1]
                         structured_df = get_atc_level_df(visit_drugs_df, parent_atc_code, 5)
                         children_codes = ATC_structured_codes['level_4'].get(parent_atc_code)
                         atc_child_level = 'level_5'
     
     
period_string = st.sidebar.radio('Select period:', ('Yearly', 'Monthly', 'Weekly', 'Daily'), index=1)
period = {'Yearly':'YE', 'Monthly':'ME', 'Weekly':'W', 'Daily':'D'}[period_string]

if data_domain == "Diagnostics":
     ccsr_abs_freq, ccsr_rel_freq = get_diagnoses_period_freq_tables(structured_df, period)
elif data_domain == "Prescriptions":
     ccsr_abs_freq, ccsr_rel_freq = get_prescriptions_period_freq_tables(structured_df, period)


freq_type = st.sidebar.radio('Absolute or relative frequency:', ('absolute', 'relative'), index=1)
st.sidebar.write("Note: to visualize an IGT plot, you must select relative frequency.")

if freq_type == 'absolute':
     active_df = ccsr_abs_freq.rename(columns={'visit_start_date':'date'})
else:
     active_df = ccsr_rel_freq.rename(columns={'visit_start_date':'date'})
     
min_date, max_date = st.sidebar.slider("Set date boundaries: ", value = (active_df['date'].min().to_pydatetime(), active_df['date'].max().to_pydatetime()) )

active_df = active_df[(pd.to_datetime(active_df['date']) >= min_date) & (pd.to_datetime(active_df['date']) <= max_date)]

fig, ax = plt.subplots(figsize=(20,8), )
plot_df = active_df.drop('date', axis=1).T

if data_domain == "Diagnostics":
     plt.title(f'{ICD_VERSION.upper()} - Chapter {selected_chapter_code}')
elif data_domain == "Prescriptions":
     plt.title(f'ATC {parent_atc_code} selected_chapter_code - one level bellow prescriptions')
sns.heatmap(plot_df.sort_index(ascending=True), ax=ax)    

st.pyplot(fig)

save_heatmap_filename = "heatmap.png"
heatmap_img = io.BytesIO()
fig.savefig(heatmap_img, format='png')

btn = ste.download_button(
     label="save heatmap",
     data=heatmap_img,
     file_name=save_heatmap_filename,
     mime='image/png'
)

# Legend
if data_domain == "Diagnostics":
     with open(f'./data/mappings/{CCR_VERSION}_descriptions.json', 'r') as fp: # this not kosher, should be refactored
          code_descriptions_dict = json.load(fp)
          
     with st.expander(f"{CCR_VERSION.upper()} Codes' descriptions:"):
          for code in active_df.drop('date',axis=1).columns:
               st.write(f"{code}: {code_descriptions_dict.get(code, 'No description available.')}")   
     # st.write("".join([f"{code}: {code_descriptions_dict[code]}. \n" for code in active_df.drop('date',axis=1).columns]))

if data_domain == "Prescriptions":
     with open(ATC_CODES_DESCRIPTIONS, 'r') as fp:
          ATC_codes_descriptions = json.load(fp)

     with st.expander(f"ATC {atc_child_level} Codes' descriptions:"):
          for code in active_df.drop('date',axis=1).columns:
               st.write(f"{code}: {ATC_codes_descriptions.get(atc_child_level).get(code, 'No description available.')}")

# IGT plot

if freq_type == 'relative':

     with st.sidebar.form("igt_plot"):
          st.markdown("# IGT plot")
          st.caption("""
               Important: note that this can take a significant amount of time to render if the amount of periods to compare is large, 
               and using multiprocessing might reduce the time to render the plot. Conversely if the amount of periods to compare is small, 
               the time overhead to spawn and manage all processes might might be significant and delay the rendering.
               """)
          multiprocessing = st.checkbox("Use multiprocessing", value=False)
          plt_type = st.radio("Select plot type", ('2D', '3D'), index=0)
          point_labels = st.checkbox("Show point labels", value=True)
          submit = st.form_submit_button("Render IGT plot")
     
     if submit:
          with st.spinner("Rendering IGT plot..."):
               igt_embbedings = igt_embbedings(active_df.drop('date', axis=1), multiprocessing)
               if period_string == 'Yearly':
                    point_labels = active_df.index.to_flat_index().to_numpy() if point_labels else None
               else:
                    point_labels = ['-'.join([str(i) for i in row]) for row in active_df.index.to_flat_index().to_numpy()] if point_labels else None
               if plt_type == '3D':   
                    fig = get_3D_IGT_plot(igt_embbedings, point_labels)
               else:
                    fig = get_2D_IGT_plot(igt_embbedings, point_labels)
               st.pyplot(fig)
               save_igt_filename = "igt.png"
               igt_img = io.BytesIO()
               fig.savefig(igt_img, format='png')

               ste.download_button(
                    label="save IGT plot",
                    data=igt_img,
                    file_name=save_igt_filename,
                    mime='image/png'
               )
               
st.dataframe(active_df.drop('date', axis=1)) # Just to show the dataframe, will be omitted in production