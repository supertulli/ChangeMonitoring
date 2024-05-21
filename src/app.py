import io
import json
import os

from datetime import date

import streamlit as st
import streamlit_ext as ste
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from icdmappings import Mapper

from inout.load_omop import OMOP_data
from data_handler import period_freq_tables
from igt import get_dist_matrix, get_IGT_embeddings, get_2D_IGT_plot, get_3D_IGT_plot, multiprocessing_get_dist_matrix

#load .env variables
CSV_FOLDER_PATH = os.getenv('CSV_FOLDER_PATH')
TABLE_STRUCTURE_JSON = os.getenv('TABLE_STRUCTURE_JSON')
ICD10_CHAPTER_DESCRIPTION = os.getenv('ICD10_CHAPTER_DESCRIPTION')

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

     visit_condition_df['ccsr_code'] = visit_condition_df['condition_source_value'].apply(lambda x: mapper.map(x.strip(), 
                                                                                                              source='icd10', 
                                                                                                              target='ccsr'))
     visit_condition_df['icd10_chapters'] = visit_condition_df['condition_source_value'].apply(lambda x: mapper.map(x.strip(), 
                                                                                                                   source='icd10', 
                                                                                                                   target='chapter'))
     
     return visit_condition_df[visit_condition_df['ccsr_code'].notnull() & visit_condition_df['icd10_chapters'].notnull()]

@st.cache_data
def get_chapter_df(visit_condition_df:pd.DataFrame, selected_chapter_code:str) -> pd.DataFrame:
     return visit_condition_df[visit_condition_df['icd10_chapters'] == selected_chapter_code].drop(['icd10_chapters'], axis=1)

@st.cache_data
def get_period_freq_tables(chapter_df:pd.DataFrame, period_string) -> tuple[pd.DataFrame, pd.DataFrame]:
     return period_freq_tables(chapter_df, 'ccsr_code', period_string)

# @st.cache_data(show_spinner="Calculating IGT embeddings...")
@st.cache_data
def igt_embbedings(source_df:pd.DataFrame, multiprocessing:bool = False) -> np.ndarray:
     
     dist_matrix = multiprocessing_get_dist_matrix(source_df) if multiprocessing else get_dist_matrix(source_df)
     igt_embeddings = get_IGT_embeddings(dist_matrix)
     return igt_embeddings
     
st.set_page_config(page_title="Temporal Change Detection", layout="wide")

st.sidebar.title("Temporal Change Detection")

omop_db = load_pandas_db()
mapper = Mapper()
visit_condition_df = get_visit_conditions_df(omop_db)
with open(ICD10_CHAPTER_DESCRIPTION, 'r') as fp:
     ICD10_chapters_mapping = json.load(fp)
     
chapter_selection = st.sidebar.selectbox('Select ICD10 Chapter',[f'{key}: {value}' for key, value in ICD10_chapters_mapping.items() if key in visit_condition_df['icd10_chapters'].unique()])
selected_chapter_code = chapter_selection.split()[0][:-1]
structured_df = get_chapter_df(visit_condition_df, selected_chapter_code)

period_string = st.sidebar.radio('Select period:', ('Yearly', 'Monthly', 'Weekly', 'Daily'), index=1)
period = {'Yearly':'YE', 'Monthly':'ME', 'Weekly':'W', 'Daily':'D'}[period_string]
ccsr_abs_freq, ccsr_rel_freq = get_period_freq_tables(structured_df, period)

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
plt.title(f'ICD-10 Chapter {selected_chapter_code}')

sns.heatmap(plot_df.sort_index(ascending=False), ax=ax)    

st.pyplot(fig)

save_heatmap_filename = "heatmap.png"
heatmap_img = io.BytesIO()
fig.savefig(heatmap_img, format='png')

btn = ste.download_button(
     label="download heatmap",
     data=heatmap_img,
     file_name=save_heatmap_filename,
     mime='image/png'
)

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
                    label="download IGT",
                    data=igt_img,
                    file_name=save_igt_filename,
                    mime='image/png'
               )
               
st.dataframe(active_df.drop('date', axis=1)) # Just to show the dataframe, will be omitted in production