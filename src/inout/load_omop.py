# imports
from dataclasses import dataclass, field, fields, InitVar
import os

import pandas as pd
import dask.dataframe as dd 

# define a dataclass to hold the db in memory
# NOTE: this is soly for development purposes, data should be handled by a DB, to overcome memory limitations

@dataclass
class ClinicalTables:
    person: pd.DataFrame | None = None
    observation_period: pd.DataFrame | None = None
    death: pd.DataFrame | None = None
    visit_occurrence: pd.DataFrame | None = None
    visit_detail: pd.DataFrame | None = None
    condition_occurrence: pd.DataFrame | None = None
    drug_exposure: pd.DataFrame | None = None
    procedure_occurrence: pd.DataFrame | None = None
    device_exposure:pd.DataFrame | None = None
    measurement: pd.DataFrame | None = None
    observation: pd.DataFrame | None = None
    note: pd.DataFrame | None = None
    note_nlp: pd.DataFrame | None = None
    specimen: pd.DataFrame | None = None
    fact_relationship: pd.DataFrame | None = None
    
@dataclass 
class HealthSystemTables:
    provider: pd.DataFrame | None = None
    care_site: pd.DataFrame | None = None
    location: pd.DataFrame | None = None
    
@dataclass
class HealthEconomicsTabels:
    cost: pd.DataFrame | None = None
    payer_plan_period: pd.DataFrame | None = None
    
@dataclass
class StandartizedDerivedElementsTables:
    condition_era: pd.DataFrame | None = None
    drug_era: pd.DataFrame | None = None
    dose_era: pd.DataFrame | None = None
    episode: pd.DataFrame | None = None
    episode_event: pd.DataFrame | None = None
    cohort: pd.DataFrame | None = None
    cohort_definition: pd.DataFrame | None = None

@dataclass
class MetadataTables:
    metadata: pd.DataFrame | None = None
    cdm_source: pd.DataFrame | None = None
    
@dataclass
class VocabularyTables:
    concept: pd.DataFrame | None = None
    concept_class: pd.DataFrame | None = None
    vocabulary: pd.DataFrame | None = None
    source_to_concept_map: pd.DataFrame | None = None
    domain: pd.DataFrame | None = None
    concept_synonym: pd.DataFrame | None = None
    concept_relationship: pd.DataFrame | None = None
    relationship: pd.DataFrame | None = None
    drug_strength: pd.DataFrame | None = None

@dataclass
class OMOP_data:
    csv_data_path: InitVar[str | None] = None
    tables_structure: InitVar[dict | None] = None
    use_dask: InitVar[bool] = False
    clinical_tables: ClinicalTables = field(default_factory=ClinicalTables)
    health_system_tables: HealthSystemTables = field(default_factory=HealthSystemTables)
    health_economics_tables: HealthEconomicsTabels = field(default_factory=HealthEconomicsTabels)
    standartized_derived_elements_tables: StandartizedDerivedElementsTables = field(default_factory=StandartizedDerivedElementsTables)
    metadata_tables: MetadataTables = field(default_factory=MetadataTables)
    vocabulary_tables: VocabularyTables = field(default_factory=VocabularyTables)
    
    def __post_init__(self, csv_data_path:str, tables_structure:dict, use_dask:bool = False):
        self.csv_data_path = csv_data_path
        self.tables_structure = tables_structure
        self.use_dask = use_dask
        self._import_csv_files(tables_structure)
    
    def _import_csv_files(self, tables_structure:dict):
        for field in fields(self):
            print ("Ingesting", field.name+":")
            for table in fields(field.type):   
                print("Ingesting table", table.name+".")         
                file_path = os.path.join(self.csv_data_path, table.name+'.csv')
                if os.path.isfile(file_path):
                    try:
                        read_function = dd.read_csv if self.use_dask else pd.read_csv
                        df_table = read_function(file_path, 
                                            usecols=tables_structure.get(table.name).get('column_list'),
                                            dtype=tables_structure.get(table.name).get('dtype_dict'),
                                            parse_dates=tables_structure.get(table.name).get('parse_dates')
                                            )
                        setattr(getattr(self,field.name),table.name, df_table)
                        print('Ingesting file', file_path, "was successful.")
                    except Exception as e:
                        print(f"Unable to ingest {field.name}.{table.name} given {table.name}.csv file is off standards due to: {e}.")
                    
                else:
                    print(f"Unable to ingest {field.name}.{table.name} as there is not corresponding {table.name}.csv file.")
                
            print ("\n*****\n")