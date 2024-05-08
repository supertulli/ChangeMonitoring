### Deployment and running instructions

To build the docker container run the command:

```docker build -t omop-app .```

from the main project folder, which contains the docker file.

Then, before running the container, set up a folder holding all data files and 
edit the environment variables *docker.env* to the correct local paths to:

- CSV_FOLDER_PATH: the local folder holding all csv files; 
- TABLE_STRUCTURE_JSON: json file holding the respective OMOP db structure in the correct version; and
- ICD10_CHAPTER_DESCRIPTION: json file mapping icd10 chapter numbers to their name description.

as an example, the docker.env could hold the following lines:

```
CSV_FOLDER_PATH=/data/mimic-iv-demo-data-in-the-omop-common-data-model-0.9/1_omop_data_csv
TABLE_STRUCTURE_JSON=/data/OMOP_structure_and_types/OMOPCDM_5.3_structure.json
ICD10_CHAPTER_DESCRIPTION=/data/mappings/icd10_chapter_description.json
```

such that all files are within a common folder that will bind mounts into the docker container.

Then, to run the dockerized application, just run:

```docker run -p 8501:8501 -v ./data:/data --env-file docker.env omop-app```

and the html application will be available in the following address: 

http://localhost:8501.