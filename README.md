Entity Sentiment Extraction
==============================

Extract sentiments sentences in which a specific entity appears. For each entity and each text document, the sentiment
of sentences in which the entity appears is extracted and aggregated to descriptive statistics (count, mean, quantiles, min, max).
The entities to search for as well as the descriptive statistics desired can be specified in the config file.

To extract entity sentiments:

1. Place a csv file with one text document per row in the 'text_inputs' folder

2. Specify configuration settings. See below for more details on these.

3. In the root directory open the CLI and run:

> make sentiments

For an example on the structure of the extraction script as well as proof of concept, see 'examples.ipynb' in the notebooks folder. For convenience example file input and config variables have been left in.


Requirements
------------

* GNU Make

* Python 3.6+

Project Organization
------------

    ├── LICENSE
    ├── Makefile                         <- Makefile with commands like `make sentiments`
    ├── README.md                        <- The top-level README for developers using this project.
    │              
    ├── notebooks                        <- Jupyter notebooks. See example.ipynb for a sample of how the
    │                                       entity sentiment extraction works
    │                                       
    │              
    ├── logs                             <- Logs for the make command that pulls entity sentiments
    │              
    ├── text_input/output                <- Folder for text input (in csv format) and outputted csv with sentiments extracted
    │              
    ├── requirements.txt                 <- The requirements file for reproducing the analysis environment, e.g.
    │                                       generated with `pip freeze > requirements.txt`   
    │
    ├── src                              <- Source code for use in this project.
    │   ├── __init__.py                  <- Makes src a Python module
    │   │
    │   ├── pull_entity_sentiment.py     <- Scripts to pull entity sentiment from input text
    │   │
    │   ├── textblob_entity_sentiment.py <- Code for searching text for entities and extracting sentiment
    │   │
    │   ├── helper-functions             <- Some Pandas helper functions
    │   │
    │   └── convenience_functions        <- Convenience functions for pandas and textacy, including a Dask
    │                                       multiprocessing partitioned apply to dataframes
    │
    ├── config.yaml                      <- Config file with environment variables required for extracting entity sentiments
        ├── input_filepath               <- the filepath to the text csv in the 'inputs' folder
        │
        ├── text_col                     <- The column name in the input csv which holds text
        │
        ├── entities                     <- A list of the entities for which to search and extract sentiment from
        │
        ├── sentiment_descriptive_stats  <- A list of the descriptive stats to pull for an entities sentiment for 
	                                        a given text. May choose from the following:
											'count', 'mean', 'min', '25%', '50%', '75%', 'max'
											25% and 75% refer to quantiles	 
	 