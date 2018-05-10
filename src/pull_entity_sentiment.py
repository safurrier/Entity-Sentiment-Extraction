
# coding: utf-8

# ## Imports

# In[1]:


import spacy
import textacy
import pandas as pd
import os
import ruamel.yaml as yaml
import datetime
import logging
import sys
import en_core_web_sm
nlp = en_core_web_sm.load()


# ## Change to root directory
NO_CONFIG_ERR_MSG = """No config file found. Root directory is determined by presence of "config.yaml" file."""

original_wd = os.getcwd()

# Number of times to move back in directory
num_retries = 10
for x in range(0, num_retries):
    # try to load config file
    try:
        with open("config.yaml", 'r') as stream:
            cfg = yaml.safe_load(stream)
    # If not found move back one directory level
    except FileNotFoundError:
        os.chdir('../')
        # If reached the max number of directory levels change to original wd and print error msg
        if x+1 == num_retries:
            os.chdir(original_wd)
            print(NO_CONFIG_ERR_MSG)

def main():
    # ## Add current wd to path for local imports
    path = os.getcwd()

    if path not in sys.path:
        sys.path.append(path)


    from src.convenience_functions.textacy_convenience_functions import load_textacy_corpus
    from src.convenience_functions.textacy_convenience_functions import entity_statements
    from src.convenience_functions.textacy_convenience_functions import list_of_entity_statements
    from src.convenience_functions.textacy_convenience_functions import dask_df_apply
    from src.textblob_entity_sentiment import textblob_entity_sentiment


    # ## Create log file

    # In[4]:


    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
    logging.basicConfig(filename='logs/{}.txt'.format(now),
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')


    # ## Load Data

    # In[5]:


    logging.info("""Reading in data from {}""".format(cfg['input_filepath']))


    df = pd.read_csv(cfg['input_filepath'])


    # ## Dask Multiprocessing of applied textacy docs

    # Using dask to multiprocess the loading of textacy docs for each text
    #
    # 1. Use dask to create partitioned dataframe
    #
    # 2. To each partition map an apply that creates textacy docs from the Policy_Text column
    #
    # 3. Concatenate back to original df

    # In[6]:


    logging.info("""Creating textacy Doc objects using the text found in the '{}' column""".format(cfg['text_col']))

    df = dask_df_apply(df, cfg['text_col'], inplace=True)


    # ## Extracting Entity Text, Counts and Sentiments

    # #### For each entity selected, return the count of entity occurence as well as mean, min and max of sentiments of sentences that contain said entity

    # In[7]:


    logging.info("""Extracting the following descriptive stats for entity sentiments: {} """.format(cfg['sentiment_descriptive_stats']))

    logging.info("""Extracting the sentiments for the following entities: {} """.format(cfg['entities']))

    sentiments = [textblob_entity_sentiment(df=df,
                                            textacy_col='textacy_doc',
                                            entity=entity,
                                            inplace=False,
                                            keep_stats=cfg['sentiment_descriptive_stats'])
                  for entity
                  in cfg['entities']]
    # Concat to single df
    sentiments = pd.concat(sentiments, axis=1)


    # #### Concat sentiment features and original df

    # In[8]:


    texts_with_sentiment_info = pd.concat([df, sentiments], axis=1).drop(labels=['textacy_doc'], axis=1)


    # In[9]:


    texts_with_sentiment_info.columns


    # ## Export features

    # In[10]:


    now = datetime.datetime.now().strftime("%Y-%m-%d %H-%M")
    archive_output_path = 'output/{}.csv'.format(now)
    logging.info("""Outputting sentiments to {}""".format(archive_output_path))
    print("""Outputting sentiments to {}""".format(archive_output_path))
    texts_with_sentiment_info.to_csv(archive_output_path, index=False)

if __name__ == "__main__":
    main()
