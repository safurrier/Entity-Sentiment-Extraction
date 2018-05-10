from src.convenience_functions.pandas_functions import pivot_df_to_row
from src.convenience_functions.textacy_convenience_functions import entity_statements
import pandas as pd

import textblob

# make sure to put textblob imports if moving this function to outside module
# Cache the return of this since it takes a while to process
# # If the function is called with different arguments it will run again rather read the cache
# memory = Memory(cachedir='/tmp', verbose=0)
# @memory.cache
def textblob_entity_sentiment(df,
                              textacy_col,
                              entity,
                              inplace=True,
                              subjectivity=False,
                              keep_stats=['count', 'mean', 'min', '25%', '50%', '75%', 'max']):
    """
    Pull the descriptive sentiment stats of text sentence with a specified entity in it.

    Parameters
    ----------
    df : DataFrame
        Dataframe which holds the text
    textacy_col : str
        The name to give to the column with the textacy doc objects
    entity : str
        The entity to search the textacy Doc object for
    inplace : bool
        Whether to return the entire df with the sentiment info or the sentiment info alone
        Default is False
    subjectivity : bool
        Whether to include the subjectivity of the sentiment. Defaults to False.
    keep_stats : list
        A list of the summary statistics to keep. Default is all returned by pandas DataFrame.describe() method

    Returns
    -------
    DataFrame
        Either the dataframe passed as arg with the sentiment info as trailing columns
        or the sentiment descriptive stats by itself
    """
    sentiment_rows = []
    for text in df[textacy_col].values:
        text_entities = list(entity_statements(text, entity))

         # Iterate through all sentences and get sentiment analysis
        entity_sentiment_info = [textblob.TextBlob(sentence).sentiment_assessments
                                for
                                sentence
                                in
                                text_entities]

        # After taking sentiments, turn into a dataframe and describe
        try:
            # Indices and columns to keep
            #keep_stats = ['count', 'mean', 'min', '25%', '50%', '75%', 'max']
            keep_cols = ['polarity']

            # If subjectivity is set to true, values for it will also be captured
            if subjectivity:
                keep_cols.append('subjectivity')

            # Describe those columns
            summary_stats = pd.DataFrame(entity_sentiment_info).describe().loc[keep_stats, keep_cols]

            # Add row to list
            sentiment_rows.append(pivot_df_to_row(summary_stats))

        # If there's nothing to describe
        except ValueError as e:
            # Create a summary stats with nulls
            summary_stats = pd.DataFrame(index=keep_stats, columns=keep_cols)

            # Add to list of rows
            sentiment_rows.append(pivot_df_to_row(summary_stats))
    # Concatenate All rows together into one dataframe
    sentiment_df = pd.concat(sentiment_rows).add_prefix(entity+'_')

    if not inplace:
        return sentiment_df.reset_index(drop=True)
    else:
        # Return original df with new sentiment attached
        return pd.concat([df, sentiment_df], axis=1)
