import pandas as pd
import textacy

from dask import dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count
import textacy
import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()

def dask_df_apply(df, text_col, textacy_col_name='textacy_doc', ncores=None, inplace=False):
    """
    Use dask to parallelize apply textacy Doc object creation from a dataframe

    Parameters
    ----------
    df : DataFrame
        Dataframe which holds the text
    text_col : str
        The name of the text column in the df
    textacy_col_name : str
        The name to give to the column with the textacy doc objects
    ncores : int
        Number of cores to use for multiprocessing. Defaults to all cores in cpu minus one.
    inplace : bool
        Whether to return the entire df with the textacy doc series concatenated
        or only textacy doc series.
        Default is False
    Returns
    -------
    DataFrame / Series
        Either the dataframe passed as arg with the textacy series as last column or
        just the textacy column
    """
    # If no number of cores to work with, default to max
    if not ncores:
        nCores = cpu_count() - 1
        nCores

    # Partition dask dataframe and map textacy doc apply
	# Sometimes this fails because it can't infer the dtypes correctly
	# meta=pd.Series(name=0, dtype='object') is a start
	# This is also a start https://stackoverflow.com/questions/40019905/how-to-map-a-column-with-dask?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    # Possibly both the inner lambda apply and outer lambda df both need metadata?	
    textacy_series = dd.from_pandas(df, npartitions=nCores).map_partitions(
      lambda df : df[text_col].apply(lambda x : textacy.doc.Doc(x, lang=nlp))).compute(get=get)

    # Name the series
    textacy_series.name = textacy_col_name

    # If inplace return the dataframe and textacy Series
    if inplace:
        return pd.concat([df, textacy_series], axis=1)
    # Else return just the Textacy series
    else:
        return textacy_series		

def load_textacy_corpus(df, text_col, metadata=True, metadata_columns=None):
    # Fill text columns nulls with empty strings
    df[text_col] = df[text_col].fillna('')
    if metadata:
        # Default to metadata columns being every column except the text column
        metadata_cols = list(df.columns)
        # If list is provided use those
        if metadata_columns:
            metadata_cols = metadata_columns

        # Add text column to metadata columns
        # These will constitute all the information held in the textacy corpus
            metadata_columns.append(text_col)
        # Subset to these
        df = df[metadata_cols]

        # Convert to nested dict of records
        records = df.to_dict(orient='records')
        # Split into text and metadata stream
        text_stream, metadata_stream = textacy.io.split_records(records, text_col)

        # Create Corpus
        return textacy.corpus.Corpus(lang='en', texts=text_stream, metadatas=metadata_stream)
    # With no metadata
    else:
        text_stream = (text for text in df[text_col].values)
        return textacy.corpus.Corpus(lang='en', texts=text_stream)

def corpus_entity_counts(corpus, include=None, exclude=None):
    """
    Given a textacy corpus, return a dataframe of entities and their respective counts.

    Parameters
    ----------
    corpus : int
        Description of arg1
    include : str or Set[str]
        Remove named entities whose type IS NOT in this param;
        if “NUMERIC”, all numeric entity types (“DATE”, “MONEY”, “ORDINAL”, etc.) are included
    exclude : str or Set[str]
        remove named entities whose type IS in this param; if “NUMERIC”,
        all numeric entity types (“DATE”, “MONEY”, “ORDINAL”, etc.) are excluded

    Returns
    -------
    Dataframe
        A pandas dataframe with entities and their respective counts, sorted by highest count

    """
    from collections import Counter

    # Extract all entities
    entities = [list(textacy.extract.named_entities(doc, include_types=include, exclude_types=exclude))
                for doc in
                corpus]
    # Pull all non-null entities to flattened list
    non_null_entities = []
    for entity in entities:
        if entity:
            non_null_entities.extend(entity)
    # Change dtype to string so counter can distinguish
    non_null_entities = [str(x) for x in non_null_entities]

    # Count entities
    entity_counts = Counter(non_null_entities)

    # Entity Dataframe
    df = (pd.DataFrame.from_dict(entity_counts, orient='index')
          .reset_index()
          .rename(columns={'index':'Entity', 0:'Count'})
          .sort_values(by='Count', ascending=False)
          .reset_index(drop=True))

    return df

def entity_statements(doc, entity, ignore_entity_case=True,
                      min_n_words=1, max_n_words=300, return_entity=False):
    """
    Extract sentences with a specified entity present in it
    Modified from source code of Textacy's textacy.extract.semistructured_statements()

    Args:
        doc (``textacy.Doc`` or ``spacy.Doc``)
        entity (str): a noun or noun phrase of some sort (e.g. "President Obama",
            "global warming", "Python")
        ignore_entity_case (bool): if True, entity matching is case-independent
        min_n_words (int): min number of tokens allowed in a matching fragment
        max_n_words (int): max number of tokens allowed in a matching fragment

    Yields:
        (``spacy.Span`` or ``spacy.Token``) or (``spacy.Span`` or ``spacy.Token``, ``spacy.Span`` or ``spacy.Token``):
        dependin on if return_entity is enabled or not


    Notes:
        Inspired by N. Diakopoulos, A. Zhang, A. Salway. Visual Analytics of
        Media Frames in Online News and Blogs. IEEE InfoVis Workshop on Text
        Visualization. October, 2013.

        Which itself was inspired by by Salway, A.; Kelly, L.; Skadiņa, I.; and
        Jones, G. 2010. Portable Extraction of Partially Structured Facts from
        the Web. In Proc. ICETAL 2010, LNAI 6233, 345-356. Heidelberg, Springer.
    """
    if ignore_entity_case is True:
        entity_toks = entity.lower().split(' ')
        get_tok_text = lambda x: x.lower_
    else:
        entity_toks = entity.split(' ')
        get_tok_text = lambda x: x.text

    first_entity_tok = entity_toks[0]
    n_entity_toks = len(entity_toks)
    #cue = cue.lower()
    #cue_toks = cue.split(' ')
    #n_cue_toks = len(cue_toks)

    def is_good_last_tok(tok):
        if tok.is_punct:
            return False
        if tok.pos in {CONJ, DET}:
            return False
        return True

    for sent in doc.sents:
        for tok in sent:

            # filter by entity
            if get_tok_text(tok) != first_entity_tok:
                continue
            if n_entity_toks == 1:
                the_entity = tok
                the_entity_root = the_entity

            elif all(get_tok_text(tok.nbor(i=i + 1)) == et for i, et in enumerate(entity_toks[1:])):
                the_entity = doc[tok.i: tok.i + n_entity_toks]
                the_entity_root = the_entity.root
            else:
                continue
            if return_entity:
                yield (the_entity, sent.orth_)
            else:
                yield (sent.orth_)
            break

def list_of_entity_statements(corpus, entity):
    """
    Given an entity and a textacy corpus, return a list of all the sentences in which this entity occurs

    Parameters
    ----------
    corpus : textacy Corpus object
    entity : str
        The entity for which to search all the sentences within the corpus
    Returns
    -------
    entity_sentences
        A list of strings, each being a sentence which contains the entity search
    """

    entity_sentences = [list(entity_statements(doc, entity=entity))
                        for doc
                        in corpus
                        if list(entity_statements(doc, entity=entity))] # If statement that removes null sentences

    entity_sentences = [item for sublist in entity_sentences for item in sublist]

    return entity_sentences
