import pandas as pd
from fast_keywords.objects import keywords
import os
import dill
import re


WORDLIST = "Suchworte.xlsx"
PREFIX = "fast_keywords/res/nielsen/"

def json_to_str(json:dict, ret:str, txt:str)->str:
    '''
    Concatenates text attributes from
    json files into a single, string
    of text.

    Parameters
    ---------
        json : dict
            Loaded json dictionary.
        ret : str
            Attribute for return value.
        txt : str
            Attribute for text value.

    Returns
    ---------
        text : list[str]
            List of page texts as cleaned
            strings.
    '''
    text = []
    for page in json[ret]:
        text.append(page[txt].strip())

    return [' '.join(item.split()) for item in text]


def get_distribution(output:'pd.DataFrame'):
    '''
    Get keyword distribution statistics
    that we can output these
    to the final csv.

    Parameters
    ---------
        output : pd.DataFrame
            Output with
            identified entities.

    Returns
    ---------
        distribution : pd.DataFrame
            Distribution statistics
            in tabular form.
    '''
    distribution = []
    for name, group in output.groupby(["Keyword"]):
        distribution.append(
                {
                    "Keyword": name,
                    "Count": len(group),
                }
            )

    return pd.DataFrame(distribution)


def evaluate_classifiers(filename):
    '''
    Evaluate classification
    accuracy in a messy fashion,
    by averaging all results
    to see how the classifers
    are collectively performing.

    Parameters
    ---------
        file : str
            File for eval.

    Returns
    ---------
        score : float
            Classification accuracy.
    '''
    words = pd.read_excel(f"{PREFIX}{WORDLIST}")
    kw = keywords.Keywords(words.searchtext.tolist(), ids=words.id.tolist())
    scores = []
    output = pd.read_excel(filename).infer_objects()
    for file in os.listdir("fast_keywords/models/german"):
        with open(f'fast_keywords/models/german/{file}', 'rb') as f:
            model = dill.load(f)

        rows = output[output['Keyword'] == file]
        X = rows['Surrounding Text'].tolist()
        X = [re.sub(r'[A-Z]', '', x) for x in X]
        X = [kw.get_vector(x).toarray()[0] for x in X]
        y = rows['Match is Invalid'].tolist()
        scores.append((file, model.model.score(X, y)))

    return scores


def load_keyword_product_dict(keyword_to_product:str) -> dict:
    '''
    Load a mapping from keyword id to possible product ids,
    that these can then be used for an additional
    filtering step during entity extraction.

    Parameters
    ---------
        keyword_to_product : str
            Filepath.

    Returns
    ---------
        output : dict
            Mapping.
    '''
    # Noise
    noise = [".", ",", "image"]
    output = {}
    df = pd.read_csv(keyword_to_product)
    for idx, group in df.groupby(by=["Keyword ID"]):
        id_to_word = {}
        for _, row in group.iterrows():
            for word in row['Surrounding Text'].split():
                if not word.lower() == row['Keyword'].lower() \
                        and not word.lower() in noise:
                    id_to_word[word.lower()] = row["File"]

        output[idx] = id_to_word

    return output


def load_product_data_dict(products:str) -> dict:
    '''
    Load product metadata keyed to product ids that
    these data can then be associated with
    identified entities.

    Parameters
    ---------
        products : str
            Filepath.

    Returns
    ---------
        output : dict
            Mapping.
    '''
    df = pd.read_csv(products)
    df = df.set_index("Unnamed: 0")

    return df.to_dict(orient="index")
