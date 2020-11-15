import pandas as pd


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
