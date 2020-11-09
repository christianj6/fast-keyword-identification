

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
        text : str
            Concatenated and cleaned string.
    '''
    text = []
    for item in json[ret]:
        text.append(item[txt].strip())

    return ' '.join([' '.join(item.split()) for item in text])
