

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
