import pandas as pd
import json
from fast_keywords.preprocessing import json_to_str
from fast_keywords.objects import keywords, document


LANGUAGE = 'german'
WORDLIST = "Suchworte.xlsx"
PREFIX = "fast_keywords/res/nielsen/"
ARGS = {
    "ret": "ParsedResults",
    "txt": "ParsedText",
}
FILES = [
    "CHIP_010_2020",
    "COMPUTER BILD_023_2020",
    "PC_WELT_011_2020",
    "SPIEGEL_054_2020",
    "WIRTSCHAFTSWOCHE_044_2020"
]

def main():
    corpus = []
    for file in FILES:
        with open(f"{PREFIX}{file}.json", 'r') as f:
            # Appends list of page texts.
            corpus.append(json_to_str(json.load(f), **ARGS))

    df = pd.read_excel(f"{PREFIX}{WORDLIST}")
    kw = keywords.Keywords(df.searchtext.tolist(), ids=df.id.tolist())
    output = []
    # Each text is a list of pages.
    for file, text in zip(files, corpus):
        doc = document.Doc(
                text,
                keywords=kw,
                file=file,
                language=LANGUAGE,
                window=2,
                )
        output.append(doc.entities)

    output = pd.concat(output)
    output.to_csv('OUTPUT.csv', index=False)


if __name__ == '__main__':
    main()


# TODO: Investigate different windows.
# TODO: Increase the match threshold.
# TODO: Forgot any csv attributes?
