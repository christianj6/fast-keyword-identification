import os
import pandas as pd
from .objects import keywords, document
from .tools import get_distribution
from tqdm import tqdm


def run(**kwargs):
    assert all(
        [
            os.path.exists(pth)
            for pth in tuple(map(lambda x: kwargs[x], ("keywords", "corpus")))
        ]
    ), "You must provide a correct absolute path for both the keywords and corpus files."
    bound = 0.95
    if kwargs["bound"] is not None:
        bound = kwargs["bound"]

    with open(kwargs["keywords"], "r") as f:
        words = f.read().splitlines()

    kw = keywords.Keywords(words=words, ids=list(range(len(words))))
    corpus = pd.read_csv(kwargs["corpus"])

    output = []
    for t in tqdm(corpus.text.tolist()):
        doc = document.Doc(
            text=t.split(" "),
            keywords=kw,
            file=kwargs["corpus"],
            language=kwargs["language"],
            bound=bound,
        )
        output.append(doc.entities)

    output = pd.concat(output)

    assert len(output) > 0, "No keywords were found."

    # Combine output and distribution into a single .xls
    writer = pd.ExcelWriter("output.xlsx")
    output.to_excel(writer, "keywords", index=False)
    # Get distribution for each file.
    for name, file in output.groupby(["Keyword"]):
        distribution = get_distribution(file)
        # Add as a new sheet.
        distribution.to_excel(writer, name, index=False)

    writer.save()
