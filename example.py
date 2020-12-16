import pandas as pd
import json
from fast_keywords.utils import json_to_str, get_distribution, load_keyword_product_dict, load_product_data_dict
from fast_keywords.objects import keywords, document


LANGUAGE = 'german'
WORDLIST = "Suchworte.xlsx"
PREFIX = "fast_keywords/res/nielsen/"
ARTICLE_DIR = "articles/"
ARGS = {
    "ret": "ParsedResults",
    "txt": "ParsedText",
}
FILES = [
    "CHIP_010_2020",
    "COMPUTER BILD_023_2020",
    "PC_WELT_011_2020",
    "SPIEGEL_054_2020",
    "WIRTSCHAFTSWOCHE_044_2020",
    "COSMOPOLITAN 1020",
    "TV MOVIE 2220",
    "MADAME 1020",
]
PRODUCTS = "products_new.csv"
KEYWORD_TO_PRODUCT = "keyword_to_product_new.csv"

def main():
    corpus = []
    for file in FILES:
        with open(f"{PREFIX}{ARTICLE_DIR}{file}.json", 'r') as f:
            # Appends list of page texts.
            corpus.append(json_to_str(json.load(f), **ARGS))

    df = pd.read_excel(f"{PREFIX}{WORDLIST}")
    kw = keywords.Keywords(df.searchtext.tolist(), ids=df.id.tolist())
    keyword_to_product = load_keyword_product_dict(f"{PREFIX}{KEYWORD_TO_PRODUCT}")
    products = load_product_data_dict(f"{PREFIX}{PRODUCTS}")
    output = []
    # Each text is a list of pages.
    for file, text in list(zip(FILES, corpus))[:1]:
        doc = document.Doc(
                text,
                keywords=kw,
                file=file,
                language=LANGUAGE,
                window=2,
                bound=.99,
                trained_filter=True,
                keyword_to_product=keyword_to_product,
                products=products
                )
        output.append(doc.entities)

    output = pd.concat(output)
    # Combine output and distribution into a single .xls
    writer = pd.ExcelWriter('OUTPUT.xls')
    output.to_excel(writer, 'keywords', index=False)
    # Get distribution for each file.
    for name, file in output.groupby(['File']):
        distribution = get_distribution(file)
        # Add as a new sheet.
        distribution.to_excel(writer, name, index=False)

    writer.save()


if __name__ == '__main__':
    main()


# TODO: Wrap the main() method into a pipeline function so it is cleaner.
# TODO: Ensure entity validation protocol is sensible. Consolidated entities?
# TODO: Flag areas of the text which are probably ads.
# TODO: Parse special characters eg ß.
