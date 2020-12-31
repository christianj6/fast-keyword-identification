############################################################
############################################################
############################################################

# window = 2

# text = 'a new car by porsche design'
# text = text.split()

# for i, token in enumerate(text):
#     # print(token.upper())
#     for j in range(window):
#         # print(text[i-(j+1):i+1])
#         ngram = ' '.join(text[i-(j+1):i+1])
#         if ngram:
#             print(ngram)
#         # candidates = matcher(ngram)


# # for j in range(window):
# #     print(-(j+1))

# for i in range(3, 4):
#     print(i)

# print(range(3, 4)[0])

# if 3 in range(3, 4):
#     print(True)

# span = range(3, 4)
# env = ''
# for token in text[span[0]-2:span[-1]+3]:
#     env += token + ' '

# print(env)

############################################################
############################################################
############################################################

# files = [1, 2, 3]
# for f in files[:1]:
#     print(f)

############################################################
############################################################
############################################################

# import pandas as pd
# from tqdm import tqdm

# df = pd.read_excel('fast_keywords/res/nielsen/Suchworte.xlsx')
# words = df.searchtext.tolist()

# lexicon = []
# with open('fast_keywords/res/tokens_german.txt', 'r') as f:
#     for line in f:
#         lexicon.append(line.lower().strip())

# with open('fast_keywords/res/tokens_english.txt', 'r') as f:
#     for line in f:
#         lexicon.append(line.lower().strip())

# collisions = []
# for word in tqdm(words):
#     if str(word).lower() in lexicon:
#         collisions.append(word)

# for word in collisions:
#     print(word)

############################################################
############################################################
############################################################

# from fast_keywords.utils import evaluate_classifiers

# print(evaluate_classifiers('OUTPUT.xls'))

############################################################
############################################################
############################################################

# import json
# from fast_keywords.utils import json_to_str

# PREFIX = "fast_keywords/res/nielsen/"
# ARGS = {
#     "ret": "ParsedResults",
#     "txt": "ParsedText",
# }
# FILES = [
#     "CHIP_010_2020",
#     "COMPUTER BILD_023_2020",
#     "PC_WELT_011_2020",
#     "SPIEGEL_054_2020",
#     "WIRTSCHAFTSWOCHE_044_2020",
#     "COSMOPOLITAN 1020",
#     "TV MOVIE 2220",
#     "MADAME 1020",
# ]

# def main():
#     corpus = []
#     for file in FILES:
#         with open(f"{PREFIX}{file}.json", 'r') as f:
#             # Appends list of page texts.
#             corpus.append(json_to_str(json.load(f), **ARGS))

#     texts = ' '.join([' '.join(doc) for doc in corpus])
#     with open('texts.txt', 'w') as f:
#         f.write(texts)

# main()

############################################################
############################################################
############################################################

# MATCHING THE KEYWORDS TO THE NEW LISTS

# import pandas as pd
# import json
# from fast_keywords.utils import json_to_str, get_distribution
# from fast_keywords.objects import keywords, document

# LANGUAGE = 'german'
# WORDLIST = "Suchworte.xlsx"
# PREFIX = "fast_keywords/res/nielsen/"

# a = pd.read_excel(f"{PREFIX}Martsystematik redaktionelle Erwähnungen (1).xlsx", sep=';')
# b = pd.read_csv(f"{PREFIX}German Market Data.csv", sep=";")
# c = pd.concat([a, b], sort=False)

# c.reset_index(inplace=True)
# c.to_csv('products_new.csv')

# products = c.Produkt.values.tolist()
# products = list(map(lambda x: str(x).lower(), products))

# ids = c.index.values.tolist()

# df = pd.read_excel(f"{PREFIX}{WORDLIST}")
# kw = keywords.Keywords(df.searchtext.tolist(), ids=df.id.tolist())

# output = []
# for text, i in list(zip(products, ids)):
#     try:
#         doc = document.Doc(
#             [text],
#             keywords=kw,
#             file=i,
#             language=LANGUAGE,
#             window=2,
#             bound=.8,
#             trained_filter=False,
#             )
#         output.append(doc.entities)
#     except IndexError:
#         pass

# output = pd.concat(output)cd
# output.to_csv('out_new.csv')

############################################################
############################################################
############################################################

# TRYING TO IDENTIFY ARTICLE BOUNDARIES

# import json
# from fast_keywords.utils import json_to_str

# LANGUAGE = 'german'
# PREFIX = "fast_keywords/res/nielsen/"

# FILES = [
#     "CHIP_010_2020",
#     "COMPUTER BILD_023_2020",
#     "PC_WELT_011_2020",
#     "SPIEGEL_054_2020",
#     "WIRTSCHAFTSWOCHE_044_2020",
#     "COSMOPOLITAN 1020",
#     "TV MOVIE 2220",
#     "MADAME 1020",
# ]

# for file in FILES[:1]:
#     with open(f"{PREFIX}{file}.json", 'r') as f:
#         print(json.dumps(json.load(f), indent=4))

############################################################
############################################################
############################################################
