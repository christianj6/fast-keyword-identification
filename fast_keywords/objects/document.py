from fast_keywords.objects import entity, keywords
from typing import Callable
import pandas as pd
from collections import Counter
from tqdm import tqdm


class Doc():
    '''
    Object for storing attributes of a single
    parsed document, including the
    original text and any entities
    identified against a keyword matcher.
    '''
    def __init__(
        self,
        text:list,
        keywords:keywords.Keywords,
        file:str,
        language:str,
        window:int=2,
    ):
        '''
        Instantiate object and extract
        filtered entities from keywords.

        Parameters
        ---------
            text : list
                Original text as list of strings. Each
                string is the text of a single page.
            keywords : keywords.Keywords
                Keyword matcher object. Should
                be already fitted to
                a keyword list.
            file : str
                Name of the file. Needed in
                case we later want to
                concatenate all the entities from
                separate documents, that they can
                be mapped back to the original files.
            language : str
                Source language.
            window : int
                Maximum n-gram window for which
                we want to recursively check for
                more-relevant matches.
        '''
        # Assign miscellaneous attributes.
        self.language = language
        self.keywords = keywords
        self.file = file
        self.window = window
        # Lowercase and split the text.
        self.text, self.page_mask = self.preprocess_text(text)
        # Extract entities from text.
        self.entities = self.get_entities()

    @staticmethod
    def preprocess_text(text):
        '''
        Process the original text into a list
        of lists, where each higher-order list
        is the text of a single page, and the lower-order
        list are the tokens of the text of that page.

        Parameters
        ---------
            text : list[str]
                List of page texts as strings.

        Returns
        ---------
            text : list[str]
                List of all tokens in the text.
            mask : list[int]
                Mask each token to its page no.
        '''
        txt = []
        mask = []
        for i, page in enumerate(text):
            # Basic preprocessing to separate punctuations.
            for token in page.lower().replace(".", " . ").replace(",", " , ").split():
                txt.append(token)
                mask.append(i)

        return txt, mask

    def get_entities(self)->pd.DataFrame:
        '''
        Extract recognizeable entities from
        object text and return a dataframe
        summarizing their location
        and properties.

        Returns
        ---------
            entities : pd.DataFrame
                Entities organized
                in tabular form
        '''
        # Get matches.
        matches = self.get_matches(
                        self.text,
                        self.keywords.match,
                        self.window
                    )
        # Filter the matches by several heuristics.
        matches = self.filter_matches(matches)
        # Cast filtered matches to entities.
        entities = []
        for span, (word, score, i) in matches:
            entities.append(
                    entity.Entity(
                            page=self.page_mask[span[0]],
                            location=span,
                            string=' '.join([self.text[j] for j in span]),
                            match=word,
                            idx=self.keywords.ids[i],
                            score=score,
                            text=self.text,
                        )
                )

        # TODO: Filter entities based on the environment.
        # TODO: Consolidate entities by keywords.
        # TODO: Issue with variable assignment in environment method?

        # Cast all entities to df.
        df = []
        for entity in entities:
            df.append(
                {
                    "File": self.file,
                    "Page": entity.page,
                    "Location": entity.location,
                    "Keyword": entity.match,
                    "Keyword ID": entity.idx,
                    "Matched String": entity.string,
                    "Match Confidence": entity.score,
                    "Surrounding Text": entity.environment,

                }
            )

        return pd.DataFrame(df)

    @staticmethod
    def get_matches(
        text:list,
        matcher:Callable[[str],list],
        window:int,
    )->list:
        '''
        Get all matches for input text,
        returning information on the matched
        string and its location.

        Parameters
        ---------
            text : list
                List of tokens.
            matcher : Callable
                Function for matching strings.
            window : int
                Window for looking back when getting matches.

        Returns
        ---------
            matches : list
                List of tuples containing
                match information including the location.
        '''
        matches = []
        for i, token in enumerate(text):
            # Get matches for current token.
            candidates = matcher(token)
            # Add these to the list with their locations.
            matches.extend([(range(i, i+1), candidate) for candidate in candidates])
            # Get matches for each ngram looking back.
            for j in range(window):
                ngram = ' '.join(text[i-(j+1):i+1])
                # At beginning of a text strings will be empty.
                if ngram:
                    candidates = matcher(ngram)
                    # Add to list with location ranges.
                    matches.extend([(range(i-(j+1), i+1), candidate) for candidate in candidates])

        return matches

    @staticmethod
    def filter_matches(matches:list)->list:
        '''
        Filter matches according to several
        heuristics which aim to overcome collisions
        by prioritizing spans with more tokens and
        higher match scores.

        Parameters
        ---------
            matches : list
                List of tuples containing
                match info and location spans.

        Returns
        ---------
            matches : list
                Filtered matches.
        '''
        # Use a mask to track the filtering.
        mask = []
        # Count attested tokens so we can immediately
        # extract those which have no collisions.
        counter = Counter([i for span,_ in matches for i in span])
        for span, (word, score, i) in matches:
            # Immediately grab matches of single tokens with
            # no collisions.
            if len(span) == 1 and Counter[span[0]] == 1:
                mask.append(True)
                continue

            # If single token is matched multiple times, assume
            # that waiting will yield a superior match to
            # a longer string of text.
            elif len(span) == 1 and Counter[span[0]] >1:
                mask.append(False)
                continue

            # Immediately grab 100% matches.
            elif score == 1.0:
                mask.append(True)
                continue

            # Otherwise don't accept the match.
            else:
                mask.append(False)

        # Apply the mask.
        return [match for match,boolean in zip(matches, mask) if boolean == True]
