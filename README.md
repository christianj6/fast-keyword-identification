### Fast Keyword Identification
Fast keyword identification with n-gram vector string matching.
***

#### Overview
This package provides a generic pipeline for fuzzy-identification of keywords in large document collections. For example, if you wish to find all occurrences of the keyword "Walmart" in a large document collection, but expect some typos or variations in spelling, this module will allow you to quickly identify all matches. The matcher is based on a character n-gram vector model rather than the slower string edit distance. The module is originally intended for brand monitoring applications.

***

### Installation

```
pip install fast-keywords
```

***

### CLI

```
python -m fast-keywords --help
```

***

### Usage

```
python -m fast-keywords -k keywords.csv -c corpus.csv
```

***

