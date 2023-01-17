import re
import time

import os
import nltk
import json
from flask import Flask, request, jsonify
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from backend_calculations import *

nltk.download('stopwords')

# --- Global variables --- #
BUCKET_NAME = "bucket2121"

PS = PorterStemmer()
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
STOP_WORDS = frozenset(stopwords.words('english')).union(["category", "references", "also", "external", "links", \
                                                          "may", "first", "see", "history", "people", "one", "two", \
                                                          "part", "thumb", "including", "second", "following", \
                                                          "many", "however", "would", "became"])
# --- Inverted index directories --- #
BODY_DIR = "body"
TITLE_DIR = "titles"
ANCHOR_DIR = "anchor"

# --- Inverted index files --- #
BODY_IND_FILE = "body_index"
TITLE_IND_FILE = "title_index"
ANCHOR_IND_FILE = "anchor_index"

# --- Components JSON files --- #
TITLE_JSON = "titles"
PAGE_RANK_JSON = "pr"
PAGE_VIEWS_JSON = "pv"


# --- Helper Functions --- #
def import_index(base_dir, file):
    """
    Reads the index from the bucket using "read_index" method
    from class InvertedIndex
    :param base_dir: str
    :param file: str
    :return: InvertedIndex
    """
    return InvertedIndex.read_index(BUCKET_NAME, base_dir, file)


def read_json_file(component):
    """
    Reads the json file from the bucket using "json" library
    :param component: str
    :return: json
    """
    os.system(f"gsutil cp gs://{BUCKET_NAME}/{component}/{component}.json .")
    with open(f"{component}.json") as f:
        res = json.load(f)
    return res


def tokenize(text, filter_flag=False):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filters stopwords.

    Parameters:
    -----------
    text: string , representing the text to tokenize.
    filter_flag: boolean, used for stemming if needed.
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in STOP_WORDS]
    if filter_flag:
        list_of_tokens = [PS.stem(token) for token in list_of_tokens]
    return list_of_tokens


def get_title(scores):
    """
    Returns the title for every wiki_id
    :param scores: list of pairs (wiki_id, score).
    :return: list, (wiki_id, title)
    """
    return [(x[0], TITLES[str(x[0])]) for x in scores]


# --- MyFlaskApp Class --- #
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


# --- Initializations --- #

# --- initialize the index for body,title,anchor --- #

BODY_INVERTED_INDEX: InvertedIndex = import_index(BODY_DIR, BODY_IND_FILE)

TITLE_INVERTED_INDEX: InvertedIndex = import_index(TITLE_DIR, TITLE_IND_FILE)

ANCHOR_INVERTED_INDEX: InvertedIndex = import_index(ANCHOR_DIR, ANCHOR_IND_FILE)

# --- read the json file for pagerank ,pageviews, titles --- #

PAGE_RANK = read_json_file(PAGE_RANK_JSON)


PAGE_VIEWS = read_json_file(PAGE_VIEWS_JSON)


TITLES = read_json_file(TITLE_JSON)



@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # TODO check homeworks

    # Body
    tokens = tokenize(query)
    body_results = get_top_n_score_for_queries(tokens, BODY_INVERTED_INDEX, BODY_DIR, 100)

    # Titles
    tokens_stems = tokenize(query)
    titles_results = sorting_results_using_ranking(TITLE_INVERTED_INDEX, tokens_stems, TITLE_DIR)[:100]

    # Anchors
    anchors_results = sorting_results_using_ranking(ANCHOR_INVERTED_INDEX, tokens, ANCHOR_DIR)[:100]

    # All candidates:
    res = merge_results(titles_results, body_results, anchors_results, PAGE_RANK, PAGE_VIEWS, n=100)
    res = get_title(res)
    # END SOLUTION
    return jsonify(res)


# --- Search in body function --- #
@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    # Get top 100 results sorted by TF-IDF score using cosine similarity.
    top_n_results = get_top_n_score_for_queries(tokens, BODY_INVERTED_INDEX, BODY_DIR, 100)
    # Reformatting results to include titles.
    res = get_title(top_n_results)
    # END SOLUTION
    return jsonify(res)


# --- Search in title function --- #
@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    res = sorting_results_using_ranking(TITLE_INVERTED_INDEX, tokens, TITLE_DIR)
    res = get_title(res)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with an anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    # TODO check
    res = sorting_results_using_ranking(ANCHOR_INVERTED_INDEX, tokens, ANCHOR_DIR)
    res = get_title(res)
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # Get the page rank from pr dictionary by given wiki_id.
    for wiki_id in wiki_ids:
        wiki_id = str(wiki_id)
        if wiki_id in PAGE_RANK:
            res.append(PAGE_RANK[wiki_id])
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provided wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # Get the page views from pv dictionary by given wiki_id.
    # BEGIN SOLUTION
    for wiki_id in wiki_ids:
        wiki_id = str(wiki_id)
        if wiki_id in PAGE_VIEWS:
            res.append(PAGE_VIEWS[wiki_id])
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
