import math
import numpy as np
import pandas as pd
from collections import Counter
from inverted_index_gcp import InvertedIndex


def sorting_results_using_ranking(index, tokens, comp):
    """
    The function returns all components in index sorted by the ranking scores. Tha ranking is calculated using binary
    similarity. Returns list of ALL (not just top 100) search results, ordered from best to worst where each element
    is a tuple (wiki_id, title).
    :param comp: string, directory name for component in gcp
    :param index: InvertedIndex
    :param tokens: list of tokens of query to search
    :return: list
    """

    index_scores_dict = {}
    for token in tokens:
        pls = index.read_posting_list(token, comp)
        for doc_id, tf in pls:
            index_scores_dict[doc_id] = index_scores_dict.get(doc_id, 0) + 1 / len(tokens)
    sorted_scores = sorted(index_scores_dict.items(), key=lambda pair: pair[1], reverse=True)
    return sorted_scores


def generate_query_tfidf_vector(query_to_search, index: InvertedIndex):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrieval' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    query_vec = np.zeros(len(query_to_search))
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys() and token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = query_to_search.index(token)
                query_vec[ind] = tf * idf
            except:
                pass
    return query_vec


def get_posting_iter(index, comp):
    """
    This function returning the iterator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter(comp))
    return words, pls


def get_candidate_documents_and_scores(query_to_search, index, comp):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrieval' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for item in query_to_search:
        pls = index.read_posting_list(item, comp)
        if len(pls) != 0:
            normalized_tfidf = [(doc_id, (freq / index.DL[doc_id]) * math.log(len(index.DL) / index.df[item], 10))
                                for
                                doc_id, freq in pls]
            for doc_id, tfidf in normalized_tfidf:
                candidates[(doc_id, item)] = candidates.get((doc_id, item), 0) + tfidf
    return candidates


def generate_document_tfidf_matrix(query_to_search, index, comp):
    """
    Generate a DataFrame `doc_mat` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrieval' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.


    words,pls: iterator for working with posting.

    Returns:
    -----------
    DataFrame of tfidf scores.
    """
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index,
                                                           comp)  # We do not need to utilize all document. Only the documents which have corresponding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    doc_mat = np.zeros((len(unique_candidates), len(query_to_search)))
    doc_mat = pd.DataFrame(doc_mat)

    doc_mat.index = unique_candidates
    doc_mat.columns = query_to_search

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        doc_mat.loc[doc_id][term] = tfidf

    return doc_mat


def cosine_similarity(doc_mat, query_vec):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarity score.
    """
    # YOUR CODE HERE

    cos_sim_scores = {}
    for doc_id in doc_mat.index:
        doc_vec = doc_mat.loc[doc_id].values
        cos_sim_scores[doc_id] = np.dot(doc_vec, query_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(
            query_vec))  # calculates the cosine similarity between two vectors doc_vec and Q. since we can't use sklearn.
    return cos_sim_scores


def get_top_n(sim_dict, n=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :n]


def get_top_n_score_for_queries(queries_to_search, index, comp, n=3):
    """
    Generate a dictionary that gathers for every query its topN score.

    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows:
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default, N = 3, for the topN function.

    Returns:
    -----------
    return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """
    # words, pls = get_posting_iter(index, queries_to_search, comp)
    # words = index.posting_locs.keys()
    doc_mat = generate_document_tfidf_matrix(queries_to_search, index,
                                             comp)  # DataFrame of tfidf scores (rows are documents candidate for a given query and columns are terms).
    query_vec = generate_query_tfidf_vector(queries_to_search, index)  # vectorized query with tfidf scores.
    sim_scores = cosine_similarity(doc_mat, query_vec)  # send both to get the cosine similarity dictionary.
    return get_top_n(sim_scores, n)  # save best N sim_scores of the query we are iterating at.


def merge_results(title_scores, body_scores, anchor_scores, page_rank, page_views, title_weight=0.2, text_weight=0.2,
                  anchor_weight=0.2, pr_weight=0.2, pv_weight=0.2, n=3):
    """
    This function merge and sort documents retrieved by its weighted score (e.g., title and body).

    Parameters: ----------- title_scores: a dictionary build upon the title index of queries and tuples representing
    scores as follows: key: query_id value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
    key: query_id value: list of pairs in the following format:(doc_id,score) title_weight: float, for weighted
    average utilizing title and body scores text_weight: float, for weighted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default, N = 3,
    for the topN function.

    Returns:
    -----------
    dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    max_score_title = max(title_scores, key=lambda x: x[1])[1] if len(title_scores) != 0 else 1
    max_score_body = max(body_scores, key=lambda x: x[1])[1] if len(body_scores) != 0 else 1
    max_score_anchor = max(anchor_scores, key=lambda x: x[1])[1] if len(anchor_scores) != 0 else 1

    # Get all candidate doc_ids
    title_scores, body_scores, anchor_scores = dict(title_scores), dict(body_scores), dict(anchor_scores)
    all_candidate_docs = set(title_scores.keys()) | set(body_scores.keys()) | set(anchor_scores.keys())
    relevant_page_ranks = []
    relevant_page_views = []
    for wiki_id in all_candidate_docs:
        wiki_id = str(wiki_id)
        relevant_page_ranks += [page_rank[wiki_id]] if wiki_id in page_rank else []
        relevant_page_views += [page_views[wiki_id]] if wiki_id in page_views else []

    max_page_ranks = max(relevant_page_ranks) if len(relevant_page_ranks) != 0 else 1
    max_page_views = max(relevant_page_views) if len(relevant_page_views) != 0 else 1

    merged_score_dict = {}

    for doc_id in all_candidate_docs:
        # calculate scores
        try:
            page_rank_score = page_rank[str(doc_id)] * pr_weight / max_page_ranks
            page_view_score = page_views[str(doc_id)] * pv_weight / max_page_views
            title_score = title_weight * next((s for d, s in title_scores.items() if d == doc_id), 0) / max_score_title
            body_score = text_weight * next((s for d, s in body_scores.items() if d == doc_id), 0) / max_score_body
            anchor_score = anchor_weight * next((s for d, s in anchor_scores.items() if d == doc_id), 0) / max_score_anchor
            merged_score = title_score + body_score + anchor_score + page_rank_score + page_view_score
            merged_score_dict[doc_id] = merged_score
        except KeyError:
            continue

    merged_scores = sorted([(doc_id, score) for doc_id, score in merged_score_dict.items()],
                           key=lambda x: x[1], reverse=True)[:n]
    return merged_scores
