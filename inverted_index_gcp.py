import io
import os
from collections import defaultdict, Counter
import pickle
import hashlib
from pathlib import Path
from itertools import count
from google.cloud import storage
from contextlib import closing
from operator import itemgetter

# --- Global Variables --- #

# --- Block Size --- #
BLOCK_SIZE = 1999998

# --- Bucket Name --- #
BUCKET_NAME = "bucket2121"

# --- Reading/Writing params --- #
NUM_BUCKETS = 124
TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


# --- hash function --- #
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


# --- token 2 bucket_id --- #
def token2bucket_id(token):
    return int(_hash(token), 16) % NUM_BUCKETS


# --- MultiFileWriter --- #
class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()


# --- MultiFileReader --- #
class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(BUCKET_NAME)

    def read(self, locs, n_bytes, base_dir):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                blob = self.bucket.blob(f"{base_dir}/{f_name}")
                self._open_files[f_name] = io.BytesIO(blob.download_as_string())
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class InvertedIndex:
    def __init__(self):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        self.DL = {}
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs.
        self.posting_locs = defaultdict(list)


    @staticmethod
    def read_index(bucket_name, base_dir, name):
        os.system(f"gsutil cp gs://{bucket_name}/{base_dir}/{name}.pkl .")
        with open(f'{name}.pkl', 'rb') as f:
            res = pickle.load(f)
            return res


    def read_posting_list(self, term, comp):
        posting_list = []
        with closing(MultiFileReader()) as reader:
            if term in self.posting_locs.keys() and self.df.keys():
                locs = self.posting_locs[term]
                # read a certain number of bytes into variable b
                b = reader.read(locs, self.df[term] * TUPLE_SIZE, comp)
                # convert the bytes read into `b` to a proper posting list.
                for i in range(self.df[term]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                return posting_list
        return []

    def posting_lists_iter(self, query, comp):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        reader = MultiFileReader()
        try:
            for w in query:
                if w in self.posting_locs.keys():
                    locs = self.posting_locs[w]
                    # read a certain number of bytes into variable b
                    b = reader.read(locs, self.df[w] * TUPLE_SIZE, comp)
                    posting_list = []
                    # convert the bytes read into `b` to a proper posting list.
                    for i in range(self.df[w]):
                        doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                        tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                        posting_list.append((doc_id, tf))
                    yield w, posting_list
        finally:
            reader.close()

    def write(self, base_dir, name):
        """ Write the in-memory index to disk and populate the `posting_locs`
            variables with information about file location and offset of posting
            lists. Results in at least two files:
            (1) posting files `name`XXX.bin containing the posting lists.
            (2) `name`.pkl containing the global term stats (e.g. df).
        """
        # POSTINGS
        self.posting_locs = defaultdict(list)
        with closing(MultiFileWriter(base_dir, name)) as writer:
            # iterate over posting lists in lexicographic order
            for w in sorted(self._posting_list.keys()):
                self._write_a_posting_list(w, writer, sort=True)
        # GLOBAL DICTIONARIES
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def _write_a_posting_list(self, w, writer, sort=False):
        # sort the posting list by doc_id
        pl = self._posting_list[w]
        if sort:
            pl = sorted(pl, key=itemgetter(0))
        # convert to bytes
        b = b''.join([(int(doc_id) << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                      for doc_id, tf in pl])
        # write to file(s)
        locs = writer.write(b)
        # save file locations to index
        self.posting_locs[w].extend(locs)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        # TODO check
        state = self.__dict__.copy()
        del state['_posting_list']
        return state


    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()
