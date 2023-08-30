import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
import scann
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader


def get_default_config():
    return {
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'train': {
            'epochs': 20,
            'batch_size': 16,
            'loss': losses.MultipleNegativesRankingLoss
        },
        'mappings': None,
        'db_embeddings': None,
        'query_embeddings': None,
        'db_recs': None,
        'search': {
            'candidates_len': 30,
            'num_leaves': 200,
            'num_leaves_to_search': 50
        },
        'description_fn': None
    }


class Searcher():
    def __init__(self, config):
        self.is_cuda = torch.cuda.is_available()
        self.config = self.apply_config(get_default_config(), config)
        self.db_recs = self.config.get('db_recs')
        self.db_embeddings = self.config.get('db_embeddings')
        self.query_embeddings = self.config.get('query_embeddings')
        self.mappings = self.config.get('mappings')

        model_name = self.config.get('model_name')
        print(f'Loading {model_name}...')
        if self.is_cuda:
            self.model = SentenceTransformer(model_name, device='cuda:0')
        else:
            self.model = SentenceTransformer(model_name)

        self.get_description = config.get('description_fn')
        self.engine = None
        self.assert_required()

    def encode(self, descriptions):
        encodings = []
        batch_size = 1024
        for idx in range(0, len(descriptions), batch_size):
            if idx % 2048:
                print(f'Generating embeddings for {idx} descriptions')
            encodings.extend(self.model.encode(
                descriptions[idx: idx + batch_size]))
        return encodings

    def apply_config(self, dc, config):
        def_config = dc.copy()
        for key, value in config.items():
            if key in def_config and isinstance(def_config[key], dict) and isinstance(value, dict):
                def_config[key] = self.apply_config(def_config[key], value)
            else:
                def_config[key] = value

        return def_config

    def init_search(self):
        # init scaNN and run search
        print("Init search engine...")
        cand_len = self.config.get('search', {}).get('candidates_len')
        num_leaves = self.config.get('search', {}).get('num_leaves')
        num_leaves_to_search = self.config.get(
            'search', {}).get('num_leaves_to_search')
        print(cand_len, num_leaves, num_leaves_to_search)

        self.engine = scann.scann_ops_pybind.builder(self.db_embeddings_norm, cand_len, "dot_product").tree(num_leaves=num_leaves,
                                                                                                            num_leaves_to_search=num_leaves_to_search, training_sample_size=250000).score_ah(
            2, anisotropic_quantization_threshold=0.2).reorder(num_leaves_to_search).build()

    def embed_and_search(self, query_test_recs):
        # Embded all database records
        if not self.db_recs:
            raise Exception('Product records need to be set.')
        if not self.db_embeddings:
            print(
                f'Generate descriptions for all {len(self.db_recs)} products')
            db_descriptions = [self.get_description(r) for r in self.db_recs]
            self.db_embeddings = self.encode(db_descriptions)

            db_embeddings_np = np.stack(self.db_embeddings, axis=0)
            self.db_embeddings_norm = db_embeddings_np / \
                np.linalg.norm(db_embeddings_np, axis=1)[:, np.newaxis]

        if not self.query_embeddings:
            # Embded test query records
            if not query_test_recs:
                raise Exception(
                    'Query records need to be passed to the function.')

            print('Generate descriptions for all queries')
            query_descriptions = [
                self.get_description(q) for q in query_test_recs]
            self.query_embeddings = self.encode(query_descriptions)

            self.query_embeddings_np = np.stack(self.query_embeddings, axis=0)

        self.search()

    def search(self):
        if not self.engine:
            self.init_search()

        print(
            f"Searching to {self.config.get('search', {}).get('candidates_len')} candidates for test queries...")
        neighbors, distances = self.engine.search_batched_parallel(self.query_embeddings_np,
                                                                   leaves_to_search=self.config.get(
                                                                       'search', {}).get('candidates_len'),
                                                                   pre_reorder_num_neighbors=self.config.get('search', {}).get('num_leaves_to_search'))
        self.neighbors = neighbors
        self.distances = distances
        print(f'Neighbors len: {len(neighbors)}')
        return self.neighbors, self.distances

    def fine_tune_embeddings(self, train_pairs):
        train_examples = [InputExample(texts=pair) for pair in train_pairs]

        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=self.config.get('train', {}).get('batch_size'))
        train_loss = self.config.get('train', {}).get('loss')(model=self.model)
        self.model.fit(train_objectives=[
                       (train_dataloader, train_loss)], epochs=self.config.get('train', {}).get('epochs'))

    def get_predictions_for_top_ks(self, k=None):
        k = k if k else self.config.get('search', {}).get('candidates_len')
        predictions = [{} for i in range(len(self.neighbors))]
        top_ks = [i for i in range(1, k + 1)]
        for top_k in top_ks:
            for i, neighbor in enumerate(self.neighbors):
                p = self.mappings[i]['source_id'] in [
                    self.db_recs[idx]['subject_id'] for idx in neighbor][:top_k]
                predictions[i][str(top_k)] = p

        return predictions

    def get_metrics(self, predictions, k=None):
        k = k if k else self.config.get('search', {}).get('candidates_len')
        top_ks = [i for i in range(1, k + 1)]

        y_true = [a['matching'] for a in self.mappings]
        results = []
        for top_k in top_ks:
            y_pred = [a[str(top_k)] for a in predictions]
            rep = metrics.classification_report(
                y_true, y_pred, output_dict=True)
            results.append(rep)
        return results

    def assert_required(self):
        if not self.model:
            raise Exception("model cannot be undefined")
        if not self.db_recs:
            raise Exception("db_recs cannot be undefined")
        if not self.mappings:
            raise Exception(
                "mappings from source to target cannot be undefined")
        if not self.get_description:
            raise Exception("Description function for records is required")
