from sentence_transformers import SentenceTransformer, losses, InputExample
import scann
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader

def_config = {
    'model_name': 'sentence-transformers/all-mpnet-base-v2',
    'train': {
        'epochs': 20,
        'batch_size': 16,
        'loss': losses.MultipleNegativesRankingLoss
    },
    'db_embeddings': None,
    'query_embeddings': None,
    'db_recs': None,
    'candidates_len': 10,
    'num_leaves': 200,
    'num_leaves_to_search': 50
}

class Searcher():
    def __init__(self, config):
        self.config = config
        self.db_recs = self.get_config('db_recs')
        self.db_embeddings = self.get_config('db_embeddings')
        self.query_embeddings = self.get_config('query_embeddings')
        self.mappings = self.get_config('mappings')
        self.model = SentenceTransformer(self.get_config('model_name'))
        self.get_description = config['get_description']

        self.assert_required()

    def get_config(self, key):
        return self.config[key] if key in config and config[key] else def_config[key]

    def encode(self, descriptions):
        encodings = []
        if len(descriptions) > 5000:
            batch_size = 1024
            for idx in range(0, len(descriptions), batch_size):
                if idx % 10240:
                    print(f'Generating embeddings for {idx} descriptions')
                encodings.extend(self.model.encode(descriptions[idx : idx + batch_size]))
            return encodings

        return [self.model.encode(descr) for descr in descriptions]

    

    def encode_all(self, query_test_recs):
        # Embded all database records
        if not self.db_embeddings:
            db_descriptions = [self.get_description(r) for r in self.db_recs]
            self.db_embeddings = self.encode(db_descriptions)

        db_embeddings_np = np.stack(self.db_embeddings, axis = 0)
        self.db_embeddings_norm = db_embeddings_np / np.linalg.norm(db_embeddings_np, axis=1)[:, np.newaxis]

        # Embded test query records
        if not self.query_embeddings:
            # query_test_recs = [ buy_id_to_rec_map[a['target_id']] for a in abt_buy_test_recs]
            query_descriptions = [self.get_description(q) for q in query_test_recs]
            self.query_embeddings = self.encode(query_descriptions)

        self.query_embeddings_np = np.stack(self.query_embeddings, axis = 0)

    def init_search():
        #init scaNN and run search
        self.engine = scann.scann_ops_pybind.builder(self.db_embeddings_norm, self.get_config('candidates_len'),
         "dot_product").tree(num_leaves=self.get_config('num_leaves'), 
         num_leaves_to_search=self.get_config('num_leaves_to_search'), training_sample_size=250000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(self.get_config('num_leaves_to_search')).build()

    def search(self):
        if not self.engine:
            self.init_search()
        neighbors, distances = self.engine.search_batched_parallel(self.query_embeddings_np,
            leaves_to_search=self.get_config('num_leaves'),
            pre_reorder_num_neighbors=self.get_config('num_leaves_to_search'))
        self.neighbors = neighbors
        self.distances = distances
        return self.neighbors, self.distances

    def fine_tune_embeddings(self, train_pairs):
        train_examples = [InputExample(texts=pair) for pair in train_pairs]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.get_config('batch_size'))
        train_loss = self.get_config('loss')(model=self.model)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.get_config('epochs')) 

    def get_predictions_for_top_ks(k = self.get_config('candidates_len')):
        predictions = [{} for i in range(len(self.neighbors))]
        top_ks = [i for i in range(1, k + 1)]
        for top_k in top_ks:
            for i, neighbor in enumerate(self.neighbors):
            p = self.mappings[i]['source_id'] in [self.db_recs[idx]['subject_id'] for idx in neighbor][:top_k]
            predictions[i][str(top_k)] = p

        return predictions


    def get_metrics(self, predictions, k=self.get_config['candidates_len']):
        top_ks = [i for i in range(1, k + 1)]

        y_true = [a['matching'] for a in self.mappings]
        results = []
        for top_k in top_ks:
            y_pred = [a[str(top_k)] for a in predictions]
            rep = metrics.classification_report(y_true, y_pred, output_dict=True)
            results.append(rep)
        return results

    def assert_required():
        if not self.model:
            raise Exception("model cannot be undefined")
        if not self.db_recs:
            raise Exception("db_recs cannot be undefined")
        if not self.mappings:
            raise Exception("mappings from source to target cannot be undefined")
        if not self.get_description:
            raise Exception("Description function for records is required")
        