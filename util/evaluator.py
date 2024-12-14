import re 
import sys
import json
import time
import torch
import evaluate

from util.embedding import Embedding
from util.database import VectorDatabase
from util.dataset import DatasetUtility

from tqdm import tqdm
from collections import Counter
from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score

class ProccessEvaluator():
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        
    def breaking(self, variable):
        print(variable)
        sys.exit(1)

    def meteor(self, candidate: str, reference: str) -> float:
        candidate = re.sub(r'\W', ' ', str(candidate)).lower().split()
        reference = re.sub(r'\W', ' ', str(reference)).lower().split()
        
        return meteor_score([reference], candidate)
    
    def bertscore(self, candidate: str, reference: str, evaluator):
        candidate = re.sub(r'\W', ' ', str(candidate)).lower()
        reference = re.sub(r'\W', ' ', str(reference)).lower()

        candidate = [candidate]
        reference = [reference]

        prec, recl, f1 = evaluator.score(candidate, reference)
        
        prec = str(prec.mean()).replace('tensor(', '').replace(')', '')
        recl = str(recl.mean()).replace('tensor(', '').replace(')', '')
        f1 = str(f1.mean()).replace('tensor(', '').replace(')', '')

        return float(prec), float(recl), float(f1)
    
    def rouge(self, candidate: str, reference: str, evaluator):
        candidate = re.sub(r'\W', ' ', str(candidate)).lower()
        reference = re.sub(r'\W', ' ', str(reference)).lower()

        candidate = [candidate]
        reference = [reference]
        
        score = evaluator.compute(
            predictions=candidate, 
            references=reference, 
        )
        
        return score['rougeL']
    
    def f1_score(self, candidate, reference):
        candidate = re.sub(r'\W', ' ', str(candidate)).lower()
        reference = re.sub(r'\W', ' ', str(reference)).lower()

        candidate_tokens = candidate.split()
        reference_tokens = reference.split()
        
        candidate_counter = Counter(candidate_tokens)
        reference_counter = Counter(reference_tokens)
        overlap = sum((candidate_counter & reference_counter).values())
        
        precision = overlap / len(candidate_tokens)
        recall = overlap / len(reference_tokens)
        
        if precision + recall == 0:
            return 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1

    def _document_storing(self, config, embedding_function, chunk_size, chunk_overlap, k):
        print("Begin storing document for this configuration: ")
        print(f"chunk size: {chunk_size} \nchunk overlap: {chunk_overlap} \ntop-k document: {k}")

        start_time = time.perf_counter()
    
        du = DatasetUtility()
        train_dataset = du.dataset_loader(
            path = config['train_dataset']
        )
        train_dataset, total_doc = du.json_to_document(train_dataset, chunk_size, chunk_overlap)
    
        store_folder_path = f"{config['folder_path']}_{chunk_size}_{chunk_overlap}_{k}",
        store_index_name = f"{config['name']}_{chunk_size}_{chunk_overlap}_{k}"

        store_db = VectorDatabase(
            folder_path = store_folder_path,
            index_name = store_index_name,
            embedding_function = embedding_function,
        )
        store_client = store_db.store_client()   
        store_db.store(store_client, train_dataset)

        collection = {
            'dataset': config['train_dataset'],
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap, 
            'top-k': k, 
            'num_of_docs': total_doc
        }

        with open(f"./evaluation/hyperparameter/store_{store_index_name}.json", 'w') as f:
            json.dump(collection, f, indent=4)

        end_time = time.perf_counter()
        print(f"Store has been completed in {end_time - start_time}")

        del loader 
        del store_db
        del store_client
    
    def hyperparameter_grid(self, config):
        collection = []

        embedding = Embedding()
        ef = embedding.load_embedding_function()

        for cfg in config: 
            for cs in cfg['chunk_size']:
                for co in cfg['chunk_overlap']:
                    for tk in cfg['top_k']:
                        self._document_storing(
                            config = cfg,
                            embedding_function = ef, 
                            chunk_size = cs, 
                            chunk_overlap = co, 
                            k = tk
                        )        
        
        bertscore = BERTScorer(batch_size=128, nthreads=12, device=self.device, model_type='bert-base-uncased')
        rougescore = evaluate.load('rouge')

        for cfg in config: 
            for cs in cfg['chunk_size']:
                for co in cfg['chunk_overlap']:
                    for tk in cfg['top_k']:
                        recap = self._do_hyperparameter(ef, cfg, cs, co, tk, bertscore, rougescore)

                        collection.append(recap)

        with open(f"./evaluation/hyperparameter/recap_{cfg['name']}.json", 'w') as f:
            json.dump(collection, f, indent=4)

    def _do_hyperparameter(self, embedding_function, config, chunk_size, chunk_overlap, k, bertscore, rougescore):
        print("Begin hyperparameter tuning for this configuration: ")
        print(f"chunk size: {chunk_size} \nchunk overlap: {chunk_overlap} \ntop-k document: {k}")
        start_time = time.perf_counter()
        
        load_db = VectorDatabase(
            folder_path = f"{config['folder_path']}_{chunk_size}_{chunk_overlap}_{k}",
            index_name = f"{config['name']}_{chunk_size}_{chunk_overlap}_{k}",
            embedding_function = embedding_function,
        )
        load_client = load_db.load()

        utility = DatasetUtility()
        groundtruth_data = utility.load_groundtruth(path = config['test_dataset'])
        
        parent_result_collection = []
        result_collection = []
        details_collection = []

        total_l2_mean = [] 
        total_meteor_mean_context = []
        total_rougel_mean_context = []
        total_bertscore_mean_precision_context = []
        total_bertscore_mean_recall_context = [] 
        total_bertscore_mean_f1_context = []

        report_full_path = f"./evaluation/hyperparameter/result_{config['name']}_{chunk_size}_{chunk_overlap}_{k}.json"

        for row in tqdm(groundtruth_data, desc='Searching context by retriever'):
            print('\n')
            results = load_client.similarity_search_with_score(
                k = k,
                query = row['question']    
            )
            
            if results is None or len(results) == 0:
                self.breaking('No data collected')

            l2_mean = [] 
            meteor_mean_context = []
            rougel_mean_context = []
            bertscore_precision_mean_context = []
            bertscore_recall_mean_context = [] 
            bertscore_f1_mean_context = []

            retrieved_context = []
            for result in results:
                candidate = result[0].page_content
                reference_context = row['context']
                
                retrieved_context.append(reference_context)
                
                meteor_context = self.meteor(candidate=candidate, reference=reference_context)
                rouge_context = self.rouge(candidate=candidate, reference=reference_context, evaluator=rougescore)
                bertscore_p_context, bertscore_r_context, bertscore_f1_context = self.bertscore(candidate=candidate, reference=reference_context, evaluator=bertscore)

                l2_mean.append(result[1])
                meteor_mean_context.append(meteor_context)
                rougel_mean_context.append(rouge_context)
                bertscore_precision_mean_context.append(bertscore_p_context)
                bertscore_recall_mean_context.append(bertscore_r_context)
                bertscore_f1_mean_context.append(bertscore_f1_context)

                details_collection.append(retrieved_context)

            result = {
                'question': row['question'],
                'context': row['context'],
                'groundtruth': row['groundtruth'],
                'mean_l2': str(sum(l2_mean) / len(l2_mean)),
                'mean_meteor_context': str(sum(meteor_mean_context) / len(meteor_mean_context)),
                'mean_rougel_context': str(sum(rougel_mean_context) / len(rougel_mean_context)),
                'mean_bertscore_precision_context': str(sum(bertscore_precision_mean_context) / len(bertscore_precision_mean_context)),
                'mean_bertscore_recall_context': str(sum(bertscore_recall_mean_context) / len(bertscore_recall_mean_context)),
                'mean_bertscore_f1_context': str(sum(bertscore_f1_mean_context) / len(bertscore_f1_mean_context)),
                'details': details_collection,
            }
            result_collection.append(result)

            total_l2_mean.append(sum(l2_mean) / len(l2_mean))
            total_meteor_mean_context.append(sum(meteor_mean_context) / len(meteor_mean_context))
            total_rougel_mean_context.append(sum(rougel_mean_context) / len(rougel_mean_context))
            total_bertscore_mean_precision_context.append(sum(bertscore_precision_mean_context) / len(bertscore_precision_mean_context))
            total_bertscore_mean_recall_context.append(sum(bertscore_recall_mean_context) / len(bertscore_recall_mean_context)) 
            total_bertscore_mean_f1_context.append(sum(bertscore_f1_mean_context) / len(bertscore_f1_mean_context))

        parent_result_collection = {
            'dataset': config['name'],
            'chunk_size': str(chunk_size),
            'chunk_overlap': str(chunk_overlap),
            'top_k': k,
            'mean_total_l2': str(sum(total_l2_mean) / len(total_l2_mean)),
            'mean_total_meteor_context': str(sum(total_meteor_mean_context) / len(total_meteor_mean_context)),
            'mean_total_rougel_context': str(sum(total_rougel_mean_context) / len(total_rougel_mean_context)),
            'mean_total_bertscore_precision_context': str(sum(total_bertscore_mean_precision_context) / len(total_bertscore_mean_precision_context)),
            'mean_total_bertscore_recall_context': str(sum(total_bertscore_mean_recall_context) / len(total_bertscore_mean_recall_context)),
            'mean_total_bertscore_f1_context': str(sum(total_bertscore_mean_f1_context) / len(total_bertscore_mean_f1_context)),
        }
        
        with open(report_full_path, 'w') as f:
            json.dump(parent_result_collection, f, indent=4)

        end_time = time.perf_counter()

        del load_db 
        del load_client
        del loader

        print(f"Hyperparameter completed in {round(end_time - start_time, 2)} second")

        return parent_result_collection