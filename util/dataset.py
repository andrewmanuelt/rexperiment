import os
import re 
import json
import pandas as pd
import unidecode

from tqdm import tqdm
from sklearn.model_selection import KFold

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DatasetUtility():
    def _parse_answer(self, answer, code = None) -> str:   
        if code == 'covid':
            return answer
        
        strjson = str(answer).replace('"', "'").replace("'r'", '"r"').replace("'n'", '"n"').replace(": '", ': "').replace("', ", '", ')
        strjson = strjson.replace("['", '["').replace("']", '"]').replace('", \'', '", "')
        obj = json.loads(strjson)
        
        if len(obj['n']) > 1:
            obj = " dan ".join(obj['n'])
            return obj
        else:
            obj = ["".join(word) for word in obj['n']]
            return " ".join(obj)
        
    def _parse_context(self, context):
        context = re.sub(r'\[+[0-9]+]', '', context)
        context = re.sub(r'\[+[0-9]+\, +[0-9]{2}\]', '', context)
        context = re.sub(r'https?://\S+|www\.\S+', '', context)
        context = re.sub(r'^https?:\/\/.*[\r\n]*', '', context)
        context = re.sub(r'^http?:\/\/.*[\r\n]*', '', context)
        context = re.sub(r'[0-9{4}]+\-[0-9]{1,2}\-[0-9]{1,2}', ' ', context)
        context = str(context).replace('http//:', '') 
        context = str(context).split(' LATAR BELAKANG: ')
        
        if len(context) > 1:
            context = context[1]
        else:
            context = context[0]
            
        context = str(context).replace('ATAU: ', '')
        context = str(context).replace('KESIMPULAN: ', '')
        context = str(context).replace('METODE DAN TEMUAN: ', '')
        context = str(context).replace('LATAR BELAKANG: ', '')
        context = re.sub(r"\(\s*(.*?)\s*\)", r"(\1)", context)
        context = re.sub(r"(\d),\s+(\d)", r"\1,\2", context)
        context = re.sub(r"(?<=\d)\.(?=\d)", ",", context)
        
        group = []
        for row in context.split('.'):
            group.append(row)
        
        return ". ".join(group)
    
    def _split_dataset(self, k, src, dst_folder):
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder, exist_ok=True)
            
        df = pd.read_json(src) 
        
        fold = KFold(n_splits=k)
        
        index = 1
        for train_index, test_index in fold.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            
            train.to_csv(f"{dst_folder}/train_{index}.csv")
            test.to_csv(f"{dst_folder}/test_{index}.csv")
            
            index = index + 1

        for f in os.listdir(dst_folder):
            try:
                path = f"{dst_folder}/{f}"
                collection = []
                temp = pd.read_csv(path)
                types = str(f).split('_')[0]
                index = str(str(f).split('_')[1]).split('.')[0]
                
                for _, row in temp.iterrows():
                    row = {
                        'question': unidecode.unidecode(row['question']),
                        'answer': unidecode.unidecode(row['answer']),
                        'context': unidecode.unidecode(row['context'])
                    }
                    collection.append(row)
                    
                with open(f"{dst_folder}/{types}_{index}.json", 'w') as f:
                    json.dump(collection, f, indent=4)
                
                os.remove(path)
            except:
                return None
    
    def _clean_context_berita(self, context: str):
        context = re.sub(r'[a-zA-Z]{2,}\,\sKOMPAS.TV\s-\s', '', context)
        
        return context
      
    def _parsing_dataset(self, config: str):
        with open(config[1]) as f:
            datas = json.load(f)
        
        dataset = []
        for item in tqdm(datas, desc=f"parsing {config[0]} dataset"):
            row = {
                'question': item['question'],
                'answer': self._parse_answer(item['answer'], config[0]),
            }
            
            if config[0] == 'berita':
                row['context'] = self._clean_context_berita(item['context'])
                row['category'] = item['category']
            else:
                row['context'] = self._parse_context(item['context'])
            
            dataset.append(row)
            
        with open(config[2], 'w') as f:
            json.dump(dataset, f, indent=4)
        
        if config[0] == 'covid':
            self._split_dataset(5, config[2], config[3])
    
    def load_groundtruth(self, path: str):
        
        with open(path) as f:
            data = json.load(f)
        
        collection = []
        for row in data:
            row = {
                'question': row['question'],
                'context': row['context'],
                'groundtruth': row['answer']
            }

            collection.append(row)
            
        return collection

    def dataset_loader(self, path: str, attr = None):
        with open(path) as f:
            data = json.load(f)
        
        return data
    
    def json_to_document(self, json_array: list, chunk_size: int, chunk_overlap: int):
        total_doc = 0

        if chunk_size is not None and chunk_overlap is chunk_overlap is not None:
            collection = []
            for row in tqdm(json_array, desc='wrapping json to document'):
                total_doc = total_doc + 1

                doc = Document(
                    page_content = row['context'],
                    metadata = {
                        'question': row['question'],
                        'answer': row['answer']
                    }
                )

                collection.append(doc)
            
            return collection, total_doc
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len, 
                is_separator_regex=False
            )
            
            collection = []
            for row in tqdm(json_array, desc='wrapping json to document'):    
                contexts = splitter.split_text(row['context'])

                for context in contexts:
                    total_doc = total_doc + 1

                    doc = Document(
                    page_content = context,
                    metadata = {
                        'question': row['question'],
                        'answer': row['answer']
                    }
                )

                collection.append(doc)
            return collection, total_doc
             
    def run(self):
        datasets = [
            ('complex', './dataset/complex_train.json', './dataset/complex_train_parsed.json'),
            ('complex', './dataset/complex_test.json', './dataset/complex_test_parsed.json'),
            ('single', './dataset/single_train.json', './dataset/single_train_parsed.json'),
            ('single', './dataset/single_test.json', './dataset/single_test_parsed.json'),
            ('covid', './dataset/covid_all.json', './dataset/covid_all_parsed.json', './dataset/covid'),
            # ('berita', './dataset/berita_all.json', './dataset/berita_all_parsed.json'),
        ]
        
        for dataset in datasets:
            self._parsing_dataset(dataset)