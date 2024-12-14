import re
import json 
import os 

from tqdm import tqdm

class GeneralUtility():
    def get_max_length(self):
        listfile = []
        for item in os.listdir('./dataset'):
            if os.path.isfile(os.path.join('dataset', item)):
                if re.search('parsed', item):
                    listfile.append(os.path.join('dataset', item)) 
                    
        info = []

        for dataset in tqdm(listfile, desc='Getting context length per dataset'):
            content = None 
            maxl = 0
            minl = 0
            maxs = None 
            mins = None

            with open(dataset) as f:
                content = json.load(f)

                maxl = len(str(content[0]['context']))
                minl = len(str(content[0]['context']))
                maxs = str(content[0]['context'])
                mins = str(content[0]['context'])

                for item in content:
                    ctx = str(item['context'])
                    ctxlen = len(str(item['context']))

                    if ctxlen < minl:
                        minl = ctxlen
                        mins = ctx

                    if ctxlen > minl:
                        maxl = ctxlen
                        maxs = ctx

            row = {
                'dataset': dataset,
                'max_content': maxs,  
                'max_length': maxl, 
                'min_content': mins, 
                'min_length': minl
            }

            info.append(row)

        with open('./eval/content/content_length.json', 'w') as f:
            json.dump(info, f, indent=4)