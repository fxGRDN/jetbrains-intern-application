from dataclasses import dataclass
from typing import Union
import pandas as pd
from functools import reduce
from typing import Callable
import torch
import json
from operator import itemgetter
import os
import glob
from tqdm import tqdm

@dataclass
class Chunker:
    '''Chunker must implement the split_text method'''

    split_text: Callable[[str], list]

@dataclass
class Embbeder:
    
    embed: Callable[[Union[str, list[str]]], torch.Tensor]


class RetriEval:
    
    def __init__(self, chunker: Chunker, embedding: Embbeder, retrived_chunks: int):
        self.chunker = chunker
        self.embedding = embedding
        self.retrived_chunks = retrived_chunks

        if hasattr(self.chunker, 'split_text') and not callable(self.chunker.split_text):
            raise ValueError("chunker must implement the split_text method")
        
        if hasattr(self.embedding, 'embed') and not callable(self.embedding.embed):
            raise ValueError("embedding must implement the embed method")

    def _retrival(self, chunks, query):
        corpus_embeddings = self.embedding.embed(chunks)
        query_embeddings = self.embedding.embed(query)

        cos_similarities = torch.nn.functional.cosine_similarity(query_embeddings, corpus_embeddings, dim=-1)
        top_k_indices = torch.topk(cos_similarities, self.retrived_chunks).indices
        
        return top_k_indices.cpu().numpy()
    
    def _find_target_in_document(self, document, target):
        start_index = document.find(target)
        if start_index == -1:
            return None
        end_index = start_index + len(target)
        return start_index, end_index

    def _load_data(self):

        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        md_file_paths = glob.glob(os.path.join(f'{repo_path}/data', '*.md'))
        
        questions_df = pd.read_csv(os.path.join(f'{repo_path}/data', 'questions_df.csv'))

        questions_df['references'] = questions_df['references'].apply(json.loads)

        markdown_files = {}
        
        for file_path in md_file_paths:
            filename = os.path.splitext(os.path.basename(file_path))[0]

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            markdown_files[filename] = content
        
        return markdown_files, questions_df

    def evaulate(self):
        '''
        Evaluate the retrival system.

        :return: DataFrame with the metrics "f1", "precision", "recall", "iou"
        :rtype: pd.DataFrame

        '''

        markdown_files, questions_df = self._load_data()
        
        chunked_files = {}
        chunks_range = {}

        for filename in markdown_files:
            chunked_files[filename] = self.chunker.split_text(markdown_files[filename])
            chunks_range[filename] = [self._find_target_in_document(markdown_files[filename], chunk) for chunk in chunked_files[filename]]

        del markdown_files

        metrics = {'precision': [], 'recall': [], 'f1': [], 'iou': []}
        
        for _, row in tqdm(questions_df.iterrows(), desc="Evaluating...", total=questions_df.shape[0]):

            query = row['question']
            corpus_id = row['corpus_id']

            topk_indices = self._retrival(chunked_files[corpus_id], query)

            # We dont want to mess up list types with numpy arrays
            relevant_chunks_range = list(itemgetter(*topk_indices)(chunks_range[corpus_id]))
            
            excerpts_ranges = [(excerpt['start_index'], excerpt['end_index']) for excerpt in row['references']]


            int_len = 0
            exc_len = 0
            relevant_chunks_len = sum([self._range_length(r) for r in relevant_chunks_range])
            unused_ranges = relevant_chunks_range.copy()

            for excerpt_range in excerpts_ranges:

                for chunk_range in relevant_chunks_range:
                    
                    intersection_range = self._intersection(chunk_range, excerpt_range)

                    if intersection_range:
                        unused_ranges = reduce(self._reduce_to_list, [self._left_diffrence(u_range, intersection_range) for u_range in unused_ranges])
                        
                        
                exc_len += self._range_length(excerpt_range)
            
            unused_len = sum([self._range_length(range) for range in unused_ranges])
            int_len = relevant_chunks_len - unused_len

            metrics['precision'].append(int_len / relevant_chunks_len)
            metrics['recall'].append(int_len / exc_len)
            metrics['iou'].append(int_len / (relevant_chunks_len + exc_len - int_len))
            metrics['f1'].append(2 * int_len / (relevant_chunks_len + exc_len))
            
    
        return pd.DataFrame(metrics)
    
    def _intersection(self, range1, range2):

        if range1 is None or range2 is None:
            return None

        start1, end1 = range1
        start2, end2 = range2
        
        return (min(max(start1, start2), end1), max(min(end1, end2), start1))

    def _left_diffrence(self, range1, range2):

        if range1 is None or range2 is None:
            return [None]
        
        start1, end1 = range1
        start2, end2 = range2

        if end1 < start2 or start1 > end2:
            return [range1]
        
        if start2 < start1 and end2 > end1:
            return [None]
        
        if start1 <= start2 and end1 >= end2:
            return [(start1, start2), (end2, end1)]
        
        if end1 < end2:
            return [(start1, start2)]
        
        return [(end1, end2)]

    def _range_length(self, range):

        return range[1] - range[0] if range else 0


    def _reduce_to_list(self, list1, list2):
        if list1 is None:
            return list2
        if list2 is None:
            return list1
        
        return list1 + list2
        
