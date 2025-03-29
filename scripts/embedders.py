from RetriEval import Embbeder
import torch
import torch.nn.functional as F

class HFMiniLM(Embbeder):
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def embed(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        chunks_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        return F.normalize(chunks_embeddings, p=2, dim=1)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class HFMPNet(Embbeder):
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")
        self.model = AutoModel.from_pretrained("sentence-transformers/multi-qa-mpnet-base-dot-v1")

    def _cls_pooling(self, model_output):
        return model_output.last_hidden_state[:,0]

    def embed(self, texts):
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self._cls_pooling(model_output)

        return embeddings