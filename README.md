# Additional Assignment Description
## Assignment 1 Instructions

### 1. Loading Models from HuggingFace: 

HuggingFace provides a wide array of pre-trained models that can be easily loaded for various tasks like text classification, question-answering, etc. 

To load a model from HuggingFace, you can use the provided model identifier from the HuggingFace model repository. Here’s an example of how to load a model for the SST-2 task using BERT architecture (using `lit_nlp/examples/glue/demo.py`):

```python
_MODELS = flags.DEFINE_list(
    "models",
    [
        "sst2-bert:sst2:aviator-neural/bert-base-uncased-sst2",
    ],
```

This will load a fine-tuned BERT-based model fine-tuned for sentiment classification on the SST-2 dataset. The model is fetched from the HuggingFace repository using the specified identifier (`aviator-neural/bert-base-uncased-sst2`). You can replace the model identifier from the HuggingFace. 

### 2. Non-BERT Model Wrapping: 

If you wish to load a model that does not follow the BERT architecture (e.g., T5, GPT, etc.), you will need to implement a wrapper to make it compatible with your task and LIT framework.

In the example, models are loaded using the `MODELS_BY_TASK` mapping, which points to task-specific model classes (e.g., `SST2Model`). If your desired model architecture (e.g., a T5 model) is not directly supported, you will need to:
- Define a new class that wraps the architecture of your chosen model.
- Ensure that the model's inputs, outputs, and processing steps are compatible with the task (like `SST-2`, etc.). More examples code you could find in `lit_nlp/examples/glue/models.py`.

``` python
class T5SST2Wrapper:
    def def __init__(self, model_name_or_path="t5-small", **config_kw):
        self.config = AutoConfig.from_pretrained(model_name_or_path, **config_kw)
        self._load_model(model_name_or_path)
        self._lock = threading.Lock()
        
    def _load_model(self, model_name_or_path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        .....

```

After defining your wrapper, you can load the model as follows:

```python
models['t5-sst2'] = T5SST2Wrapper('path/to/your/model')
```

### 3. Running the Demo: 

After loading the models, you can start the LIT demo server by running the following command:
```bash 
python -m lit_nlp.examples.glue.demo --port=5432
```











