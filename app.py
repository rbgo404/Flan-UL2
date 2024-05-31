from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

class InferlessPythonModel:
    def initialize(self):
        model_id = "google/flan-ul2"  # Specify the model repository ID
        self.model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto", load_in_8bit=True)                                                                 
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    def infer(self,inputs):
        input_string = inputs["prompt"]
        inputs = self.tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(inputs, max_length=200)
        result_output = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
        return {'generated_result': result_output}

    def finalize(self):
        pass
