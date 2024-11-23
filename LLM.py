from openai import OpenAI

class QwenVllm:
    '''
    This is an openai module template for the Qwen model. 
    must receive the api_key, base_url
    
    '''
    def __init__(self, api_key, base_url, model_id = "Qwen2.5-72B-Instruct"):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key = self.api_key,base_url = self.base_url)

    def inference(self, message, max_length=512):
        chat_completion = self.client.chat.completions.create(
            messages=message,
            model=self.model_id,
            max_tokens=max_length
        )
        return chat_completion.choices[0].message.content



# code for running the model locally directly using the transformers library
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch    
# class Qwen:
#     '''
#     Singleton class for loading and inferencing the model
    
#     inference(messages, max_tokens=512) => (unmasked_response, masked_response)

#     '''
#     _instance = None 
#     _is_initialized = False 

#     def __new__(cls):
#         #singleton pattern
#         if cls._instance is None: 
#             cls._instance = super().__new__(cls)
#         return cls._instance 

#     def __init__(self):
#         #initialize the model
#         self.model_name = "Qwen/Qwen2.5-72B-Instruct" 
#         if not self._is_initialized:
#             print("Loading... model_id:", self.model_name) 
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name, 
#                 torch_dtype = torch.bfloat16, 
#                 device_map = "auto"
#             )
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.model_name,
#                 torch_dtype = torch.float16,
#             )
#             print("model prepared...warming up...")

#             self.model.eval()
#             self._warmup()
#             print("model warmed up...")

#             Qwen._is_initialized = True

#     def _warmup(self):
#         #for model warmup
#         warmup_input = "hello"
#         messages = [
#             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#             {"role": "user", "content": warmup_input}
#         ]
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

#         with torch.no_grad():
#             _ = self.model.generate(
#                 **model_inputs,
#                 max_new_tokens = 1,
#             )

#     def inference(self, messages, max_length=512): 
#         '''
#         Inference the model with the given messages
        
#         return: unmasked response, masked response

#         '''
#         #get the text from the messages
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         #tokenize the input texts
#         model_inputs = self.tokenizer(
#             [text],
#             return_tensors="pt", 
#         ).to(self.model.device) 
#         #generate the response
#         with torch.no_grad():
#             generated_ids = self.model.generate(
#                 **model_inputs,
#                 max_new_tokens = max_length,
#             ) 
#         #decode the response
#         unmasked_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         #get the masked response
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]
#         masked_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#         return unmasked_response, masked_response