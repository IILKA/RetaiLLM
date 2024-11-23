# RetaiLLM
This is an APAI4011/STAT4011 course project repo. 





## Installation

### Server side Installation 
In this project, we use the vllm frame work to deploy the QwenV2.5 72B instruct model (how to download https://huggingface.co/Qwen/Qwen2.5-72B-Instruct). This approach is faster and more efficient than using the transformers library directly. 

Server installation can be skipped if you want to use existing API. The client side configuration allows flexible options. However, remember that the reliability of models other than QwenV2.5 72B instruct is not tested. 

Here is the code for the server deployment. 

Install the vllm package:
```bash
pip install vllm
```

Start the server using cli: 
```bash
vllm serve <model_path> \ 
    --tensor-parallel-size <num_of_gpu> \
    --host 127.0.0.1 \
    --port 6006 \
    --served-model-name <model_name>\
    --api-key <api_key> \
    --gpu-memory-utilization 0.9 \
    --max_model_len <max_model_len> \
    --enforce-eager \
    --max-num-batched-tokens 8000
```
<model_path>: the path to the model you want to deploy. 
if you are using hugging face model, you should directly hugging face model path. In this project, we use hugging face model QwenV2.5 72B instruct model, and the path is "Qwen/Qwen2.5-72B-Instruct".
<served_model_name>: the model name you expect your client to access your model. It should be consistent with the client side.
<api_key>: the api key you want to use to access the model. It should be consistent with the client side.
<max_model_len>: the maximum length of the model. Generally, setting 2000 is enough for our project. 

sample code: 
```
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 8 \
    --host 127.0.0.1 \
    --port 6006 \
    --served-model-name Qwen2.5-72B-Instruct \
    --api-key APAI4011 \
    --gpu-memory-utilization 0.9 \
    --max_model_len 2000 \
    --enforce-eager \
    --max-num-batched-tokens 8000
```

### Client side Installation 
clone the repo: 
```
git clone https://github.com/IILKA/RetaiLLM/
```

set up the environment: 

```
conda create -f environment.yaml
conda activate retaillm
```










