# Loading Dataset
from datasets import load_dataset 

# this dataset uses the new Image feature :)
dataset = load_dataset("layoutlmv3.py", trust_remote_code=True)
print(dataset)
