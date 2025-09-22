from datasets import load_dataset
ds = load_dataset("MMB-25/knowledge","pin_p", split="test")
print(ds.features)
