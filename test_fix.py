# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import mteb

from datasets import load_dataset

category_map = {
    "Art": "Art", "Art_Theory": "Art", "Design": "Art", "Music": "Art",
    "Sociology": "Humanities", "Literature": "Humanities", "History": "Humanities", "Psychology": "Humanities",
    "Clinical_Medicine": "Medicine", "Diagnostics_and_Laboratory_Medicine": "Medicine", "Basic_Medical_Science": "Medicine", "Pharmacy": "Medicine",
    "Biology": "Science", "Chemistry": "Science", "Geography": "Science", "Agriculture": "Science",
    }

tasks = mteb.get_tasks(tasks=["KnowledgeAny2AnyRetrieval"])
model_name = "royokong/e5-v"

model = mteb.get_model(model_name=model_name, device="cuda:1")

encode_kwargs = {
    "batch_size": 1,
    "show_progress_bar": True,
    "convert_to_tensor": True,
}

# Test that the model loading and initial processing works
print("Testing model loading and query processing...")

# Load the task to test
task = tasks[0]
task.load_data(text_vision=False)

# Get first few queries to test
queries = task.queries["test"][:3]  # Just test first 3 queries
print(f"Testing with {len(queries)} queries")

# Test the model can process them
try:
    # Test text embeddings
    texts = [q["text"] for q in queries if q["text"]]
    images = [q["image"] for q in queries if q["image"]]
    
    print(f"Processing {len(texts)} texts and {len(images)} images...")
    
    # This should work now without the image token error
    embeddings = model.get_fused_embeddings(
        texts=texts,
        images=images,
        task_name="KnowledgeAny2AnyRetrieval",
        prompt_type=mteb.PromptType.query,
        batch_size=1
    )
    
    print(f"Success! Generated embeddings with shape: {embeddings.shape}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()