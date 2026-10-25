import os
import time
import torch
from PIL import Image


from llava.model.builder import load_pretrained_model
from utils.get_bbox import get_bbox  
from utils.extract_features import extract_semantic_features
from models.semantic_predictor import SemanticPredictor
from train.train_predictor import load_predictor_weights 


MODEL_PATH = "/root/autodl-tmp/llava-v1.5-7b"
IMG_DIR = "/root/autodl-tmp/datasets/mini_coco/train2017"
NUM_IMAGES = 100

print("loading models...")
tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL_PATH, None, "llava-v1.5-7b")
predictor = load_predictor_weights("checkpoints/predictor.pth") 
predictor.cuda().eval()

files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])[:NUM_IMAGES]

time_yolo = []
time_semantic = []
time_predictor = []
time_llm_generate = []
time_total = []

for i, f in enumerate(files):
    image = Image.open(os.path.join(IMG_DIR, f)).convert("RGB")
    
 
    torch.cuda.synchronize()
    start_pre = time.time()
    
    bboxes = get_bbox(image)                   
    masks = bbox_to_mask(bboxes)               
    sem_feats = extract_semantic_features(image) 
    
    torch.cuda.synchronize()
    elapsed_pre = time.time() - start_pre
    

    image_tensor = process_images([image], image_processor, model.config).half().cuda()
    sem_feats = sem_feats.half().cuda()
    
    torch.cuda.synchronize()
    start_pred = time.time()
    

    with torch.no_grad():
        visual_tokens = model.encode_images(image_tensor)
        selected_tokens = predictor(visual_tokens, sem_feats, masks) 
        
    torch.cuda.synchronize()
    elapsed_pred = time.time() - start_pred
    
 
    question = "Describe this image." 
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    
    torch.cuda.synchronize()
    start_llm = time.time()
    
    with torch.no_grad():
        
        output = model.generate(
            inputs.input_ids, 
            images=selected_tokens, 
            max_new_tokens=50 
        )
        
    torch.cuda.synchronize()
    elapsed_llm = time.time() - start_llm


    time_yolo.append(elapsed_pre)
    time_predictor.append(elapsed_pred)
    time_llm_generate.append(elapsed_llm)
    time_total.append(elapsed_pre + elapsed_pred + elapsed_llm)

    if i % 10 == 0:
        print(f"Image {i:03d}: Pre={elapsed_pre*1000:.1f}ms, Pred={elapsed_pred*1000:.1f}ms, LLM={elapsed_llm*1000:.1f}ms")

print("\n=== Average Latency (ms) ===")
print(f"Preprocessing (YOLO+Semantic): {sum(time_yolo)/len(time_yolo)*1000:.1f} ms")
print(f"Token Predictor:               {sum(time_predictor)/len(time_predictor)*1000:.1f} ms")
print(f"LLM Generation:                {sum(time_llm_generate)/len(time_llm_generate)*1000:.1f} ms")
print(f"Total End-to-End:              {sum(time_total)/len(time_total)*1000:.1f} ms")
