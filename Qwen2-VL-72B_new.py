# Load model directly
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv
import gc
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

def clear_gpu_memory():
    """
    Clears Pytorch's CUDA cache to free resources for GPU
    """
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Current GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    gc.collect() # Run a garbage collector to collect information to be printed
    print(f"GPU Memory after clearing: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

def load_image(image_path, max_size=512):
     # Load and convert image
        image = Image.open(image_path).convert("RGB")
        
        # Calculate new size maintaining aspect ratio
        width, height = image.size
        ratio = min(max_size/width, max_size/height)
        new_size = (int(width*ratio), int(height*ratio))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        
        # Process image using the model's processor
        processed_image = processor(images=image, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            processed_image = {k: v.cuda() for k, v in processed_image.items()}
            
        return processed_image

def prompt_model(image_path, processor, model, prompt):
    """
    Function to process a single image and generate a title
    Returns:
    response - response generated from Qwen based on the provided prompt
    """
    try:
        image = load_image(image_path) #Apply preprocessing onto the given image
        clear_gpu_memory()
        #Create message format required by Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}, #Add image to message
                    {"type": "text", "text": prompt}   #Add prompt to message
                ]
            }
        ]

        
        #Process inputs into tensor format
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            return_attention_mask = True
        ).to("cuda") # Convert input image into tensors to be fed into Qwen model
        
        #Generate the actual description
        generated_ids = model.generate(
            **inputs,  # Changed to match example
            num_beams=5,       # Track top 5 token possibilities at each step
            temperature=0.1,     # Ensures model always picks most likely token and thus output is most consistent
            no_repeat_ngram_size=2,  # No repeated 2-word phrases
            early_stopping=True      # Stops after model finds natural ending
        )
        
        #Process the generated tokens into text
        response = processor.batch_decode(
            generated_ids,  # Simplified to match example
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip() #Tensors decoded back into text output
        #print(response)
        return response
        
    except Exception as e:
        print(f"Error processing image at {image_path}: {e}")
        return f"ERROR: {str(e)}"
        
    finally:
        if 'inputs' in locals():
            del inputs # Delete input tensor
        if 'generated_ids' in locals():
            del generated_ids
        torch.cuda.empty_cache() #Frees resources for GPU to run more efficiently

def run_only_front(images_front, processor, model):
    #Create output directory if it doesn't exist
    Path("Outputs").mkdir(exist_ok=True)
    
    #Create CSV file that we will write to in the Outputs directory
    with open('Outputs/front_only.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Title', 'Abstract']) #Write CSV headers
        
        for image in tqdm(images_front): # tqdm to show image processing stats
            title_prompt = "Looking at this image, generate a concise, descriptive title. Response should be just the title."
            title = prompt_model(image, processor, model, title_prompt)
            
            clear_gpu_memory()

            abstract_prompt = "Generate a detailed description of this image in 2-3 sentences."
            abstract = prompt_model(image, processor, model, abstract_prompt)
            
            writer.writerow([image.name, title, abstract])

if __name__ == "__main__":
    # Define image paths
    image_front_dir = Path("Test_Image_Sets/Front") #Make sure to remove the extra Recto and Verso folders 
    images_front = sorted(list(image_front_dir.iterdir())) #Gets every image in the directory as a list that can be iterated through
    image_back_dir = Path("Test_Image_Sets/Back")
    images_back = sorted(list(image_back_dir.iterdir()))

    print(f"Found {len(images_front)} front images and {len(images_back)} back images")
    print(f"First front image: {images_front[0]}")
    print(f"First back image: {images_back[0]}")

    #Model initialization
    clear_gpu_memory() #Clears Cuda Cache
    model_path = "Qwen/Qwen2-VL-7B-Instruct" #Switch to 72B model on GPU Cluster
    
    
    #Code to quantize model to run on local 8gb VRAM GPU
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #llm_int8_threshold=6.0,  
    # Controls how outlier weights are handled
    # - Higher value (>6.0): More weights in fp16, better accuracy but more memory
    # - Lower value (<6.0): More weights in int8, less memory but potentially lower accuracy
    # - 6.0 is a good default balance
    
    #llm_int8_has_fp16_weight=True,
    # Keeps some important weights in fp16 precision
    # - True: Critical weights stay in fp16 for better accuracy
    # - False: All weights in int8, saves memory but might affect quality
    # Like keeping the important parts more precise
    
    #llm_int8_enable_fp32_cpu_offload=True
    #Offloads some of the task onto the cpu to fit model on GPU
    )
   
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        quantization_config = quantization_config
    )
    """
    
    LLM Size Chart
    7B parameters:
    32-bit (no quantization): ~28GB
    16-bit (half precision): ~14GB
    8-bit quantization: ~7GB    # Fits in your 8GB RTX 3070
    """
    processor = AutoProcessor.from_pretrained(model_path)
   
    print("Model loaded successfully!")
    run_only_front(images_front, processor, model)