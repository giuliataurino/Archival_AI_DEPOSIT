# Load model directly
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
 # Changed to correct model class
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv

def load_image(image_path):
    """
    Helper function to load and preprocess images for Qwen 2-VL to process
    """
    return Image.open(image_path).convert("RGB")

def prompt_model(image_path, processor, model, prompt):
    """
    Function to process a single image and generate a title
    Returns:
    response - response generated from Qwen based on the provided prompt
    """
    try:
        image = load_image(image_path) #Apply preprocessing onto the given image
        
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
        
        #Prepare the inputs using Qwen's chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        #Process inputs into tensor format
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to("cuda") # Convert input image into tensors to be fed into Qwen model
        
        #Generate the actual description
        generated_ids = model.generate(
            inputs.input_ids,  # Changed to match example
            max_length=150,    # Max length of response
            min_length=30,     # Min length of response
            num_beams=5,       # Track top 5 token possibilities at each step
            temperature=0,     # Ensures model always picks most likely token and thus output is most consistent
            no_repeat_ngram_size=2,  # No repeated 2-word phrases
            early_stopping=True      # Stops after model finds natural ending
        )
        
        #Process the generated tokens into text
        response = processor.batch_decode(
            generated_ids,  # Simplified to match example
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0] #Tensors decoded back into text output
        
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
            title_prompt = "Generate a title based on the given image"
            title = prompt_model(image, processor, model, title_prompt)
            
            abstract_prompt = "Generate an abstract based on the given image"
            abstract = prompt_model(image, processor, model, abstract_prompt)
            
            writer.writerow([image.name, title, abstract])

if __name__ == "__main__":
    # Define image paths
    image_front_dir = Path("Test_Image_Sets/Recto") #Make sure to remove the extra Recto and Verso folders 
    images_front = sorted(list(image_front_dir.iterdir())) #Gets every image in the directory as a list that can be iterated through
    image_back_dir = Path("Test_Image_Sets/Verso")
    images_back = sorted(list(image_back_dir.iterdir()))

    print(f"Found {len(images_front)} front images and {len(images_back)} back images")
    print(f"First front image: {images_front[0]}")
    print(f"First back image: {images_back[0]}")

    #Model initialization
    model_path = "Qwen/Qwen2-VL-7B-Instruct" #Switch to 72B model on GPU Cluster
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto")


    processor = AutoProcessor.from_pretrained(model_path)
   
    print("Model loaded successfully!")
    
    
    
    run_only_front(images_front, processor, model)