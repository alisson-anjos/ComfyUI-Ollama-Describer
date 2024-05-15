import ollama
from ollama import Client
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64

class OllamaUtil:
    def __init__(self):
        pass
    
    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
    
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    
        image = Image.fromarray(image_np, mode='RGB')
        return image
        
    def image_to_bytes(self, image: Image.Image):
        with BytesIO() as output:
            image.save(output, format="PNG")
            image_bytes = output.getvalue()
        return image_bytes

    def pull_model(self, model, client):
        current_digest, bars = '', {}
        for progress in client.pull(model, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get('status'))
                continue

            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'pulling {digest[7:19]}', unit='B', unit_scale=True)

            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest
            
class LlavaDescriber:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["llava:7b-v1.6", "llava:13b-v1.6", "llava:34b-v1.6"],),
                "api_host": ("STRING", {
                  "default": "http://localhost:11434"
                }),
                "timeout": ("INT", {
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": 300
                }),
                "temperature": ("FLOAT", {
                    "min": 0,
                    "max": 1,
                    "step": 0.1,
                    "default": 0.2
                }),
                "seed_number": ("INT", {
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": 0
                }),
                "max_tokens": ("INT", {
                    "step": 10,
                    "default": 200
                }),
                "keep_model_alive": ("INT", {
                    "step": 1,
                    "default": -1
                }),
                "image": ("IMAGE",),  
                "system_context": ("STRING", {
                    "multiline": True,
                    "default": """You are an assistant who describes the content and composition of images. 
                    Describe only what you see in the image, not what you think the image is about.Be factual and literal. 
                    Do not use metaphors or similes. 
                    Be concise.""",
                    "title":"system"
                }),
                "prompt": ("STRING", {
                    "default": "Return a list of danbooru tags for this image, formatted as lowercase, separated by commas.",
                    "multiline": True
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)

    FUNCTION = "ollama_image_describe"

    OUTPUT_NODE = True

    CATEGORY = "Ollama"

    def ollama_image_describe(self, model, api_host, timeout, temperature, seed_number, max_tokens, keep_model_alive, prompt, system_context, image):
        
        client = Client(api_host, timeout=timeout)
        
        ollama_util = OllamaUtil()
        
        models = [model_l['name'] for model_l in client.list()['models']]
        
        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)
            
        print('System Context: "{}"'.format(system_context))
        
        full_response = ""
                
        print('Converting Tensor to Image')
        img = ollama_util.tensor_to_image(image)
    
        print('Converting Image to bytes')
        img_bytes = ollama_util.image_to_bytes(img)
    
        print('Generating Description from Image')
        full_response =  client.generate(model=model, system=system_context, prompt=prompt, images=[img_bytes], keep_alive=keep_model_alive, stream=False, options={
                'num_predict': max_tokens,
                'temperature': temperature,
                'seed': seed_number
        })
        
        print('Finalized')
        return (full_response['response'], )



class OllamaDescriber:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        
        
        return {
            "required": {
                "model": (["gemma:2b", 
                           "gemma:7b",
                           "llama2:7b", 
                           "llama2:13b", 
                           "llama2:70b", 
                           "llama3:8b", 
                           "llama3:70b",
                           "mistral:7b"],),
                "api_host": ("STRING", {
                  "default": "http://localhost:11434"
                }),
                "timeout": ("INT", {
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": 300
                }),
                "temperature": ("FLOAT", {
                    "min": 0,
                    "max": 1,
                    "step": 0.1,
                    "default": 0.2
                }),
                "seed_number": ("INT", {
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": 0
                }),
                "max_tokens": ("INT", {
                    "step": 10,
                    "default": 200
                }),
                "keep_model_alive": ("INT", {
                    "step": 1,
                    "default": -1
                }),
                "system_context": ("STRING", {
                    "multiline": True,
                    "default": """You are an assistant who describes content about art.
Describe your knowledge of all types of visual art, your objective is to describe in detail what is requested, be concise.
                    """,
                    "title":"system"
                }),
                "prompt": ("STRING", {
                    "default": "In one row, return a list, formatted in lowercase letters, separated by commas, describing Little Red Riding Hood.",
                    "multiline": True
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)

    FUNCTION = "ollama_describe"

    OUTPUT_NODE = True

    CATEGORY = "Ollama"

    def ollama_describe(self, model, api_host, timeout, temperature, seed_number, max_tokens, keep_model_alive, prompt, system_context):
        client = Client(api_host, timeout=timeout)
        ollama_util = OllamaUtil()
        
        models = [model_l['name'] for model_l in client.list()['models']]
        
        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)
            
        print('System Context: "{}"'.format(system_context))
        
        full_response = ""
                
        print('Generating Response')
        full_response =  client.generate(model=model, system=system_context, prompt=prompt, keep_alive=keep_model_alive, stream=False, options={
                'num_predict': max_tokens,
                'temperature': temperature,
                'seed': seed_number
        })
        
        print('Finalized')
        return (full_response['response'], )
    
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""


# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LLaVaDescriber": LlavaDescriber,
    "OllamaDescriber": OllamaDescriber
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LLaVaDescriber": "🌋 LLaVa Describer 🦙",
    "OllamaDescriber": "🦙 Ollama Describer 🦙"
}
