from ollama import Client
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import re

vision_models = ["llava:7b-v1.6-vicuna-q2_K (Q2_K, 3.2GB)",
                 "llava:7b-v1.6-mistral-q2_K (Q2_K, 3.3GB)",
                 "llava:7b-v1.6 (Q4_0, 4.7GB)", 
                 "llava:13b-v1.6 (Q4_0, 8.0GB)", 
                 "llava:34b-v1.6 (Q4_0, 20.0GB)", 
                 "llava-llama3:8b (Q4_K_M, 5.5GB)", 
                 "llava-phi3:3.8b (Q4_K_M, 2.9GB)", 
                 "moondream:1.8b (Q4, 1.7GB)", 
                 "moondream:1.8b-v2-q6_K (Q6, 2.1GB)",
                 "moondream:1.8b-v2-fp16 (F16, 3.7GB)"]

text_models = ["qwen2:0.5b (Q4_0, 352MB)",
                           "qwen2:1.5b (Q4_0, 935MB)",
                           "qwen2:7b (Q4_0, 4.4GB)",
                           "gemma:2b (Q4_0, 1.7GB)", 
                           "gemma:7b (Q4_0, 5.0GB)",
                           "gemma2:9b (Q4_0, 5.4GB)", 
                           "phi3:mini (3.82b, Q4_0, 2.2GB)",
                           "phi3:medium (14b, Q4_0, 7.9GB)",
                           "llama2:7b (Q4_0, 3.8GB)", 
                           "llama2:13b (Q4_0, 7.4GB)", 
                           "llama3:8b (Q4_0, 4.7GB)", 
                           "llama3:8b-text-q6_K (Q6_K, 6.6GB)",
                           "mistral:7b (Q4_0, 4.1GB)"]

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
    
    def image_to_base64(self, image: Image.Image):
        with BytesIO() as output:
            image.save(output, format="PNG")
            image_bytes = output.getvalue()
            image_base64 = base64.b64encode(image_bytes)
        return image_base64
    

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
                
class OllamaImageDescriber:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "model": (vision_models,),
                "custom_model": ("STRING", {
                    "default": ""
                }),
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
                    "max": 10,
                    "step": 0.1,
                    "default": 0.2
                }),
                "top_k": ("INT", {
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "default": 40
                }),
                "top_p": ("FLOAT", {
                    "min": 0,
                    "max": 10,
                    "step": 0.1,
                    "default": 0.9
                }),
                "repeat_penalty":("FLOAT", {
                    "min": 0,
                    "max": 10,
                    "step": 0.1,
                    "default": 1.1
                }),
                "seed_number": ("INT", {
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": 42
                }),
                "max_tokens": ("INT", {
                    "step": 10,
                    "default": 200
                }),
                "keep_model_alive": ("INT", {
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": -1
                }),
                "images": ("IMAGE",),  
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
    RETURN_NAMES = ("result",)

    FUNCTION = "ollama_image_describe"

    OUTPUT_NODE = True

    CATEGORY = "Ollama"

    def ollama_image_describe(self, 
                              model, 
                              custom_model, 
                              api_host,
                              timeout,
                              temperature,
                              top_k,
                              top_p,
                              repeat_penalty, 
                              seed_number,
                              max_tokens,
                              keep_model_alive,
                              prompt,
                              system_context,
                              images):
        
        client = Client(api_host, timeout=timeout)
        
        ollama_util = OllamaUtil()
        
        models = [model_l['name'] for model_l in client.list()['models']]
        
        model = model.split(' ')[0].strip()
        custom_model = custom_model.strip()
        
        model = custom_model if custom_model != "" else model
        
        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)
            
        print('System Context: "{}"'.format(system_context))
        print('Prompt: "{}"'.format(prompt))
        
        full_response = ""
        
        images_base64 = []       
        
        for (batch_number, image) in enumerate(images):
            print('Converting Tensor to Image')
            img = ollama_util.tensor_to_image(image)
            print('Converting Image to bytes')
            img_base64 = ollama_util.image_to_base64(img)
            images_base64.append(str(img_base64, 'utf-8'))
        
            
            
        print('Generating Description from Image')
        full_response =  client.generate(model=model, system=system_context, prompt=prompt, images=images_base64, keep_alive=keep_model_alive, stream=False, options={
                'num_predict': max_tokens,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'repeat_penalty': repeat_penalty, 
                'seed': seed_number,
                'main_gpu': 0,
                'low_vram': False,
        })
        
        result = full_response['response']
        
        print('Finalized')
        return (result, )




class OllamaTextDescriber:
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (text_models,),
                "custom_model": ("STRING", {
                    "default": ""
                }),
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
                "top_k": ("INT", {
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "default": 40
                }),
                "top_p": ("FLOAT", {
                    "min": 0,
                    "max": 10,
                    "step": 0.1,
                    "default": 0.9
                }),
                "repeat_penalty": ("FLOAT", {
                    "min": 0,
                    "max": 10,
                    "step": 0.1,
                    "default": 1.1
                }),
                "seed_number": ("INT", {
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": 42
                }),
                "max_tokens": ("INT", {
                    "step": 10,
                    "default": 200
                }),
                "keep_model_alive": ("INT", {
                     "min": -1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": -1
                }),
                "system_context": ("STRING", {
                    "multiline": True,
                    "default": """Your job is to generate prompts to be used in stable diffusion/midjourney,
you will receive a text and you need to extract the most important information/characteristics from this text and transform this into tags in booru format.
                    """,
                    "title":"system"
                }),
                "prompt": ("STRING", {
                    "default": """Extract the booru tags from this text in detail but do not extract text information, such as the artist, return a line with the tags in lowercase letters and separated by commas in the following format: <TAGS>tag1, tag2, tag3</TAGS>.

Example:
Input: "A beautiful scenery with mountains and a light blue river."
Output: <TAGS>scenery, mountains, (light blue river:1.2)</TAGS>

Input: "A portrait of a woman with a red dress and a hat."
Output: <TAGS>portrait, woman, (red dress:1.2), hat</TAGS>

Now, extract the tags from the following text: """,
                    "multiline": True
                })
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)

    FUNCTION = "ollama_text_describe"

    OUTPUT_NODE = True

    CATEGORY = "Ollama"

    def ollama_text_describe(self, 
                             model, 
                             custom_model, 
                             api_host, 
                             timeout, 
                             temperature,
                             top_k,
                             top_p,
                             repeat_penalty, 
                             seed_number, 
                             max_tokens, 
                             keep_model_alive, 
                             prompt, 
                             system_context):
        
        client = Client(api_host, timeout=timeout)
        ollama_util = OllamaUtil()
        
        models = [model_l['name'] for model_l in client.list()['models']]
        
        model = model.split(' ')[0].strip()
        custom_model = custom_model.strip()
        
        model = custom_model if custom_model != "" else model
        
        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)
            
        print('System Context: "{}"'.format(system_context))
        print('Prompt: "{}"'.format(prompt))
        
        full_response = ""
                
        print('Generating Response')
        full_response =  client.generate(model=model, system=system_context, prompt=prompt, keep_alive=keep_model_alive, stream=False, options={
                'num_predict': max_tokens,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'repeat_penalty': repeat_penalty, 
                'seed': seed_number
        })
        
        result = full_response['response']
        
        print('Finalized')
        return (result, )
    
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

class TextTransformer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
             "optional": {
                "prepend_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "append_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "replace_find_mode":  (["normal", "regular expression (regex)"],),
                "replace_find": ("STRING", {
                    "default": "",
                }),
                "replace_with": ("STRING", {
                    "default": "",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transform_text"
    CATEGORY = "Ollama"

    def unescape_string(self, s):
        return (s.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\r", "\r")
                .replace("\\b", "\b")
                .replace("\\f", "\f")
                .replace("\\v", "\v")
                .replace("\\\\", "\\")
                .replace("\\'", "'")
                .replace('\\"', '"')
                .replace("\\a", "\a"))

    def transform_text(self, text, prepend_text="", append_text="", replace_find_mode="normal", replace_find="", replace_with=""):

        text = self.unescape_string(text)
        
        if prepend_text:
            text = self.unescape_string(prepend_text) + text
        
        if append_text:
            text = text + self.unescape_string(append_text)

        if replace_find_mode == "normal":
            if replace_find:
                text = text.replace(replace_find, replace_with)
        elif replace_find_mode == "regular expression (regex)":
            if replace_find:
                try:
                    text = re.sub(replace_find.strip(), replace_with, text)
                except re.error as e:
                    print(f"Error: Invalid regular expression '{replace_find}'. Details: {e}")
                    return (f"Error: Invalid regular expression '{replace_find}'. Details: {e}",)

        return (text,)

class InputText:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{ "string": ("STRING", {"default": "", "multiline": True }) }}
    
    FUNCTION = "run"
    CATEGORY = "Ollama"
    RETURN_TYPES = ("STRING",)

    def run(self, text):
        return (text,)
    
# class ShowText:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":{ "string": ("STRING", {"default": "", "multiline": True, }) }}
    
#     FUNCTION = "run"
#     CATEGORY = "Ollama"
#     RETURN_TYPES = ("STRING",)

#     def run(self, text):
#         return (text,)
    
# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "OllamaImageDescriber": OllamaImageDescriber,
    "OllamaTextDescriber": OllamaTextDescriber,
    "TextTransformer": TextTransformer,
    "InputText": InputText,
    # "ShowText": ShowText
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaImageDescriber": "🦙 Ollama Image Describer 🦙",
    "OllamaTextDescriber": "🦙 Ollama Text Describer 🦙",
    "TextTransformer": "📝 Text Transformer 📝",
    "InputText": "📝 Input Text (Multiline) 📝",
    # "ShowText": "📝 Show Text 📝"
}
