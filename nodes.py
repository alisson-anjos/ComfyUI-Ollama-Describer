from ollama import Client
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import re
import os
import comfy.model_management
import comfy.utils
import folder_paths
from .config import configurations

multimodal_models = list(configurations["multimodal_models"])
text_models = list(configurations["text_models"])
caption_types = list(configurations["caption_types"].keys())
caption_lengths = list(configurations["caption_lengths"])
extra_options = list(configurations["extra_options"])

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
           
class OllamaCaptionerExtraOptions:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        options = extra_options
        required = {}
        for option in options:
            required[option] = ("BOOLEAN", {"default": False})
        return {
            "required": required
        }

    CATEGORY = "Ollama"
    RETURN_TYPES = ("Extra_Options",)
    FUNCTION = "run"

    def run(self, **kwargs):
        options_selected = list(kwargs.values())
        options = extra_options
        values = []
        for selected, option in zip(options_selected, options):
            if selected:
                values.append(option)
        return (values, )
            
class OllamaImageCaptioner:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (multimodal_models,),
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
                "input_dir": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": ""}),
                "max_images": ("INT", { "default": -1 }),
                "low_vram": ("BOOLEAN", { "default": False }),
                "keep_model_alive": ("INT", {
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": -1
                }),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "caption_type": (caption_types,),
                "caption_length": (caption_lengths,),
                "name": ("STRING", {"default": ""}),
                "custom_prompt": ("STRING", {"default": ""}),
                "prefix_caption": ("STRING", {"default": ""}),
                "suffix_caption": ("STRING", {"default": ""}),
            },
            "optional": {
                "extra_options": ("Extra_Options", ),
            }
        } 
          
    FUNCTION = "run_captioner"

    OUTPUT_NODE = True
    
    RETURN_TYPES = ("STRING",)

    CATEGORY = "Ollama"

    def run_captioner(self, model, custom_model, api_host, timeout, low_vram, keep_model_alive, input_dir,output_dir, max_images,caption_type,caption_length,name, custom_prompt, top_p,temperature, prefix_caption, suffix_caption, extra_options=[]):
        
        client = Client(api_host, timeout=timeout)
        
        ollama_util = OllamaUtil()
        
        models = [model_l['name'] for model_l in client.list()['models']]
        
        model = model.split(' ')[0].strip()
        custom_model = custom_model.strip()
        
        model = custom_model if custom_model != "" else model
        
        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)
        
        length = None if caption_length == "any" else caption_length
        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass
        
        # Build prompt
        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")
        
        system_context = "You are a helpful image captioner."
        
        caption_type_map = configurations["caption_types"]
        prompt_str = list(caption_type_map[caption_type])[map_idx]
        
            # Add extra options
        if len(extra_options) > 0:
            prompt_str += " " + " ".join(extra_options)
        
        prompt_str = prompt_str.format(name=name, length=caption_length, word_count=caption_length)
        
        print('System Context: "{}"'.format(system_context))
        print('Prompt: "{}"'.format(prompt_str))
        
        if output_dir is None or output_dir.strip() == "":
            output_dir = input_dir

        finished_image_count = 0
        error_image_count = 0
        image_count = 0
        
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")) & (max_images == -1 or image_count < max_images):
                image_count += 1
        
        pbar = comfy.utils.ProgressBar(image_count)
        step = 0
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")) & (max_images == -1 or step < max_images):
                image_path = os.path.join(input_dir, filename)
                text_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')

                try:
                    print(f"Processing: {image_path}")
                    with Image.open(image_path) as img:
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        pbar.update_absolute(step, image_count)
                        image = img.resize((384, 384), Image.LANCZOS)
                        
                        print('Converting Image to base64')
                        image_base64 = ollama_util.image_to_base64(image)
                        image_base64 = [str(image_base64, 'utf-8')]
                        
                        print('Generating Description from Image')
                        full_response =  client.generate(model=model, system=system_context, prompt=prompt_str, images=image_base64, keep_alive=keep_model_alive, stream=False, options={
                                'temperature': temperature,
                                'top_p': top_p,
                                'main_gpu': 0,
                                'low_vram': low_vram,
                        })
        
                        caption = full_response['response']
                        
                        if prefix_caption:
                            caption = f"{prefix_caption} {caption}"
                        if suffix_caption:
                            caption = f"{caption} {suffix_caption}"
        
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(caption)
                    finished_image_count += 1
                except Exception as e:
                    print(f"Error processing {filename} :{e}")
                    error_image_count += 1
                step += 1
                
        return (f"result: finished count: {finished_image_count}, error count: {error_image_count}", )

            
class OllamaImageDescriber:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "model": (multimodal_models,),
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
                "num_ctx": ("INT", {
                    "step": 64,
                    "default": 2048
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
                              num_ctx,
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
                'num_ctx': num_ctx,
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
                 "num_ctx": ("INT", {
                    "step": 64,
                    "default": 2048
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
                             num_ctx,
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
                'num_ctx': num_ctx,
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
    "OllamaImageCaptioner": OllamaImageCaptioner,
    "OllamaTextDescriber": OllamaTextDescriber,
    "TextTransformer": TextTransformer,
    "InputText": InputText,
    "OllamaCaptionerExtraOptions": OllamaCaptionerExtraOptions
    # "ShowText": ShowText
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaImageDescriber": "🦙 Ollama Image Describer 🦙",
    "OllamaImageCaptioner": "🦙 Ollama Image Captioner 🦙",
    "OllamaTextDescriber": "🦙 Ollama Text Describer 🦙",
    "TextTransformer": "📝 Text Transformer 📝",
    "InputText": "📝 Input Text (Multiline) 📝",
    "OllamaCaptionerExtraOptions": "🦙 Ollama Captioner Extra Options 🦙"
    # "ShowText": "📝 Show Text 📝"
}
