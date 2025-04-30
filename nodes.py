from ollama import Client, ListResponse
from ollama import list as OllamaList
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import re
import os
import json
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
    
    def __init__(self, client):
        self.client = client
    
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
    
    def get_models(self, client):
        response: ListResponse = client.list()
        return [model.model for model in response.models]
            
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

    def validate_json_syntax(json_str):
        try:
            json.loads(json_str)
            return True, "JSON Invalid."
        except json.JSONDecodeError as e:
            return False, f"Error in the JSON sintaxe: {e}"

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
                "max_images": ("INT", { "min": -1, "default": -1 }),
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
                "structured_output_format": ("STRING", { "forceInput": True, "default": None }),
            }
        } 
          
    FUNCTION = "run_captioner"

    OUTPUT_NODE = True
    
    RETURN_TYPES = ("STRING",)

    CATEGORY = "Ollama"

    def run_captioner(self, model, custom_model, api_host, timeout, low_vram, keep_model_alive, input_dir,output_dir, max_images,caption_type,caption_length,name, custom_prompt, top_p,temperature, prefix_caption, suffix_caption, extra_options=[], structured_output_format=None):
        
        client = Client(api_host, timeout=timeout)
        
        ollama_util = OllamaUtil(client)
        
        models = ollama_util.get_models()
        
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
        
        
        if custom_prompt != "":
            prompt_str += " " + custom_prompt
            
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
        
        if structured_output_format != None and type(structured_output_format) == str:
            structured_output_format = json.loads(structured_output_format.replace("'", '"'))

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
                        full_response =  client.generate(model=model, system=system_context, prompt=prompt_str, images=image_base64, keep_alive=keep_model_alive, stream=False, format=structured_output_format, options={
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
                    "default": """You are a helpful AI assistant specialized in generating detailed and accurate textual descriptions of images. Your task is to analyze the information provided about an image and create a clear, concise, and informative description. Focus on the key elements of the image, such as objects, people, actions, and the overall scene. Ensure the description is easy to understand and relevant to the context.""",
                    "title":"system"
                }),
                "prompt": ("STRING", {
                    "default": """Describe the following image in detail, focusing on its key elements such as objects, people, actions, and the overall scene. Provide a clear and concise description that highlights the most important aspects. Image:""",
                    "multiline": True
                })
            },
            "optional": {
                "structured_output_format": ("STRING", { "forceInput": True, "default": None }),
            }
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
                              images, 
                              structured_output_format=None):
        
        client = Client(api_host, timeout=timeout)
        
        ollama_util = OllamaUtil(client)
        
        models = ollama_util.get_models(client)
        
        model = model.split(' ')[0].strip()
        custom_model = custom_model.strip()
        
        model = custom_model if custom_model != "" else model
        
        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)
            
        print('System Context: "{}"'.format(system_context))
        print('Prompt: "{}"'.format(prompt))

        if structured_output_format != None and type(structured_output_format) == str:
            structured_output_format = json.loads(structured_output_format.replace("'", '"'))
        
        full_response = ""
        
        images_base64 = []       
        
        for (batch_number, image) in enumerate(images):
            print('Converting Tensor to Image')
            img = ollama_util.tensor_to_image(image)
            print('Converting Image to bytes')
            img_base64 = ollama_util.image_to_base64(img)
            images_base64.append(str(img_base64, 'utf-8'))
        
            
            
        print('Generating Description from Image')
        full_response =  client.generate(model=model, system=system_context, prompt=prompt, images=images_base64, keep_alive=keep_model_alive, stream=False, format=structured_output_format, options={
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
                    "default": """You are a helpful AI assistant specialized in generating detailed and accurate textual descriptions. Your task is to analyze the input provided and create a clear, concise, and informative description. Focus on the key aspects of the input, and ensure the description is easy to understand and relevant to the context.""",
                    "title":"system"
                }),
                "prompt": ("STRING", {
                    "default": """Describe the following input in detail, focusing on its key features and context. Provide a clear and concise description that highlights the most important aspects. Input:""",
                    "multiline": True
                })
            },
            "optional": {
                "structured_output_format": ("STRING", { "forceInput": True, "default": None }),
            }
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
                             system_context,
                             structured_output_format=None):
        
        client = Client(api_host, timeout=timeout)
        ollama_util = OllamaUtil(client)
        
        models = ollama_util.get_models()
        
        model = model.split(' ')[0].strip()
        custom_model = custom_model.strip()
        
        model = custom_model if custom_model != "" else model
        
        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)
            
        print('System Context: "{}"'.format(system_context))
        print('Prompt: "{}"'.format(prompt))
        
        if structured_output_format != None and type(structured_output_format) == str:
            structured_output_format = json.loads(structured_output_format.replace("'", '"'))

        full_response = ""
                
        print('Generating Response')
        full_response =  client.generate(model=model, system=system_context, prompt=prompt, keep_alive=keep_model_alive, stream=False, format=structured_output_format, options={
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
    
class StructuredOutputFormat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{ "string": ("STRING", {"default": "", "multiline": True }) }}
    
    FUNCTION = "run"
    CATEGORY = "Ollama"
    RETURN_TYPES = ("STRING",)

class JsonPropertyExtractorNode:
    """
    A custom node that takes a JSON string and a property path (e.g., "user.father.name"),
    and returns the value of the specified property.
    """
    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                "json_input": ("STRING", {"multiline": True, "default": "{}"}),
                "property_path": ("STRING", {"default": ""}),  # Property path (e.g., "user.father.name")
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_value",)
    FUNCTION = "run"
    CATEGORY = "Custom Nodes/JSON"

    def run(self, json_input: str, property_path: str):
        """
        Executes the node.
        """
        if not json_input or not property_path:
            raise ValueError("JSON input and property path are required.")

        try:
            json_data = json.loads(json_input)

            path_parts = property_path.split(".")

            current_value = json_data
            for part in path_parts:
                if isinstance(current_value, dict) and part in current_value:
                    current_value = current_value[part]
                else:
                    raise ValueError(f"Property '{part}' not found in JSON.")

            # Return the found value
            return (str(current_value),)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error: {e}")
    
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
    "OllamaCaptionerExtraOptions": OllamaCaptionerExtraOptions,
    "JsonPropertyExtractorNode": JsonPropertyExtractorNode
    # "ShowText": ShowText
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaImageDescriber": "🦙 Ollama Image Describer 🦙",
    "OllamaImageCaptioner": "🦙 Ollama Image Captioner 🦙",
    "OllamaTextDescriber": "🦙 Ollama Text Describer 🦙",
    "TextTransformer": "📝 Text Transformer 📝",
    "InputText": "📝 Input Text (Multiline) 📝",
    "OllamaCaptionerExtraOptions": "🦙 Ollama Captioner Extra Options 🦙",
    "JsonPropertyExtractorNode": "📝 Json Property Extractor 📝"
    # "ShowText": "📝 Show Text 📝"
}
