from ollama import Client, ListResponse
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
tool_calling_models = list(configurations.get("tool_calling_models", []))

class OllamaUtil:
    def __init__(self, client):
        self.client = client
    
    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        # Prevent squeeze from removing spatial dimensions if they happen to be 1
        # ComfyUI image tensors are normally [H, W, C] after iteration from batch
        if len(tensor.shape) == 3:
            image_np = tensor.mul(255).clamp(0, 255).byte().numpy()
        else:
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
    
    def get_models(self):
        response: ListResponse = self.client.list()
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

    @staticmethod
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
        
        if structured_output_format is not None and isinstance(structured_output_format, str) and structured_output_format.strip():
            try:
                structured_output_format = json.loads(structured_output_format)
            except Exception:
                pass

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
                    "min": 1,
                    "max": 0xffffffffffffffff,
                    "step": 64,
                    "default": 1024
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
        
        models = ollama_util.get_models()
        
        model = model.split(' ')[0].strip()
        custom_model = custom_model.strip()
        
        model = custom_model if custom_model != "" else model
        
        if model not in models:
            print(f"Downloading model: {model}")
            ollama_util.pull_model(model, client)
            
        print('System Context: "{}"'.format(system_context))
        print('Prompt: "{}"'.format(prompt))

        if structured_output_format is not None and isinstance(structured_output_format, str) and structured_output_format.strip():
            try:
                structured_output_format = json.loads(structured_output_format)
            except Exception:
                pass
        
        full_response = ""
        
        images_base64 = []       
        
        for (batch_number, image) in enumerate(images):
            print('Converting Tensor to Image')
            img = ollama_util.tensor_to_image(image)
            print('Converting Image to bytes')
            img_base64 = ollama_util.image_to_base64(img)
            images_base64.append(str(img_base64, 'utf-8'))
        
            
            
        print('Generating Description from Image')
        
        try:
            full_response = client.generate(model=model, system=system_context, prompt=prompt, images=images_base64, keep_alive=keep_model_alive, stream=False, format=structured_output_format, options={
                    'num_ctx': num_ctx,
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p,
                    'repeat_penalty': repeat_penalty, 
                    'seed': seed_number,
            })
            result = full_response['response']
        except Exception as e:
            print(f"Error calling ollama: {e}")
            result = f"Error: {e}"
        
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
                    "min": 1,
                    "max": 0xffffffffffffffff,
                    "step": 64,
                    "default": 1024
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
        
        if structured_output_format is not None and isinstance(structured_output_format, str) and structured_output_format.strip():
            try:
                structured_output_format = json.loads(structured_output_format)
            except Exception:
                pass

        full_response = ""
                
        print('Generating Response')
        try:
            full_response = client.generate(model=model, system=system_context, prompt=prompt, keep_alive=keep_model_alive, stream=False, format=structured_output_format, options={
                    'num_ctx': num_ctx,
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p,
                    'repeat_penalty': repeat_penalty, 
                    'seed': seed_number
            })
            result = full_response['response']
        except Exception as e:
            print(f"Error calling ollama: {e}")
            result = f"Error: {e}"
        
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
    
class OllamaVideoDescriber:
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
                    "default": 4096
                }),
                "max_tokens": ("INT", {
                    "min": 1,
                    "max": 0xffffffffffffffff,
                    "step": 128,
                    "default": 4096
                }),
                "keep_model_alive": ("INT", {
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "default": -1
                }),
                "video_frames": ("IMAGE",),
                "frame_skip": ("INT", {
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "default": 5
                }),
                "max_frames": ("INT", {
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "default": 16
                }),
                "system_context": ("STRING", {
                    "multiline": True,
                    "default": """You are a helpful AI assistant specialized in analyzing a sequence of video frames and generating a detailed and accurate textual description of the events. Describe the actions, people, objects, and how the scene evolves across the frames.""",
                    "title":"system"
                }),
                "prompt": ("STRING", {
                    "default": """Describe the events happening in this sequence of video frames in detail. Provide a clear and concise description that highlights the most important actions. Video:""",
                    "multiline": True
                })
            },
            "optional": {
                "structured_output_format": ("STRING", { "forceInput": True, "default": None }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "ollama_video_describe"
    OUTPUT_NODE = True
    CATEGORY = "Ollama"

    def ollama_video_describe(self, 
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
                              video_frames,
                              frame_skip,
                              max_frames,
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

        if structured_output_format is not None and isinstance(structured_output_format, str) and structured_output_format.strip():
            try:
                structured_output_format = json.loads(structured_output_format)
            except Exception:
                pass
        
        images_base64 = []
        
        # Slicing the frames based on frame_skip and max_frames
        selected_frames = video_frames[::frame_skip]
        if len(selected_frames) > max_frames:
            print(f"Info: Capping selected video frames to {max_frames} (was {len(selected_frames)})")
            selected_frames = selected_frames[:max_frames]
        
        print(f"Processing {len(selected_frames)} frames out of {len(video_frames)} total.")
        
        for (batch_number, image) in enumerate(selected_frames):
            img = ollama_util.tensor_to_image(image)
            img_base64 = ollama_util.image_to_base64(img)
            images_base64.append(str(img_base64, 'utf-8'))
        
        print('Generating Description from Video Frames')
        try:
            full_response = client.generate(model=model, system=system_context, prompt=prompt, images=images_base64, keep_alive=keep_model_alive, stream=False, format=structured_output_format, options={
                    'num_ctx': num_ctx,
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'top_k': top_k,
                    'top_p': top_p,
                    'repeat_penalty': repeat_penalty, 
                    'seed': seed_number,
            })
            result = full_response['response']
        except Exception as e:
            print(f"Error calling ollama: {e}")
            result = f"Error: {e}"
        
        print('Finalized')
        return (result, )

class OllamaToolCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tool_1": ("OLLAMA_TOOL",),
            },
            "optional": {
                "tool_2": ("OLLAMA_TOOL",),
                "tool_3": ("OLLAMA_TOOL",),
                "tool_4": ("OLLAMA_TOOL",),
            }
        }
    RETURN_TYPES = ("OLLAMA_TOOL",)
    RETURN_NAMES = ("tools",)
    FUNCTION = "combine"
    CATEGORY = "Ollama/Tools"

    def combine(self, tool_1, tool_2=None, tool_3=None, tool_4=None):
        tools = []
        if isinstance(tool_1, list):
            tools.extend(tool_1)
        else:
            tools.append(tool_1)
            
        for t in [tool_2, tool_3, tool_4]:
            if t is not None:
                if isinstance(t, list):
                    tools.extend(t)
                else:
                    tools.append(t)
        return (tools,)

class OllamaTool_WebSearch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tool_name": ("STRING", {
                    "default": "search_internet",
                    "tooltip": "The name of the tool as seen by the Agent. You can change this to 'google_search' or 'web_lookup' to help the agent understand when to use it."
                }),
                "search_provider": (["DuckDuckGo (free)", "Ollama API (requires key)"], {
                    "default": "DuckDuckGo (free)",
                    "tooltip": "Choose the search engine. DuckDuckGo is free and requires no configuration. Ollama API gives better results but requires an API key from ollama.com/settings/keys."
                }),
                "max_results": ("INT", {"default": 5, "min": 1, "max": 10, "tooltip": "Maximum number of search results to return. Fewer results = less context used by the Agent."}),
            },
            "optional": {
                "ollama_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Your Ollama API key from https://ollama.com/settings/keys. Only used when 'Ollama API' is selected as provider."
                }),
            }
        }
    
    RETURN_TYPES = ("OLLAMA_TOOL",)
    RETURN_NAMES = ("tool",)
    FUNCTION = "get_tool"
    CATEGORY = "Ollama/Tools"
    DESCRIPTION = """Allows the Agent to search the internet for up-to-date information.

- DuckDuckGo (free): No setup needed, works out of the box.
- Ollama API: More accurate, requires a free API key from ollama.com/settings/keys.

📚 <a href='https://docs.ollama.com/capabilities/web-search' target='_blank'>Ollama Docs: Web Search</a>"""

    def get_tool(self, tool_name="search_internet", search_provider="DuckDuckGo (free)", max_results=5, ollama_api_key=""):
        _max_results = max_results
        
        if search_provider == "Ollama API (requires key)":
            api_key = ollama_api_key.strip()
            import ollama as _ollama
            
            def search_internet(query: str) -> str:
                """Search the internet for information, news, or facts using the Ollama Web Search API.
                
                Args:
                    query: The search term or question to look up on the web.
                    
                Returns:
                    A string containing search results and snippets.
                """
                try:
                    import os
                    os.environ["OLLAMA_API_KEY"] = api_key
                    client = _ollama.Client()
                    results = client.web_search(query, max_results=_max_results)
                    if not results or not results.results:
                        return f"No results found for '{query}'."
                    
                    snippets = []
                    for r in results.results:
                        snippets.append(f"• {r.title}\n  {r.url}\n  {r.content}")
                    return "Web Search Results:\n\n" + "\n\n".join(snippets)
                except Exception as e:
                    return f"Error performing Ollama web search: {str(e)}"
        else:
            # Fallback: DuckDuckGo HTML scraping
            def search_internet(query: str) -> str:
                """Search the internet for information, news, or facts.
                
                Args:
                    query: The search term or question to look up on the web.
                    
                Returns:
                    A string containing search results and snippets.
                """
                import urllib.request
                import urllib.parse
                import re
                try:
                    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    html = urllib.request.urlopen(req).read().decode('utf-8')
                    snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL)
                    if not snippets:
                        return f"No results found for '{query}'."
                    clean_snippets = [re.sub(r'<[^>]+>', '', s).strip() for s in snippets[:_max_results]]
                    return "Web Search Results (DuckDuckGo):\n- " + "\n- ".join(clean_snippets)
                except Exception as e:
                    return f"Error performing web search: {str(e)}"
            
        # Ensure the function has the user-defined name
        search_internet.__name__ = tool_name
        return (search_internet,)

class OllamaTool_FileSearch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tool_name": ("STRING", {
                    "default": "read_local_file",
                    "tooltip": "The name of the tool as seen by the Agent."
                })
            }
        }
    
    RETURN_TYPES = ("OLLAMA_TOOL",)
    RETURN_NAMES = ("tool",)
    FUNCTION = "get_tool"
    CATEGORY = "Ollama/Tools"
    DESCRIPTION = """Allows the Agent to read the contents of a local text file. Content is capped at 10,000 characters to avoid overflowing the model's context window."""

    def get_tool(self, tool_name="read_local_file"):
        def read_local_file(filepath: str) -> str:
            """Read the content of a local text file.
            
            Args:
                filepath: Absolute or relative path to the text file.
                
            Returns:
                The text content of the file, or an error string if not found.
            """
            import os
            try:
                if not os.path.exists(filepath):
                    return f"Error: File '{filepath}' does not exist."
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Limit return size to prevent context window overflow
                max_len = 10000 
                if len(content) > max_len:
                    content = content[:max_len] + f"... (file truncated, {len(content)-max_len} more characters)"
                return f"Contents of {filepath}:\n{content}"
            except Exception as e:
                return f"Error reading file '{filepath}': {str(e)}"
            
        # Ensure the function has the user-defined name
        read_local_file.__name__ = tool_name
        return (read_local_file,)

class OllamaTool_PythonCode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tool_name": ("STRING", {
                    "default": "custom_python_tool",
                    "tooltip": "The name of the tool as seen by the Agent. If left as 'custom_python_tool', it will try to use the function name defined in the code."
                }),
                "python_code": ("STRING", {
                    "multiline": True,
                    "default": 'def custom_tool(text: str) -> str:\n    """A helpful description of what this does.\n    \n    Args:\n        text: input text\n        \n    Returns:\n        a useful string\n    """\n    return f"Processed: {text} | {my_ext_var}"'
                })
            },
            "optional": {
                # Add a dummy kwargs catch-all to accept ANY input from other ComfyUI nodes
                "my_ext_var": ("*", {"tooltip": "Connect ANY node output here and it will be available as a variable inside your python code!"}),
                "my_ext_var_2": ("*", {"tooltip": "Another external variable"}),
                "my_ext_var_3": ("*", {"tooltip": "Another external variable"})
            }
        }
    
    RETURN_TYPES = ("OLLAMA_TOOL",)
    RETURN_NAMES = ("tool",)
    FUNCTION = "get_tool"
    CATEGORY = "Ollama/Tools"
    DESCRIPTION = "Define an arbitrary Python function with docstrings to act as a custom Agent Tool. Inputs plugged into 'my_ext_var' will be globally available inside the function!"

    def get_tool(self, tool_name="custom_python_tool", python_code="", **kwargs):
        import textwrap
        
        # We need to compile and extract the function from the user's string
        local_scope = {}
        
        # Inject any kwargs (from the ANY inputs) directly into the environment where the code runs!
        local_scope.update(kwargs)
        
        try:
            # Execute the user's code definition to create the function in local_scope
            exec(python_code, globals(), local_scope)
            
            # Find the first callable object defined that isn't a built-in
            funcs = [v for k, v in local_scope.items() if callable(v)]
            if not funcs:
                raise ValueError("No valid function definition found in the provided code.")
                
            # Optionally rename the function if a custom name was provided
            if tool_name and tool_name != "custom_python_tool":
                funcs[0].__name__ = tool_name
                
            return (funcs[0],)
        except Exception as e:
            # Return a dummy function that reports the error if the user code is invalid
            def error_tool() -> str:
                """Reports that the custom python tool failed to compile."""
                return f"Error compiling custom Python tool: {str(e)}"
            return (error_tool,)

class OllamaAgent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (tool_calling_models,),
                "custom_model": ("STRING", {"default": ""}),
                "api_host": ("STRING", {"default": "http://localhost:11434"}),
                "timeout": ("INT", {"min": 0, "max": 0xffffffffffffffff, "step": 1, "default": 300}),
                "temperature": ("FLOAT", {"min": 0, "max": 10, "step": 0.1, "default": 0.2}),
                "max_tokens": ("INT", {
                    "min": 1,
                    "max": 0xffffffffffffffff,
                    "step": 64,
                    "default": 2048
                }),
                "system_context": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful and intelligent agent. You have access to tools that you can call to answer user queries. \n\nIMPORTANT: If the user asks for real-time information (like current time, date, local weather, or web searches), DO NOT refuse or say you do not have access. Instead, use the available tools (like 'search_internet') IMMEDIATELY to find the answer. Always prioritize using tools over guessing or refusing.",
                    "title": "system"
                }),
                "prompt": ("STRING", {"multiline": True, "default": "What time is it right now?"}),
                "think": ("BOOLEAN", {"default": False, "tooltip": "Enable reasoning (e.g. for Qwen3, DeepSeek-R1) before outputting final answer.", "label_on": "enabled", "label_off": "disabled"})
            },
            "optional": {
                "tools": ("OLLAMA_TOOL", {"tooltip": "Connect OllamaTools nodes here so the Agent can call them to gather real-time data or perform actions."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "ollama_agent_chat"
    OUTPUT_NODE = True
    CATEGORY = "Ollama/Agent"
    DESCRIPTION = """Autonomous LLM Agent that can use connected tools to answer questions.

The agent calls the model, and if the model decides to use a tool, the agent executes it and feeds the result back for a final answer.

🧠 Supported for 'think': Qwen3, DeepSeek-R1, DeepSeek-v3.1, GPT-OSS

📚 <a href='https://docs.ollama.com/capabilities/tool-calling' target='_blank'>Ollama Docs: Tool Calling</a>"""

    def ollama_agent_chat(self, model, custom_model, api_host, timeout, temperature, max_tokens, system_context, prompt, think, tools=None):
        client = Client(api_host, timeout=timeout)
        ollama_util = OllamaUtil(client)
        
        models = ollama_util.get_models()
        model_name = model.split(' ')[0].strip()
        custom_model = custom_model.strip()
        model_name = custom_model if custom_model != "" else model_name
        
        if model_name not in models:
            print(f"Downloading model: {model_name}")
            ollama_util.pull_model(model_name, client)
            
        messages = [
            {'role': 'system', 'content': system_context},
            {'role': 'user', 'content': prompt}
        ]

        agent_tools = []
        tool_functions = {}
        
        # Parse the provided tools
        if tools is not None:
            if not isinstance(tools, list):
                tools = [tools]
            for t in tools:
                if isinstance(t, dict) and "schema" in t and "executable" in t:
                    # Legacy or explicit schema approach
                    agent_tools.append(t["schema"])
                    tool_functions[t["schema"]["function"]["name"]] = t["executable"]
                elif callable(t):
                    # Direct Python function approach (Ollama Python SDK auto-schema)
                    print(f"Adding tool function: {t.__name__}")
                    agent_tools.append(t)
                    tool_functions[t.__name__] = t

        print(f"Agent starting with model: {model_name}")
        print(f"Available tools: {list(tool_functions.keys())}")

        try:
            # First interaction to see if it wants to use a tool
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
            }
            
            # Using 'chat' instead of 'generate' for Agents
            kwargs = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": options,
                "think": think
            }
            if len(agent_tools) > 0:
                kwargs["tools"] = agent_tools
            
            # Iterative Loop for Thinking + Tool Calling
            # This allows the model to call multiple tools or the same tool multiple times
            # and refine its answer based on the results.
            iteration = 0
            max_iterations = 10 # Safety cap
            
            while iteration < max_iterations:
                iteration += 1
                print(f"Agent iteration {iteration}...")
                
                response = client.chat(**kwargs)
                messages.append(response.message)
                
                # If there are no tool calls, this is the final answer
                if not response.message.tool_calls:
                    print("No more tool calls. Formulating final answer.")
                    final_text = response.message.content
                    if think and response.message.thinking:
                        final_text = f"<think>\n{response.message.thinking}\n</think>\n\n{final_text}"
                    return (final_text,)
                
                # Process tool calls
                print(f">>> Model decided to call {len(response.message.tool_calls)} tool(s)")
                for tool_call in response.message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = dict(tool_call.function.arguments) if tool_call.function.arguments else {}
                    
                    print(f"    [TOOL CALL] {function_name}({arguments})")
                    
                    if function_name in tool_functions:
                        try:
                            tool_result = tool_functions[function_name](**arguments)
                            # Log a bit of the result
                            res_text = str(tool_result)
                            preview = (res_text[:100] + '...') if len(res_text) > 100 else res_text
                            print(f"    [TOOL RESULT] {function_name} -> {preview}")
                        except Exception as e:
                            tool_result = f"Error executing tool {function_name}: {str(e)}"
                            print(f"    [TOOL ERROR] {function_name}: {e}")
                        
                        messages.append({'role': 'tool', 'tool_name': function_name, 'content': str(tool_result)})
                    else:
                        print(f"    [TOOL WARNING] Unknown tool: {function_name}")
                        messages.append({'role': 'tool', 'tool_name': function_name, 'content': f"Error: Tool {function_name} not found."})
                
                # Update messages for the next iteration
                kwargs["messages"] = messages
            
            return ("Error: Maximum iterations reached without a final answer.",)
                
        except Exception as e:
            print(f"Error calling ollama agent: {e}")
            return (f"Error: {e}",)

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
WEB_DIRECTORY = "./js"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "OllamaImageDescriber": OllamaImageDescriber,
    "OllamaImageCaptioner": OllamaImageCaptioner,
    "OllamaTextDescriber": OllamaTextDescriber,
    "OllamaVideoDescriber": OllamaVideoDescriber,
    "OllamaAgent": OllamaAgent,
    "OllamaToolCombine": OllamaToolCombine,
    "OllamaTool_WebSearch": OllamaTool_WebSearch,
    "OllamaTool_FileSearch": OllamaTool_FileSearch,
    "OllamaTool_PythonCode": OllamaTool_PythonCode,
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
    "OllamaVideoDescriber": "🦙 Ollama Video Describer 🦙",
    "OllamaAgent": "🤖 Ollama Agent 🤖",
    "OllamaToolCombine": "🛠️ Combine Ollama Tools 🛠️",
    "OllamaTool_WebSearch": "🛠️ Tool: Web Search 🛠️",
    "OllamaTool_FileSearch": "🛠️ Tool: Read File 🛠️",
    "OllamaTool_PythonCode": "🛠️ Tool: Custom Python Code 🛠️",
    "TextTransformer": "📝 Text Transformer 📝",
    "InputText": "📝 Input Text (Multiline) 📝",
    "OllamaCaptionerExtraOptions": "🦙 Ollama Captioner Extra Options 🦙",
    "JsonPropertyExtractorNode": "📝 Json Property Extractor 📝"
    # "ShowText": "📝 Show Text 📝"
}
