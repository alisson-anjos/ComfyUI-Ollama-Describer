# ComfyUI-LLaVA-Describer

This is an extension for ComfyUI to extract descriptions from your images using the multimodal model called LLaVa. The LLaVa model - Large Language and Vision Assistant, although trained on a relatively small dataset, demonstrates exceptional capabilities in understanding images and answering questions about them. This model shows behaviors similar to multimodal models like GPT-4, even when presented with unseen images and instructions.

For more information about the LLaVa model, visit: [LLaVa Model Website](https://llava-vl.github.io/)

![alt text](image.png)

## Requirements

To use this extension, you will need the Ollama library, which facilitates the use of large-scale language models (LLMs). Ollama provides a simple and efficient interface for interacting with these models, including facilitating the use of GPUs using CUDA (NVIDIA). It also supports AMD GPUs.

Ollama is currently available for installation on Windows, Linux, and Mac. Additionally, you can run Ollama using Docker. In both cases, if you want to use your NVIDIA GPUs to speed up processing, you need to install the NVIDIA CUDA Toolkit.

Follow the guide on the Ollama website and install according to your operating system or opt for usage through a Docker container.

- [Ollama Website](https://ollama.com/)
- [Ollama Docker Hub](https://hub.docker.com/r/ollama/ollama)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Ollama GPU Installation Guide](https://github.com/ollama/ollama/blob/main/docs/gpu.md)

## Installation

1. Install and Run Ollama
    - [Ollama Website](https://ollama.com/)

2. Clone the repository into your `custom_nodes` folder:
    ```bash
    git clone https://github.com/alisson-anjos/ComfyUI-LLaVA-Describer.git
    ```
   The path should be something like `custom_nodes\ComfyUI-LLaVA-Describer`.
   
3. Open the folder and execute `install.bat` (Windows) or open a terminal in the folder and run:
    ```bash
    pip install -r requirements.txt
    ```
After installing Ollama and getting it running, you can use the extension in your ComfyUI.

## Usage
Add the node via `image` -> `LlaVa Describer by Alisson`  
- **model**: Select one of the models, 7b, 13b, 34b. The larger the model, the longer it will take. If you don't have the necessary hardware, it is advised to use the 13b model.
- **temperature**: The higher this value, or the closer to 1, the more creative/random the response will be. In other words, the model may even stray from the context of the prompt.
- **prompt**: Text provided to the model containing a question or instruction.
- **max_tokens**: Maximum length of response, in tokens. A token is approximately half a word.
- **run_mode**: There are two ways to use Ollama: through the library directly installed on the machine (Windows, Linux, and Mac) or through Docker using the API endpoint provided by the container. Choose according to the method used for installation.
- **api_host**: This host will only be used if the selected mode is API (Ollama). You can provide either a local host if using Docker or a remote host.