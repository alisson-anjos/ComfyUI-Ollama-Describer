import os
import json

def read_json_file(file_path):
  """Read a JSON file and convert to a Python dictionary."""
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
    return data
  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    return None
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in '{file_path}'.")
    return None

base_path = os.path.dirname(os.path.realpath(__file__))
configurations = read_json_file(os.path.join(base_path, "config.json"))