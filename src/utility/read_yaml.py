import yaml

def read_yaml_file(path):
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except Exception as e:
            pass
            
            # Add logger here