import adalflow as adal
from typing import Any

from api.config import configs, get_embedder_type


def get_embedder(is_local_ollama: bool = False, use_google_embedder: bool = False, embedder_type: str = None) -> adal.Embedder:
    if embedder_type:
        if embedder_type == 'ollama':
            embedder_config: dict[str, Any] = configs["embedder_ollama"]
        elif embedder_type == 'google':
            embedder_config: dict[str, Any] = configs["embedder_google"]
        else:
            embedder_config: dict[str, Any] = configs["embedder"]
    elif is_local_ollama:
        embedder_config: dict[str, Any] = configs["embedder_ollama"]
    elif use_google_embedder:
        embedder_config: dict[str, Any] = configs["embedder_google"]
    else:
        current_type: str = get_embedder_type()
        if current_type == 'ollama':
            embedder_config: dict[str, Any] = configs["embedder_ollama"]
        elif current_type == 'google':
            embedder_config: dict[str, Any] = configs["embedder_google"]
        else:
            embedder_config: dict[str, Any] = configs["embedder"]

    model_client_class = embedder_config["model_client"]
    if "initialize_kwargs" in embedder_config:
        model_client = model_client_class(**embedder_config["initialize_kwargs"])
    else:
        model_client = model_client_class()
    
    embedder_kwargs = {"model_client": model_client, "model_kwargs": embedder_config["model_kwargs"]}
    
    embedder: adal.Embedder = adal.Embedder(**embedder_kwargs)
    
    if "batch_size" in embedder_config:
        embedder.batch_size = embedder_config["batch_size"]
    return embedder
