from fastapi import FastAPI, Request
import torch
import os
import logging
import yaml
import numpy as np
from google.cloud import storage
from torch.utils.data import DataLoader

# Import directly from your project modules
from src.model.pretrain.belkaml_arch import Belka
from utils.torch_data_utils import SMILESTokenizer, BelkaRawDataset
from utils.protein_encoder import ProteinEncoder
from skfp.fingerprints import ECFPFingerprint

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("belka-serving")

app = FastAPI()

# Global variables
model = None
tokenizer = None
ecfp_transformer = None
protein_encoder = None
config = {}

# Constants
DEFAULT_CONFIG_GCS_PATH = "gs://belkamlbucket/configs/vertex_train_config.yaml"
DEFAULT_VOCAB_GCS_PATH = "gs://belkamlbucket/data/raw/vocab.txt"
DEFAULT_PROTEIN_VOCAB_GCS_PATH = "gs://belkamlbucket/data/raw/protein_vocab.txt"


def download_blob(gcs_uri: str, local_path: str):
    """Helper to download a file from GCS."""
    if not gcs_uri.startswith("gs://"):
        return False
    try:
        storage_client = storage.Client()
        bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
        blob_name = "/".join(gcs_uri.replace("gs://", "").split("/")[1:])
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to download {gcs_uri}: {e}")
        return False


@app.on_event("startup")
def load_resources():
    """Runs once on startup to load config, vocab, preprocessors, and model."""
    global model, tokenizer, ecfp_transformer, protein_encoder, config
    logger.info("Starting initialization...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Locate Artifacts
    model_dir = os.environ.get("AIP_STORAGE_URI", ".")
    model_path = os.path.join(model_dir, "model.pt")
    local_config_path = os.path.join(model_dir, "config.yaml")

    # 2. Load Configuration
    if os.path.exists(local_config_path):
        logger.info(f"Loading config from local artifact: {local_config_path}")
        with open(local_config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.info("Config not found locally. Downloading from GCS...")
        if download_blob(DEFAULT_CONFIG_GCS_PATH, "/tmp/config.yaml"):
            with open("/tmp/config.yaml", 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.warning("Using empty config (defaults will apply).")
            config = {}

    # 3. Setup Preprocessors
    vocab_gcs_path = config.get("vocab_path", DEFAULT_VOCAB_GCS_PATH)
    local_vocab_path = "/tmp/vocab.txt"

    logger.info(f"Downloading vocab from {vocab_gcs_path}...")
    download_blob(vocab_gcs_path, local_vocab_path)

    # Download protein vocab
    protein_vocab_gcs_path = config.get("protein_vocab_path", DEFAULT_PROTEIN_VOCAB_GCS_PATH)
    local_protein_vocab_path = "/tmp/protein_vocab.txt"

    logger.info(f"Downloading protein vocab from {protein_vocab_gcs_path}...")
    download_blob(protein_vocab_gcs_path, local_protein_vocab_path)

    logger.info("Initializing Tokenizer, ECFP Transformer, and Protein Encoder...")
    try:
        tokenizer = SMILESTokenizer(local_vocab_path)
        ecfp_transformer = ECFPFingerprint(fp_size=2048)
        protein_encoder = ProteinEncoder(local_protein_vocab_path)
    except Exception as e:
        logger.error(f"Failed to initialize preprocessors: {e}")
        raise e

    # 4. Initialize Model
    model_params = {
        "hidden_size": config.get('hidden_size', 32),
        "dropout_rate": config.get('dropout_rate', 0.1),
        "mode": "clf",
        "num_layers": config.get('num_layers', 2),
        "vocab_size": config.get('vocab_size', 44),
        "num_proteins": config.get('num_proteins', 3),
        "protein_embed_dim": config.get('protein_embed_dim', 16),
    }
    logger.info(f"Initializing model with params: {model_params}")

    try:
        model = Belka(**model_params).to(device)

        if not os.path.exists(model_path):
            logger.error(f"Model weights not found at {model_path}")
            return

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        logger.info("✓ Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e


@app.get("/health")
def health():
    if model is not None:
        return {"status": "healthy"}
    return {"status": "unhealthy"}, 503


@app.post("/predict")
async def predict(request: Request):
    """
    Accepts JSON: {"instances": [{"smiles": "C=CC...", "protein": "BRD4"}, ...]}
    """
    global model, tokenizer, ecfp_transformer, protein_encoder
    if not model:
        return {"error": "Model not initialized"}, 503

    try:
        body = await request.json()
        instances = body.get("instances", [])

        if not instances:
            return {"error": "No instances provided"}, 400

        # Extract SMILES strings and protein names
        smiles_list = []
        protein_list = []
        for inst in instances:
            if isinstance(inst, dict):
                smiles_list.append(inst.get("smiles", ""))
                protein_list.append(inst.get("protein", "BRD4"))  # Default to BRD4
            else:
                # Fallback: treat as SMILES string, use default protein
                smiles_list.append(str(inst))
                protein_list.append("BRD4")

        # 1. Create Dataset & Loader (Using the class from utils)
        max_length = config.get("max_length", 128)
        dataset = BelkaRawDataset(
            smiles_list=smiles_list,
            protein_list=protein_list,
            tokenizer=tokenizer,
            ecfp_transformer=ecfp_transformer,
            protein_vocab_path="/tmp/protein_vocab.txt",
            max_length=max_length
        )

        # DataLoader handles batching logic
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0  # 0 workers is safer/faster for small online requests
        )

        device = next(model.parameters()).device
        all_preds = []

        # 2. Inference Loop
        with torch.no_grad():
            for batch in loader:
                # Move features to device
                # BelkaRawDataset returns 'smiles' (token_ids), 'protein' (protein_id), and 'ecfp'
                x_smiles = batch['smiles'].to(device)
                x_protein = batch['protein'].to(device)
                outputs = model(x_smiles, x_protein)

                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy().tolist())

        return {"predictions": all_preds}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}, 500
