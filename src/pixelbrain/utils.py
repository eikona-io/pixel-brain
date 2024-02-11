import os
import logging
import sys

PIXELBRAIN_PATH = os.getenv("PIXELBRAIN_PATH", None)
MONGODB_ATLAS_KEY = os.getenv("MONGODB_ATLAS_KEY", None)
OPENAI_KEY = os.getenv("OPENAI_KEY", None)
PINECONE_KEY = os.getenv("PINECONE_KEY", None)

def get_logger(log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger