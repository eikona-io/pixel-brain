import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 