import glob
from pixelbrain.pre_processors.resnet_preprocessor import ResnetPreprocessor
from pixelbrain.pre_processors.hog_detector import HogsDetectorPreprocessor
from pixelbrain.pre_processors.resize import ResizePreprocessor
from torchvision.io import read_image
from pixelbrain.utils import PIXELBRAIN_PATH


def test_resnet_preprocessor():
    preprocessor = ResnetPreprocessor()
    test_images = [read_image(img) for img in glob.glob(f'{PIXELBRAIN_PATH}/assets/test_data/**.*')]
    processed_imgs = preprocessor(test_images)
    for img in processed_imgs:
        assert img.shape == (3, 256, 256)
        
        
def test_hog_detector_preprocessor():
    preprocessor = HogsDetectorPreprocessor()
    test_images = [read_image(img) for img in glob.glob('assets/test_data/**.*')]
    processed_imgs = preprocessor(test_images)
    for img in processed_imgs:
        assert img.shape == (512, 512)

def test_resize_preprocessor():
    preprocessor = ResizePreprocessor(size=128)
    test_images = [read_image(img) for img in glob.glob(f'{PIXELBRAIN_PATH}/assets/test_data/subset/**.*')]
    processed_imgs = preprocessor(test_images)
    for img in processed_imgs:
        assert img.shape[1] == 128 or img.shape[2] == 128