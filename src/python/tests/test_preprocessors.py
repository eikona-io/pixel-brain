import glob
from pixel_brain.pre_processors.resnet_preprocessor import ResnetPreprocessor
from pixel_brain.pre_processors.hog_detector import HogsDetectorPreprocessor
from torchvision.io import read_image

def test_resnet_preprocessor():
    preprocessor = ResnetPreprocessor()
    test_images = [read_image(img) for img in glob.glob('assets/test_data/**.*')]
    processed_imgs = preprocessor(test_images)
    for img in processed_imgs:
        assert img.shape == (3, 256, 256)
        
        
def test_hog_detector_preprocessor():
    preprocessor = HogsDetectorPreprocessor()
    test_images = [read_image(img) for img in glob.glob('assets/test_data/**.*')]
    processed_imgs = preprocessor(test_images)
    for img in processed_imgs:
        assert img.shape == (512, 512)