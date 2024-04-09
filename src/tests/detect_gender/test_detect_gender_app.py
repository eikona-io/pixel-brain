from pixelbrain.apps.detect_gender.cloudinary_detect_gender_app import CloudinaryGenderDetector
from utils import assert_env_var_present


def test_cloudinary_gender_detector_app():
    assert assert_env_var_present("CLOUDINARY_URL"), "CLOUDINARY_URL is not set"

    test_user_cloudinary_prefix = "user_photos/1702595526432"
    app = CloudinaryGenderDetector(test_user_cloudinary_prefix, download_from_hf=True)
    prob_for_famale = app.process()
    assert prob_for_famale > 0.5, "Probability for female is less then 0.5 (subject IS a female)"