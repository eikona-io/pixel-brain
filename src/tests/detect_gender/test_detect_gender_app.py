from pixelbrain.apps.detect_gender.detect_gender_app import GenderDetectorApp
from tests.test_utils import assert_env_var_present


def test_cloudinary_gender_detector_app():
    assert assert_env_var_present("CLOUDINARY_URL"), "CLOUDINARY_URL is not set"

    test_user_cloudinary_prefix = "user_photos/1702595526432"
    app = GenderDetectorApp(test_user_cloudinary_prefix, download_from_hf=True)
    prob_for_famale = app.process()
    assert (
        prob_for_famale > 0.5
    ), "Probability for female is less then 0.5 (subject IS a female)"


def test_s3_gender_detector_app():
    bucket_name = "images.selfyx.com"
    test_user_prefix = "user_photos/1702595526432"
    app = GenderDetectorApp(
        test_user_prefix,
        download_from_hf=True,
        source_type="s3",
        bucket_name=bucket_name,
    )
    prob_for_famale = app.process()
    assert (
        prob_for_famale > 0.5
    ), "Probability for female is less then 0.5 (subject IS a female)"
