from pixelbrain.apps.face_similarity.face_similarity_scorer import FaceSimilartyScorer
import os
import pytest

module_path = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.slow_suit
def test_face_similarity_scorer():
    source_dir = f"{module_path}/data/source"
    compare_to_dir = f"{module_path}/data/compare_to"
    scorer = FaceSimilartyScorer(
        source_dir=source_dir,
        compare_to_dir=compare_to_dir,
        source_type="local",
        scoring_strategy="nearest",
    )
    results = scorer.process()
    assert isinstance(
        results, list
    ), "The result should be a list of image paths or cloudinary public ids."
    assert len(results) == 2, "The result list should contain 2 results."
    expected_first_result = f"{source_dir}/1_2.jpeg"
    assert (
        results[0] == expected_first_result
    ), f"Expected the first result to be {expected_first_result}, but got {results[0]}"


@pytest.mark.slow_suit
def test_face_similarity_scorer_with_cloudinary_source():
    compare_to_dir = "user_photos/test_score_id/raw"
    source_dir = (
        "user_images/generated_images/demo/6e90b780-4be7-4504-8f73-966e0dbcd44c"
    )
    scorer = FaceSimilartyScorer(
        source_dir=source_dir,
        compare_to_dir=compare_to_dir,
        source_type="cloudinary",
        scoring_strategy="nearest",
    )
    results = scorer.process()
    assert isinstance(
        results, list
    ), "The result should be a list of image paths or cloudinary public ids."
    assert len(results) == 2, "The result list should contain 2 results."
    expected_first_result = f"{source_dir}/1"
    assert (
        results[0] == expected_first_result
    ), f"Expected the first result to be {expected_first_result}, but got {results[0]}"


def strategies_experiment():
    """This utility scripts enables experimentation with different source and compare directories and scoring strategies"""
    import shutil
    import cloudinary
    import requests

    def run_experiment(source_dir, compare_dir, strategy, source_type):
        scorer = FaceSimilartyScorer(
            source_dir=source_dir,
            compare_to_dir=compare_dir,
            source_type=source_type,
            scoring_strategy=strategy,
        )
        results = scorer.process()
        return results

    source_compare_pairs = [
        (
            "/home/ubuntu/stable-diffusion-webui/outputs/txt2img-images/nir",
            "/home/ubuntu/images/nir/15_images/166_ohwx man",
        ),
        (
            "/home/ubuntu/stable-diffusion-webui/outputs/txt2img-images/omer",
            "/home/ubuntu/images/omer/166_ohwx man",
        ),
        (
            "/home/ubuntu/stable-diffusion-webui/outputs/txt2img-images/nurit",
            "/home/ubuntu/images/nurit/nurit",
        ),
    ]
    cloudinary_source_compare = [
        (
            "user_images/generated_images/demo/cd1be2ee-ed3d-4ff6-83fa-78dcd3d428bd",
            "user_photos/cd1be2ee-ed3d-4ff6-83fa-78dcd3d428bd",
        ),
        (
            "user_images/generated_images/demo/2d79da62-73d8-441a-93df-49de4d6e2f6a",
            "user_photos/2d79da62-73d8-441a-93df-49de4d6e2f6a",
        ),
        (
            "user_images/generated_images/demo/bf7bfd0e-5338-45e9-bfe3-ac6edb64dd59",
            "user_photos/bf7bfd0e-5338-45e9-bfe3-ac6edb64dd59",
        ),
    ]
    strategies = ["nearest", "average_k_nearest", "maximum_distance"]
    experiment_count = 1

    strategies_photos = {
        "local": source_compare_pairs,
        "cloudinary": cloudinary_source_compare,
    }
    for source_type in strategies_photos:
        for source_dir, compare_dir in strategies_photos[source_type]:
            for strategy in strategies:
                print(
                    f"Running experiment {experiment_count} with source: {source_dir}, compare: {compare_dir}, strategy: {strategy}, source_type: {source_type}"
                )
                results = run_experiment(source_dir, compare_dir, strategy, source_type)
                save_dir = f"experiment_results/source_{os.path.basename(source_dir)}_compare_{os.path.basename(compare_dir)}_strategy_{strategy}"
                os.makedirs(save_dir, exist_ok=True)

                for i, image_path in enumerate(results[:5]):
                    if source_type == "local":
                        # Save only the first 5 results and rename them to 1.png, 2.png, etc.
                        new_filename = f"{i+1}.png"
                        shutil.copy(image_path, os.path.join(save_dir, new_filename))
                    else:
                        # download from cloudinary
                        cloudinary_url = cloudinary.CloudinaryImage(
                            image_path
                        ).build_url()
                        response = requests.get(cloudinary_url)
                        if response.status_code == 200:
                            with open(os.path.join(save_dir, f"{i+1}.png"), "wb") as f:
                                f.write(response.content)
                        else:
                            print(
                                f"Failed to download image from Cloudinary using public ID: {image_path}"
                            )

                experiment_count += 1