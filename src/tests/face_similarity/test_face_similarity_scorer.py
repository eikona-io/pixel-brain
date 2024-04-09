from pixelbrain.apps.face_similarity.face_similarity_scorer import FaceSimilartyScorer
import os


module_path = os.path.dirname(os.path.abspath(__file__))

def test_face_similarity_scorer():
    source_dir = f"{module_path}/data/source"
    compare_to_dir = f"{module_path}/data/compare_to"
    scorer = FaceSimilartyScorer(
        source_dir=source_dir,
        compare_to_dir=compare_to_dir,
        source_type='local',
        scoring_strategy="nearest",
    )
    results = scorer.process()
    assert isinstance(results, list), "The result should be a list of image paths or cloudinary public ids."
    assert len(results) == 2, "The result list should contain 2 results."
    expected_first_result = f"{source_dir}/1_2.jpeg"
    assert results[0] == expected_first_result, f"Expected the first result to be {expected_first_result}, but got {results[0]}"