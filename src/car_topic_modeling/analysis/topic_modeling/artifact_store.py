from typing import Any, Callable, Tuple
from bertopic import BERTopic
from car_topic_modeling.analysis.config.types import CacheMode, StagePaths
import logging

import joblib
import numpy as np

log = logging.getLogger(__name__)


class ArtifactStore:
    """
    Thin class that makes transparent the loading or generation and saving
    of the middle-steps in the Topic Modeling pipeline using BERTopic.
    It uses the BERTopicPaths interface, and takes care of the artifact
    generation, and files in .npy, .json and .bin format.
    """

    def exists(self, stage_paths: StagePaths) -> bool:
        return stage_paths.file.exists()

    def load(self, stage_paths: StagePaths) -> Tuple[Any, Any]:
        """
        Loads the artifact of the stage with its metadata (hyper-parameter
        configution for the artifact). Can be a BERTopic model or embeddings
        previously generated.
        """
        path = stage_paths.file

        if not stage_paths.meta.exists():
            raise FileNotFoundError(f"Missing metadata: {stage_paths.meta}")

        if path.suffix == ".npy":
            obj = np.load(path, allow_pickle=False)
        elif path.suffix == ".bin":
            obj = BERTopic.load(path)
        else:
            raise ValueError(f"Unsupported artifact type: {path.suffix}")

        meta: Any = joblib.load(stage_paths.meta)  # TODO: make it a Stage Config object
        return obj, meta

    def save(self, stage_paths: StagePaths, obj: Any, meta: Any) -> None:
        """
        Saves the object passed (BERTopic model/embeddings) along with
        its metadata, which is a JSON file containing the hyper-parameter
        configuration.
        """
        path = stage_paths.file
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, np.ndarray):
            np.save(path, obj)
        elif isinstance(obj, BERTopic):
            obj.save(path)
        else:
            raise TypeError(f"Cannot persist objects of type {type(obj)}")

        joblib.dump(meta, stage_paths.meta)


# TODO: make strongly-typed return functions
def run_or_load(
    store: ArtifactStore,
    stage_paths: StagePaths,
    meta: Any,
    *,
    compute_fn: Callable[[], Any] | None = None,
    mode: CacheMode = CacheMode.AUTO,
) -> Tuple[Any, Any]:
    """
    Return (artifact, meta) according to the selected *mode*.

    AUTO          : load if present, otherwise compute & save.
    FORCE_LOAD    : load; raise FileNotFoundError if absent.
    FORCE_COMPUTE : compute, overwrite existing file, then save.
    """
    file_exists = store.exists(stage_paths)

    if mode == CacheMode.FORCE_LOAD:
        if not file_exists:
            raise FileNotFoundError(stage_paths.file)
        log.info("Cache hit: %s", stage_paths.file)
        return store.load(stage_paths)

    elif mode == CacheMode.AUTO and file_exists:
        return store.load(stage_paths)

    if compute_fn is None:
        raise RuntimeError(
            "The compute function wasn't passed but "
            "the function expected to compute the artifact."
        )

    obj = compute_fn()
    store.save(stage_paths, obj, meta)
    log.info(f"Computed and cached the artifact at {stage_paths.file}")

    return obj, meta
