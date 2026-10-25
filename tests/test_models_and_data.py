import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from fruitfresh.config import MODEL_SPECS, NUM_CLASSES  # noqa: E402
from fruitfresh.datasets import compute_class_weights, make_dataset  # noqa: E402
from fruitfresh.gradcam import gradcam_heatmap  # noqa: E402
from fruitfresh.models import build_model, set_trainable_stage  # noqa: E402

SIZE = (64, 64)


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_model_builds_trains_one_step_and_gradcam(name):
    model = build_model(name, image_size=SIZE, pretrained=False)
    assert model.output_shape == (None, NUM_CLASSES)
    assert [l.name for l in model.layers][-1] == "probs"
    x = np.random.uniform(0, 255, (8, *SIZE, 3)).astype("float32")
    y = np.random.randint(0, NUM_CLASSES, 8)
    assert np.allclose(model.predict(x, verbose=0).sum(1), 1, atol=1e-4)
    model.compile("adam", "sparse_categorical_crossentropy")
    assert np.isfinite(model.train_on_batch(x, y))
    cam, cls, probs = gradcam_heatmap(model, x[0])
    assert cam.ndim == 2 and 0 <= cam.min() and cam.max() <= 1 + 1e-6 and np.isclose(probs.sum(), 1, atol=1e-4)


def test_mobilenet_freezing_stages():
    m = build_model("mobilenet_v2", image_size=SIZE, pretrained=False)
    head = set_trainable_stage(m, "head")
    assert head["backbone_layers_trainable"] == 0 and head["trainable_params"] == 1280 * NUM_CLASSES + NUM_CLASSES
    ft = set_trainable_stage(m, "finetune", ft_layers=20)
    assert 0 < ft["backbone_weighted_layers_trainable"] <= ft["backbone_layers_trainable"] <= 20
    assert ft["backbone_batchnorm_trainable"] == 0 and ft["trainable_params"] > head["trainable_params"]


def test_class_weights_balance():
    w = compute_class_weights(np.array([0] * 80 + [1] * 20))
    assert w[1] > w[0] and np.isclose(80 * w[0], 20 * w[1])


def test_dataset_shapes_weights_and_order(tmp_path):
    from PIL import Image
    import pandas as pd

    rows = []
    for i in range(10):
        p = tmp_path / f"{i}.png"
        Image.fromarray(np.full((20, 30, 3), i * 20, dtype="uint8")).save(p)
        rows.append({"image_path": p.name, "label": i % 3})
    df = pd.DataFrame(rows)
    ev = make_dataset(df, tmp_path, image_size=SIZE, batch_size=4)                   # evaluation: keeps order
    xs = np.concatenate([x.numpy() for x, _ in ev])
    assert xs.shape == (10, *SIZE, 3) and xs.dtype == np.float32 and xs.max() <= 255
    assert [int(v) for v in xs[:, 0, 0, 0]] == [i * 20 for i in range(10)]
    tr = make_dataset(df, tmp_path, image_size=SIZE, batch_size=4, training=True, augment=True,
                      class_weights={0: 1.0, 1: 2.0, 2: 3.0})
    batch = next(iter(tr))
    assert len(batch) == 3 and batch[0].shape == (4, *SIZE, 3)                        # (x, y, sample_weight)
    assert sum(1 for _ in tr) == 2                                                      # drop_remainder
