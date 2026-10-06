from ml_model import x, y, model_


def test_dataset():
    assert len(x) > 0
    assert len(y) > 0


def test_features():
    assert x.shape[1] == 4


def test_model():
    assert model is not None


def test_prediction():
    prediction = model.predict([x[0]])
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]
