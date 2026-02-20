# safety.py
from nudenet import NudeClassifier

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = NudeClassifier()  # downloads model on first run
    return _classifier

def is_explicit(image_path: str, threshold: float = 0.7) -> bool:
    clf = get_classifier()
    res = clf.classify(image_path)  # {path: {"safe": p, "unsafe": p}}
    scores = res.get(image_path, {})
    unsafe = float(scores.get("unsafe", 0.0))
    return unsafe >= threshold