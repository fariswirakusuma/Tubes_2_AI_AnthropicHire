"""Penjaga aturan tugas: lima classifier wajib from scratch (NumPy/Pandas saja)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from src.models import DecisionTree, KNN, LogisticRegression, NaiveBayes, SVM

MODEL_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
MODEL_CLASSES = [KNN, DecisionTree, LogisticRegression, SVM, NaiveBayes]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("path", sorted(MODEL_DIR.glob("*.py")), ids=lambda p: p.name)
def test_model_modules_do_not_use_sklearn(path):
    offenders = {
        module for module in _imported_modules(path) if module.split(".")[0] == "sklearn"
    }
    assert not offenders, f"{path.name} mengimpor sklearn: {sorted(offenders)}"


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_model_classes_are_independent_of_sklearn(model_cls):
    assert not any("sklearn" in base.__module__ for base in model_cls.__mro__)
    for method in ("fit", "predict", "predict_proba", "save", "load"):
        assert callable(getattr(model_cls, method))


@pytest.mark.parametrize("model_cls", MODEL_CLASSES, ids=lambda c: c.__name__)
def test_model_source_has_no_sklearn_reference(model_cls):
    assert "sklearn" not in inspect.getsource(inspect.getmodule(model_cls))
