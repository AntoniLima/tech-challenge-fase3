"""Testes das decisões de modelagem defendidas no README: vazamento, split, pipeline e limiar."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "src"))

from src.evaluation.avaliacao import escolher_limiar  # noqa: E402
from src.modeling.modelos import criar_pipeline  # noqa: E402
from src.preprocessing.pipeline import LimitarExtremos, carregar_decisoes, listar_features  # noqa: E402

DECISOES = carregar_decisoes()
FEATURES = listar_features(DECISOES)


@pytest.fixture(scope="module")
def gold():
    """Amostra versionada da Gold, lida uma vez para todos os testes."""
    return pd.read_parquet(RAIZ / "data" / "sample" / "features_aluno")


def preparar(amostra):
    """Monta X e y como na modelagem: binárias em float, alvo = risco (não alfabetizado)."""
    X = amostra[FEATURES].copy()
    for coluna in DECISOES["binarias"]:
        X[coluna] = X[coluna].astype(float)
    y = (~amostra["alfabetizado"]).astype(int)
    return X, y


def test_features_sem_vazamento():
    """Nenhuma feature é o alvo, um identificador ou uma medida da própria prova."""
    proibidas = {"alfabetizado", "proficiencia", "distancia_corte", "faixa_proximidade",
                 "alvo_taxa_2024", "id_aluno", "id_escola", "id_municipio"}
    assert not proibidas & set(FEATURES)


def test_split_nao_repete_municipio(gold):
    """O mesmo município nunca aparece no treino e no teste."""
    amostra = gold.sample(50_000, random_state=1)
    treino, teste = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                         .split(amostra, groups=amostra["id_municipio"]))
    assert set(amostra["id_municipio"].iloc[treino]).isdisjoint(amostra["id_municipio"].iloc[teste])


def test_teto_aprendido_so_no_treino():
    """O LimitarExtremos guarda o teto do fit e não se ajusta a um valor novo no teste."""
    treino = np.array([[1.0], [2.0], [3.0], [100.0]])
    transformador = LimitarExtremos(percentil_superior=0.5).fit(treino)
    assert transformador.transform(np.array([[1000.0]]))[0, 0] == np.nanquantile(treino, 0.5)


def test_limiar_atinge_recall_alvo():
    """O limiar escolhido sinaliza ao menos 70% dos casos de risco."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2_000)
    prob = np.clip(y * 0.3 + rng.random(2_000) * 0.7, 0, 1)
    limiar = escolher_limiar(y, prob, 0.70)["limiar"]
    assert ((prob >= limiar) & (y == 1)).sum() / (y == 1).sum() >= 0.70


def test_mesma_semente_mesmo_resultado(gold):
    """Treinar duas vezes com a mesma semente dá as mesmas probabilidades."""
    X, y = preparar(gold.sample(3_000, random_state=3))
    a = criar_pipeline("hist_gradient_boosting", DECISOES, max_iter=20).fit(X, y).predict_proba(X)
    b = criar_pipeline("hist_gradient_boosting", DECISOES, max_iter=20).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)
