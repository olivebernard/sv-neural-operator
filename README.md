# Opérateur neuronal spectral pour volatilité stochastique

Ce projet implémente un Fourier Neural Operator (FNO) compact pour approximer des surfaces de prix d'options sous volatilité stochastique. La formulation cible l'EDP de Heston et ajoute des contraintes différentiables de non-arbitrage statique.

## Fonctionnalités

- convolution spectrale bidimensionnelle en PyTorch ;
- grille non uniforme de prix d'exercice de type Chebyshev ;
- génération de surfaces de référence inspirées de Heston ;
- perte relative avec monotonie en strike et convexité discrète ;
- diagnostics d'arbitrage et visualisation de surface.

Pour un call européen, la contrainte principale est

\[
\frac{\partial C}{\partial K}\leq 0.
\]

## Exécution

```bash
python -m pip install -e .
python examples/demo.py
pytest -q
```

Le démonstrateur écrit `outputs/price_surface.png` et `outputs/metrics.json`.
