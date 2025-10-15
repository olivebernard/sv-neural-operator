# Contribution

Les changements doivent conserver la reproductibilité du démonstrateur et ajouter un test ciblé pour toute modification numérique.

Avant une proposition de changement :

```bash
make install
make test
make demo
```

Documenter explicitement les conventions d'unités, l'ordre des axes et les hypothèses de discrétisation. Éviter les dépendances lourdes qui ne sont pas nécessaires au chemin d'exécution principal.
