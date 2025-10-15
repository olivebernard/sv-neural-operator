# Architecture et contraintes

Le réseau élève les champs d'entrée vers une dimension latente, applique plusieurs blocs spectraux et projette la représentation vers une surface de prix. Les modes de Fourier de basse fréquence sont appris explicitement, tandis qu'une connexion locale conserve les variations à petite échelle.

La grille de strike est densifiée autour de la monnaie par transformation de Chebyshev. La perte combine une erreur quadratique, une pénalité de monotonie issue d'une différence centrale et une pénalité de convexité.

Les sorties peuvent être évaluées sans réentraîner le réseau grâce au générateur déterministe de surface de contrôle.
