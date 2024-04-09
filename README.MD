# Presentation
Ce projet vise à développer un système de comptage de pièces présentent dans des images en utilisant deux approches principales : une basée sur la détection de contours et une autre basée sur la transformation de Hough pour détecter des cercles.

La dataset fournie pour ce projet présente une grande variance en termes de positionnement des pièces, d'éclairage et de texture de l'arrière-plan. Cela constitue un défi intéressant qui nous pousse à appliquer des étapes de prétraitement et à explorer plusieurs approches.

## Approche Contour

Dans cette approche, les étapes suivantes sont suivies :

- Redimensionnement des Images: Les images sont réduites d'un facteur de 4 pour réduire la complexité du traitement.
- Conversion en Niveaux de Gris: Les images sont converties en niveaux de gris pour simplifier le traitement ultérieur.
- Flou Gaussien: Un flou gaussien est appliqué pour réduire le bruit et rendre les contours plus lisses.
- Détection de Contours avec Canny: L'algorithme de détection de contours de Canny est appliqué pour extraire les bords des objets dans les images.
- Dilatation: Une opération de dilatation est effectuée sur l'image binaire résultante pour accentuer les contours.
- Détection des Contours avec cv2.findContours: La fonction cv2.findContours est utilisée pour détecter les contours des objets dans l'image.
- Filtrage des Contours: Les contours sont filtrés en fonction de critères tels que la taille minimale et maximale des contours ainsi que leur circularité.
- Précision: Cette approche a une précision de 31,52% et un MAE de 5.03 c-à-d environ 5 piéces sur l'ensemble du jeu de données.

### Exemples de résultats

![Contour-1](readme_pictures/contour-1.png)
![Contour-2](readme_pictures/contour-2.png)
![Contour-3](readme_pictures/contour-3.png)
![Contour-4](readme_pictures/contour-4.png)

## Approche Hough

Dans cette approche, les étapes suivantes sont suivies :

- Redimensionnement des Images: Les images sont également réduites d'un facteur de 4.
- Conversion en Niveaux de Gris: Les images sont converties en niveaux de gris comme précédemment.
- Flou Gaussien: Un flou gaussien est appliqué pour réduire le bruit.
- Transformation de Cercle de Hough: La transformation de cercle de Hough est utilisée pour détecter des cercles dans les images. Des critères de distance minimale et maximale ainsi que de rayon minimum et maximum sont spécifiés.
- Précision: Cette approche atteint une précision de 50,0% sur l'ensemble du jeu de données.

![Hough-1](readme_pictures/hough-1.png)
![Hough-2](readme_pictures/hough-2.png)
![Hough-3](readme_pictures/hough-3.png)
![Hough-4](readme_pictures/hough-4.png)


## Amélioration approche Hough

Le principal défi de l'approche Hough précédente était la tendance à générer plusieurs cercles pour une seule pièce dans certaines situations. Dans cette version améliorée, nous avons cherché à résoudre ce problème en introduisant une comparaison entre les cercles détectés, permettant ainsi d'éliminer les cercles qui se chevauchent.

Ce changement à grandement amélioré les performaces, la **précision** est maintenant de **78.26%** et la **MAE est de 1.67**.

![Houghv2-1](readme_pictures/houghv2-1.png)
![Houghv2-1](readme_pictures/houghv2-1.png)

## Limitations et pistes d'amélioration : 

- Optimiser l'acquisition de données : Bien que la dataset actuelle offre une variabilité intéressante, il serait bénéfique de capturer des images de pièces dans un environnement plus contrôlé, avec des arrière-plans et un éclairage optimal. Cela simplifierait la tâche de traitement.
- Intégrer le deep learning pour améliorer les performances : L'incorporation de techniques de deep learning, notamment les réseaux de neurones convolutionnels (CNN), pourrait potentiellement accroître la précision du système de détection de pièces.
- Diversifier les techniques de prétraitement d'image : En plus du flou gaussien et de la conversion en niveaux de gris, il serait judicieux d'explorer d'autres méthodes de prétraitement d'image pour améliorer la qualité des données en entrée. Cela pourrait inclure des techniques telles que l'égalisation d'histogramme, la normalisation des couleurs ou l'amélioration du contraste.

## Conclusion

En comparant les deux approches, nous observons que la détection basée sur la transformation de Hough atteint une meilleure précision par rapport à celle basée sur les contours. Malgré que l'approche contour semble mieux reconnaitre les pièces prise sous des angles larges.. Chaque approche a donc ses avantages et ses limitations, et le choix dépend des exigences spécifiques du projet et des caractéristiques des données. Ce projet illustre l'importance de comprendre et d'explorer différentes techniques de détection d'objets pour obtenir les meilleurs résultats possibles.