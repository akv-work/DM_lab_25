# Отчёт по ЛР: понижение размерности датасета Mushroom

- Исходный размер X: (8124, 22)
- После one-hot + StandardScaler: (8124, 117)
- Класс 0 (съедобный): 4208
- Класс 1 (ядовитый): 3916

## 1. Автоэнкодер
- Лучшая 2D-архитектура: {'hidden_layers': (32, 16), 'activation': 'relu', 'max_iter': 500}, MSE = 0.34671
- Лучшая 3D-архитектура: {'hidden_layers': (32, 16), 'activation': 'relu', 'max_iter': 500}, MSE = 0.30319
- Активация ReLU, оптимизатор Adam, перебор глубины и ширины.

## 2. t-SNE
- Протестированные perplexity: [20, 30, 40, 50, 60]
- Инициализация: init='pca', предварительное сжатие до 50 компонент.

## 3. PCA (ручной из ЛР №1 vs sklearn)
- Ручной 2D: доля дисперсии 0.170165, потери 0.829835
- Ручной 3D: доля дисперсии 0.242201, потери 0.757799
- sklearn 2D: 0.170165
- sklearn 3D: 0.242201
- Ручной и sklearn PCA дают совпадающие доли дисперсии.

## 4. Визуализации
- pca_2d_comparison.png, pca_3d_comparison.png
- autoencoder_result.png
- tsne_2d_perplexity.png, tsne_3d_perplexity.png
- comparison_2d.png
