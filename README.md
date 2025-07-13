# DLS_FaceRecognition_Project
Этот проект представляет собой реализацию полного пайплайна распознавания лиц, включающего:

    Детекцию лиц (face detection) — поиск лиц на изображении;

    Выравнивание лиц (face alignment) — определение ключевых точек и геометрическая нормализация;

    Распознавание лиц (face recognition) — извлечение эмбеддингов и сравнение лиц.

Проект выполнен в рамках первого курса Deep Learning School как итоговое задание.

✅ Для датасета Celeba in the wild реализовано:

Обучена модель Stacked Hourglass Network для поиска ключевых точек лица (5 landmarks). Выполнено лица на основе найденных landmarks (alignment по шаблону): 1_FaceAlignment.ipynb

Обучена модель face recognition на CE loss и на ArcFace loss (efficientnet b1 - бэкбон без предобучения на лицах): 2_ArcFace.ipynb

Собран полный пайплайн, принимающий изображение с несколькими лицами и возвращающий эмбеддинги: 3_PipeLine

📊 Метрики

    Точность классификации (на CE и ArcFace): 76.8% и 76.8%

Репозиторий также содержит:

    0_Preprocessing.ipynb - ноутбук для предобработки скаченного полного датасета;

    model_utils.py - файл архитектурой моделей и дополнительными функциями;

Обработанный датасет доступен по ссылке: https://drive.google.com/file/d/1jpn53D-06GJpleuVwOWPeWQngFSytv37/view?usp=drive_link
