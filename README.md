# DLS_FaceRecognition_Project
Этот проект представляет собой реализацию полного пайплайна распознавания лиц, включающего:

    Детекцию лиц (face detection) — поиск лиц на изображении;
    Выравнивание лиц (face alignment) — определение ключевых точек и геометрическая нормализация;
    Распознавание лиц (face recognition) — извлечение эмбеддингов и сравнение лиц.

Проект выполнен в рамках первого курса Deep Learning School как итоговое задание.

✅ Для датасета Celeba in the wild реализовано:

    Обучена модель Stacked Hourglass Network для поиска ключевых точек лица (5 landmarks). Выполнено лица на основе найденных landmarks (alignment по шаблону): 1_FaceAlignment.ipynb;
    Обучена модель face recognition на CE loss и на ArcFace loss (efficientnet b1 - бэкбон без предобучения на лицах): 2_ArcFace.ipynb;
    Собран полный пайплайн, принимающий изображение с несколькими лицами и возвращающий эмбеддинги: 3_PipeLine.
    Выполнен расчет Identification Rate Metric: 1_dop_Identification_Rate_Metric.ipynb


📊 Метрики

    Точность классификации (на CE и ArcFace): 77.4% и 86.9%

Репозиторий также содержит:

    0_Preprocessing.ipynb - ноутбук для предобработки скаченного полного датасета;
    models.py - файл c архитектурой моделей;
    utils.py - файл c дополнительными функциями (для 3 задания);

Обработанный датасет доступен по ссылке: https://drive.google.com/file/d/1tBZkHs2T_MtV9JbA1JpZpf9KHCGQlY_Y/view?usp=drive_link 
