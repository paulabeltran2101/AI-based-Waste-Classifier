# ♻️ Clasificación Inteligente de Residuos Industriales

Este proyecto desarrolla e implementa un sistema avanzado para la **clasificación automática de residuos industriales sólidos**. El objetivo es ofrecer una **solución de visión artificial robusta** capaz de trabajar en dos modos:

1.  **Clasificación en *Batch*:** A partir de un conjunto de imágenes estáticas.
2.  **Clasificación en Tiempo Real:** Utilizando la entrada de vídeo de una **cámara industrial o web** para la segregación instantánea de residuos.

La meta es optimizar los procesos industriales, reducir costes operativos y promover la sostenibilidad ambiental mediante la automatización inteligente.

![Visualización de las clases de residuos](images/grid_clases.png)

--- 

## 🎯 Metodología y Enfoques

Para la resolución del problema y la comparación de resultados, se exploraron y compararon tres enfoques principales, analizando la interpretabilidad, el rendimiento y los requerimientos computacionales:

| Enfoque | Descripción | Modelo Principal |
| :--- | :--- | :--- |
| **A (ML Clásico)** | Extracción manual de características (color y textura) y entrenamiento con modelos tradicionales de Machine Learning. | SVM, Random Forest, etc. |
| **B (Deep Learning)** | Desarrollo de un clasificador mediante el uso de una Red Neuronal Convolucional (CNN) preentrenada y *Fine-Tuning*. | CNN (DL) |
| **C (Híbrido - Ganador)** | Uso de la CNN preentrenada como **Feature Extractor** para generar vectores de características, seguidos del entrenamiento con modelos clásicos de ML. | CNN (Extractor) + SVM (Clasificador) |

---

## 📊 Resultados Destacados

El **Enfoque C (Híbrido)** resultó ser la solución óptima para un entorno de producción que requiere baja latencia:

* **Predicciones Rápidas:** El modelo final (CNN + **SVM**) permite obtener **predicciones rápidas**, un factor clave para la clasificación en tiempo real con vídeo.
* **Bajo Coste Computacional:** El entrenamiento y la inferencia con el modelo SVM son rápidos y no presentan un coste computacional elevado.
* **Alta Precisión:** Se obtuvieron altos índices de precisión en la clasificación, superando la complejidad de los residuos industriales.
* **Optimización Industrial:** La solución es directamente aplicable para el **aumento de la productividad** y la mejora en la tasa de reciclaje.

---

## 🧪 Tecnologías

El proyecto fue desarrollado en Python, utilizando las siguientes herramientas principales:

* **Lenguaje:** Python 3.x
* **Deep Learning:** TensorFlow / Keras (para la CNN y el *Feature Extractor*)
* **Machine Learning:** scikit-learn (para el modelo SVM, el clasificador final)
* **Visión Artificial:** **OpenCV** (para el manejo de *frames* de la cámara en tiempo real y preprocesamiento de imágenes)
* **Entorno:** Jupyter Notebook

---

## 📁 Estructura del Repositorio

```plaintext
📁 Proyecto-prediccion-fallos-maquinas
│
├── app/
│ └── app.py # Aplicación Streamlit
│
├── model/
│ └── best_norm&model_svm.pkl # Modelo SVM entrenado y testeado
│
├── data/ #Archivo UCI
│
├── images/
│ └── grid_clases.jpg # clases
│
├── notebooks/
│ └── FEATURES & MODEL.ipynb # Notebook del preprocesamiento, feature y entrenamiento
│ └── EDA.ipynb # Notebook de la parte de EDA
│
├── utils/
│ └── __init__.py # Inicialización
│ └── camera_control.py # Inicialización cámara
│ └── classifier.py # Predicción
│ └── feature_extractor.py # Feature Extractor
│ └── image_flow.py # Estado cámara
│ └── load_model.py # Carga de modelo
|
├── test_cam.py #test de cámara
├── requirements.txt # Dependencias del proyecto
└── README.md # Este archivo
```

---

### ⚙️ Instalación y ejecución de la aplicación

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/paulabeltran2101/AI-based-Waste-Classifier.git
   cd AI-based-Waste-Classifier

2. **Crear entorno virtual**
   ```bash
   python -m venv env
   .\env\Scripts\activate      # En Windows
   source env/bin/activate     # En macOS / Linux

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt 
   ```

4. **Ejecutar la app**
   ```bash
   streamlit run app/app.py
   ```
   
---

## 🚀 Mejoras Futuras

Las principales líneas de trabajo para la escalabilidad e industrialización del proyecto son:

1.  **Escalabilidad Industrial (Nuevas Clases):** Adaptar la aplicación para la clasificación de nuevas categorías de residuos de alto valor (ej. **chips electrónicos** o RAEE).
2.  **Ampliación del Dataset:** Aumentar la variabilidad y la cantidad de datos para mejorar la generalización, especialmente en el entrenamiento de los modelos Deep Learning.
3.  **Optimización del *Feature Extractor*:** Refinar la extracción de características para potenciar aún más las métricas obtenidas con los modelos clásicos de ML.
4.  **Integración de *Edge AI***: Implementar el modelo en hardware dedicado (*edge devices*) para una **reducción de latencia** crítica en entornos industriales.

