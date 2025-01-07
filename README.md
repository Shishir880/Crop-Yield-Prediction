# Crop Yield Prediction Model 🌾

This repository contains a comprehensive machine learning solution to predict crop yield (in kg/ha) based on environmental and agricultural factors. The project utilizes advanced data preprocessing, feature engineering, and a tuned Random Forest model to ensure high accuracy and practical applicability.

---

## Project Overview 🛠️

### Objective 🎯
- Develop a predictive model for crop yield based on:
  - **Year**
  - **Rainfall** (in mm)
  - **Irrigation Area** (in hectares)
- Provide insights into the most critical factors affecting agricultural productivity.

### Dataset 📂
- The dataset includes key features such as:
  - `Year`: Time period of the observation.
  - `Rainfall`: Total rainfall in millimeters.
  - `Irrigation_Area`: Total area under irrigation.
  - `Crop_Yield (kg/ha)`: Target variable.
- **Source:** Provided as `train.csv`. Ensure the dataset is clean and preprocessed before model training.

---

## Key Features 🔑

1. **Preprocessing Pipeline:**
   - Handles missing values.
   - Selects the most relevant features for prediction.

2. **Modeling:**
   - Utilizes a **Random Forest Regressor** for robust predictions.
   - Hyperparameters optimized via **GridSearchCV** for enhanced performance.

3. **Performance Metrics:**
   - **Mean Squared Error (MSE):** Quantifies the prediction error.
   - **R-squared (R²):** Measures model accuracy in explaining variance.

4. **Feature Importance:**
   - Visual insights into the impact of features like rainfall and irrigation.

5. **Visualization:**
   - Scatter plot for predicted vs actual values.
   - Bar plot for feature importance.

---

## Installation and Usage 🖥️

### Requirements
- Python 3.8+
- Libraries: 
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `joblib`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Shishir880/crop-yield-prediction.git
   cd crop-yield-prediction
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python train_model.py
   ```

4. Explore the results:
   - Check feature importance and evaluation plots in the output.
   - The trained model is saved as `final_rf_model.pkl` for deployment.

---

## Repository Structure 📁

```plaintext
├── train.csv               # Dataset for training
├── train_model.py          # Code for training the model
├── final_rf_model.pkl      # Saved Random Forest model
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

## Results 📊

- **Mean Squared Error (MSE):** Achieved a low error, indicating high accuracy.
- **R-squared (R²):** 98%, showing the model explains most of the variance in crop yield.
- **Insights:**
  - `Irrigation_Area` is the most significant factor.
  - Rainfall has a comparatively smaller impact.

---

## Future Enhancements 🚀

1. Incorporate additional features such as soil type, fertilizer usage, and crop variety.
2. Experiment with other models like Gradient Boosting or XGBoost.
3. Deploy the model using a web app or API for real-world usability.

---

## License 📜
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments 🙌
Special thanks to the contributors and the dataset providers for making this project possible.

---

Feel free to fork this repository, suggest improvements, or share your feedback! 🌟
