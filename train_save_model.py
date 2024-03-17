import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import skl2onnx
from skl2onnx.common.data_types import StringTensorType

# Train your model

# Save the trained model as an ONNX file
def save_model_to_onnx(final_model, onnx_filename='final_model.onnx'):
    initial_type = [('input', StringTensorType([None]))]  # Define the input type
    onnx_model = skl2onnx.convert_sklearn(final_model, initial_types=initial_type)
    with open(onnx_filename, 'wb') as f:
        f.write(onnx_model.SerializeToString())


if __name__ == '__main__':
    final_model = None
    # X_train, y_train = ...  # Load your training data
    # final_model = ...  # Define and train your final best model
    save_model_to_onnx(final_model)