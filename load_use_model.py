import onnxruntime

# Load the ONNX model
def load_model_from_onnx(onnx_filename='final_model.onnx'):
    try:
        return onnxruntime.InferenceSession(onnx_filename)
    except:
        return None
def run_inference(onnx_session, input_data):
    try:
        output = onnx_session.run(None, {'input': [input_data]})
        return output[0][0]
    except Exception as e:
        print(f"Error running inference: {e}")
        return None

if __name__ == '__main__':
    onnx_session = load_model_from_onnx()
    if onnx_session is not None:
        input_data = input("Stavek: ")
        output = run_inference(onnx_session, input_data)
        print(output)
    else:
        print("Model yet not trained, come back when you train it.")