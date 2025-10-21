import joblib

model_path = 'concrete_strength_model_mlp.pkl'

try:
    model = joblib.load(model_path)
    print("مدل با موفقیت لود شد!")
except Exception as e:
    print("خطا در لود مدل:", e)
