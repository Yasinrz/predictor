from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ConcretePredictSerializer, ConcreteOptimizeSerializer
import joblib
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution


model_path = 'concrete_strength_model_mlp.pkl'
model = joblib.load(model_path)
feature_names = ["cement","slag","flyash","water","superplasticizer","coarseaggregate","fineaggregate"]

param_ranges = [
    (102.0, 540.0),  # cement
    (0.0, 359.4),    # slag
    (0.0, 200.1),    # flyash
    (121.8, 247.0),  # water
    (0.0, 32.2),     # superplasticizer
    (801.0, 1145.0), # coarseaggregate
    (594.0, 992.6)   # fineaggregate
]

specific_weights = [3.15, 2.22, 2.85, 1.00, 1.20, 2.645, 2.66]


class ConcretePredictAPIView(APIView):
    def post(self, request):
        serializer = ConcretePredictSerializer(data=request.data)
        if serializer.is_valid():
            data = [serializer.validated_data[feat] for feat in feature_names]
            input_df = pd.DataFrame([data], columns=feature_names)
            try:
                predicted_csMPa = model.predict(input_df)[0]
            except Exception as e:
                return Response({"error": f"Prediction failed: {e}"}, status=500)
            return Response({"predicted_csMPa": round(predicted_csMPa,2)})
        else:
            return Response(serializer.errors, status=400)


class ConcreteOptimizeAPIView(APIView):
    def post(self, request):
        serializer = ConcreteOptimizeSerializer(data=request.data)
        if serializer.is_valid():
            budget = serializer.validated_data['budget']
            unit_prices = np.array([
                serializer.validated_data['price_cement'],
                serializer.validated_data['price_slag'],
                serializer.validated_data['price_flyash'],
                serializer.validated_data['price_water'],
                serializer.validated_data['price_superplasticizer'],
                serializer.validated_data['price_coarseaggregate'],
                serializer.validated_data['price_fineaggregate']
            ])

            def objective(inputs):
                # بررسی NaN و infinity
                if np.any(np.isnan(inputs)) or np.any(np.isinf(inputs)):
                    return np.inf
                input_df = pd.DataFrame([inputs], columns=feature_names)
                try:
                    predicted_csMPa = model.predict(input_df)[0]
                except Exception as e:
                    print(f"Prediction error: {e}")
                    return np.inf

                total_cost = np.sum(unit_prices * inputs)
                penalties = 0

                # اعمال محدودیت بودجه
                if total_cost > budget:
                    penalties += 1000

                # استخراج مقادیر
                x1, x2, x3, x4, x5, x6, x7 = inputs

                # تمام محدودیت‌های نمونه 
                if not (0.5 <= (x4 + x5) / x1 <= 1.5):
                    penalties += 1000
                if not (0.3 <= (x4 + x5) / (x1 + x2 + x3) <= 0.7):
                    penalties += 1000
                if not (0.07 <= (x4 + x5) / (x1 + x2 + x3 + x6 + x7) <= 0.1):
                    penalties += 1000
                if not (0.01 <= x5 / (x1 + x2 + x3) <= 0.05):
                    penalties += 1000
                if not (0 <= x2 / (x1 + x2 + x3) <= 0.5):
                    penalties += 1000
                if not (0 <= x3 / (x1 + x2 + x3) <= 0.5):
                    penalties += 1000
                if not (0 <= (x2 + x3) / (x1 + x2 + x3) <= 0.6):
                    penalties += 1000
                if not (3 <= (x6 + x7) / (x1 + x2 + x3) <= 7):
                    penalties += 1000
                if not (0.3 <= x7 / (x6 + x7) <= 0.55):
                    penalties += 1000

                # محدودیت وزن مخصوص
                specific_weight_sum = sum(x / w for x, w in zip(inputs, specific_weights))
                if abs(specific_weight_sum - 1000) > 1e-6:
                    penalties += 1000

                return -predicted_csMPa + penalties

            # اجرای differential evolution
            try:
                result = differential_evolution(objective, param_ranges, strategy='best1bin', maxiter=1000)
            except Exception as e:
                print(f"Optimization error: {e}")
                return Response({'error': 'Optimization failed due to internal error'}, status=500)

            if result.success:
                optimal_inputs = result.x
                optimal_output = model.predict(pd.DataFrame([optimal_inputs], columns=feature_names))[0]
                total_cost = np.sum(unit_prices * optimal_inputs)

                response = {feat: round(optimal_inputs[i],2) for i, feat in enumerate(feature_names)}
                response['predicted_csMPa'] = round(optimal_output,2)
                response['total_cost'] = round(total_cost,2)

                return Response(response)
            else:
                return Response({'error': 'Optimization failed'}, status=500)

        else:
            return Response(serializer.errors, status=400)
