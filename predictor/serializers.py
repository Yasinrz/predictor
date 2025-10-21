from rest_framework import serializers


class ConcretePredictSerializer(serializers.Serializer):
    cement = serializers.FloatField()
    slag = serializers.FloatField()
    flyash = serializers.FloatField()
    water = serializers.FloatField()
    superplasticizer = serializers.FloatField()
    coarseaggregate = serializers.FloatField()
    fineaggregate = serializers.FloatField()



class ConcreteOptimizeSerializer(serializers.Serializer):
    cement = serializers.FloatField()
    slag = serializers.FloatField()
    flyash = serializers.FloatField()
    water = serializers.FloatField()
    superplasticizer = serializers.FloatField()
    coarseaggregate = serializers.FloatField()
    fineaggregate = serializers.FloatField()
    budget = serializers.FloatField()
    # قیمت هر ماده جداگانه
    price_cement = serializers.FloatField()
    price_slag = serializers.FloatField()
    price_flyash = serializers.FloatField()
    price_water = serializers.FloatField()
    price_superplasticizer = serializers.FloatField()
    price_coarseaggregate = serializers.FloatField()
    price_fineaggregate = serializers.FloatField()
