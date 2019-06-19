from django.shortcuts import render
from .machine_learn import Predict
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.models import load_model

# Create your views here.
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage


def upload(request):
    predict = None
    if request.method == "POST":
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)
        ml = Predict()

        img = ml.load_image('media/{}'.format(uploaded_file.name))

        predict = ml.prediction(img)

        os.remove('media/{}'.format(uploaded_file.name))
    return render(request, 'selectorsExercise.html', {'name': predict})
