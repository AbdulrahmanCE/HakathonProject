from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
                  path('upload/', views.upload, name="upload"),
              ] + staticfiles_urlpatterns()