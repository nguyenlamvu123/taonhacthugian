import scipy, os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

from mus_gen import Py_Transformer, sampling_rate


@csrf_exempt
def validate(request):
   if request.method == 'POST':
      olt = os.path.join(os.path.dirname(__file__), "musicgen_out.wav")
      outlocat = request.POST.get("output_location", olt)
      descri = list()
      descri1  = request.POST.get("descri1")
      if descri1 is not None: descri += [descri1]
      descri2 = request.POST.get("descri2")
      if descri2 is not None: descri += [descri2]
      descri3 = request.POST.get("descri3")
      if descri3 is not None: descri += [descri3]
      thoigian = request.POST.get("thoigian", 12)
      g_scale = request.POST.get("guidance_scale", 3)
      print("filename: ", outlocat)
      print("descri1: ", descri1)
      print("descri2: ", descri2)
      print("descri3: ", descri3)
      print("thoigian: ", thoigian)
      audio_values = Py_Transformer(input_text=descri, g_scale=int(g_scale), thoigian=int(thoigian))
      scipy.io.wavfile.write(outlocat, rate=sampling_rate, data=audio_values)
      return HttpResponse('ok')