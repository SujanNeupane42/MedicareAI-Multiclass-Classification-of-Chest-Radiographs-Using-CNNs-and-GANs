from django.contrib.auth.models import User
from django.contrib import messages
from urllib import request
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, logout
import pickle
from PIL import Image
import torch
import torchvision.transforms as tt
import torch.nn as nn
import numpy as np
import os
from django.core.files.storage import FileSystemStorage
np.set_printoptions(suppress=True)


import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Classifier"
)
mycursor = mydb.cursor()

media = 'media'


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('pass')

        data = {
            'username': username,
            'password': password
        }

        if username == '' or password == '':
            return render(request, 'login.html', {'error': True, 'data': data})

        user = authenticate(username=username, password=password)

        if user is not None:
            return redirect('home')
        else:
            # messages.alert(request, 'Username or password is incorrect!')
            return render(request, 'login.html', {'noaccount': True, 'data': data})
    return render(request, 'login.html')


def logoutuser(request):
    logout(request)
    return render(request, 'login.html')


def signup(request):
    if request.method == 'POST':
        fname = request.POST.get('fname')
        lname = request.POST.get('lname')
        username = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('pass1')
        pass2 = request.POST.get('pass2')

        data = {
            'fname': fname,
            'lname': lname,
            'username': username,
            'email': email,
            'pass1': pass1,
            'pass2': pass2
        }
        if pass1 != pass2:
            messages.error(request, "Password doesn't match on both fields")
            return render(request, 'signup.html', {'data': data})

        if User.objects.filter(username=username):
            messages.error(request, "Username already exists!!!")
            return render(request, 'signup.html', {'data': data})

        if User.objects.filter(email=email):
            messages.error(request, "Email already exists!!!")
            return render(request, 'signup.html', {'data': data})

        if fname == '' or lname == '' or username == '' or email == '' or pass1 == '' or pass2 == '':
            return render(request, 'signup.html', {'error': True, 'data': data})

        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.save()
        messages.success(request, "Account created successfully")
        return render(request, 'login.html')

    return render(request, 'signup.html')


class Model:
    def __init__(self):

        with open('GoogLeNet_With_WGANGP.pkl', 'rb') as f:
            self.model = pickle.load(f)

        mean = 0.5199
        std = 0.2488
        image_size = 64
        CHANNELS_IMG = 1
        stats = [mean for _ in range(CHANNELS_IMG)], [std for _ in range(CHANNELS_IMG)]


        self.transformations_to_perform = transform=tt.Compose([
                                    tt.Grayscale(num_output_channels=1),
                                tt.Resize(image_size),
                                tt.ToTensor(),
                                tt.Normalize(*stats)])

        self.classes = ['Covid', 'Normal', 'Viral Pneumonia']

    def predict(self, img_path):
        img = self.transformations_to_perform(Image.open(img_path))
        grey_image = np.zeros([1, 224, 224], dtype = np.float64)
        grey_image[:,80:144, 80:144] = img
        grey_image = torch.from_numpy(grey_image)
        grey_image = grey_image.type(torch.FloatTensor)
        grey_image = grey_image.reshape((1, 1, 224, 224))

        a =  self.model(grey_image)
        out = nn.Softmax(dim = 1)(a)
        out = out.detach().numpy()[0]


        outputs = {'Covid': round(out[0] * 100, 5),
            'Normal': round(out[1] * 100, 5),
            'Viral_Pneumonia': round(out[2] * 100, 5)}
        
        selected_class = np.argmax(out)
        print("OUT: ", out)
        print(selected_class)

        if selected_class in [0, 2]:
            mycursor.execute("SELECT * FROM data where Class = {}".format(selected_class))
            myresult = list(mycursor.fetchall()[0])
            # print(myresult)
    
            description = myresult[1]
            symptoms = myresult[2]
            solutions = myresult[3]
            
            f = open('test.txt', 'w')
            f.write("Case: "+self.classes[selected_class]+"\n")
            f.write("Description: "+ description+"\n")
            f.write("Symptoms: "+ symptoms+"\n")
            f.write("Solutions: "+ solutions+"\n")
            f.write("Confidence: "+ str(out[selected_class]))
            f.close()

        return outputs


def home(request):

    if request.method == 'POST' and request.FILES['upload']:

        if 'upload' not in request.FILES:
            err = 'No images selected'
            return render(request, 'index.html', {'err': err})

        f = request.FILES['upload']

        if f == '':
            err = 'No files selected'
            return render(request, 'index.html', {'err': err})

        upload = request.FILES['upload']

        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)

        filename = 'media/'+ upload.name
        file_url = fss.url(file)
        model = Model()        
        predictions = model.predict(os.path.join(media, file))
        
        return render(request, "index.html", {'pred': predictions, 'file': upload, 'file_path': file_url,'show': True, 
                                             })
    else:
        return render(request, 'index.html')


def index(request):
    if request.user.is_anonymous:
        return render(request, 'login.html')
