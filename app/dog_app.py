from flask import Flask, render_template, url_for, redirect,  request, abort
from forms import PredictionForm
from flask import flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import secrets
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
import numpy as np
import plotly
from plotly.graph_objs import Bar, Histogram, Scatter, Heatmap
import json
import cv2 
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret string'
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///site.db'




db=SQLAlchemy(app)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False, default='Dog image detected')
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    content = db.Column(db.Text, nullable=False, default='The breed is X')
    author = db.Column(db.Integer, nullable=False, default='Ahmed')
    image_file = db.Column(db.String(20), nullable=False, default='/static/uploaded_pics/default.jpg')

    def __repr__(self):
        return f"Post('{self.title}', '{self.date_posted}')"

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

transform={
    'test':    transforms.Compose([
               transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               normalize
                ]),

    'train':   transforms.Compose([
               transforms.RandomRotation(10),
               transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               normalize
                ]),

    'valid':   transforms.Compose([
               transforms.RandomRotation(10),
               transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               normalize
                ])
        }

data_dir = os.path.join(app.root_path, 'data/dog_images/')

#data_dir='dog_images/'

data_transfer={x:datasets.ImageFolder(data_dir+x, transform=transform[x])
             for x in ['train','test','valid']}


# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

#use_cuda=False

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()

### TODO: Load the model weights with the best validation loss.
model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.parameters():
    param.requires_grad=False

#from collections import OrderedDict
dog_breeds=133
model_transfer.classifier[6]=nn.Linear(model_transfer.classifier[6].in_features,dog_breeds)

#model_transfer.=classifier

# load the model that got the best validation accuracy (uncomment the line below)
model_path = os.path.join(app.root_path, 'model/model_transfer.pt')
model_transfer.load_state_dict(torch.load(model_path))
if use_cuda:
    model_transfer = model_transfer.cuda()



# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(r"C:\Users\Ahmed\AppData\Local\conda\conda\envs\deep-learning\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")

class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].classes]

def predict_breed_transfer(img_path):
    # load the image and return the predicted breed
    img = Image.open(img_path)
    
    img_tensor = preprocess(img).cuda()
    img_tensor.unsqueeze_(0)
    
    model_transfer.eval()
    fc_out = model_transfer(img_tensor)
    value, index = torch.max(fc_out, 1)
    
    
    
    return class_names[np.asscalar(index.cpu().data[0].numpy())]

def VGG16_predict(img_path):
    '''
    Use pre-trained densenet161 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to densenet161 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    # loads RGB image as PIL.Image.Image type
    img = Image.open(img_path)
    
    img_tensor = preprocess(img).cuda()
    img_tensor.unsqueeze_(0)
    
    VGG16.eval()
    fc_out = VGG16(img_tensor)
    value, index = torch.max(fc_out, 1)
    
    
    
    return np.asscalar(index.cpu().data[0].numpy()) # predicted class index


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    prediction = VGG16_predict(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

@app.route("/")
@app.route("/home")
def home():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=2)
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/uploaded_pics', picture_fn)
    #picture_path = '/static/uploaded_pics/'+ picture_fn
    output_size = (125, 125)
    #i = Image.open(form_picture)
    #i.thumbnail(output_size)
    form_picture.save(picture_path)

    return picture_fn

@app.route("/predict", methods=['GET','POST'])
def predict():
	form=PredictionForm()
	if form.validate_on_submit():
		post=Post()
		if form.picture.data:
			picture_file = save_picture(form.picture.data)
			post.image_file = picture_file
			picture_path = os.path.join(app.root_path, 'static/uploaded_pics', picture_file)
			breed= predict_breed_transfer(picture_path)

			if dog_detector(picture_path):
				#flash('dog detected','success')
				#flash('Breed: '+breed,'success')
				post.title="Dog image detected"
				post.content="The breed is "+breed
			elif face_detector(picture_path):
				#flash('face detected','success')
				#flash('Breed: '+breed,'danger')
				post.title="Human image detected"
				post.content="The image resembles to  "+breed
			else:
				flash('no face or dog detected','danger')
				post.title="Unknown image detected"
				post.content="It is neither a dog nor a human"
		db.session.add(post)
		db.session.commit()

		return redirect(url_for('home'))
	return render_template('prediction.html', title='Prediction', form=form)

@app.route("/post/<int:post_id>")
def post(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('post.html', title=post.title, post=post)

@app.route("/post/<int:post_id>/delete", methods=['POST'])
#@login_required
def delete_post(post_id):
    post = Post.query.get_or_404(post_id)
    db.session.delete(post)
    db.session.commit()
    flash('Your post has been deleted!', 'success')
    return redirect(url_for('home'))

def predict_breed_transfer_topk(img_path,model,k):
    # load the image and return the predicted breed
    img = Image.open(img_path)
    
    img_tensor = preprocess(img).cuda()
    img_tensor.unsqueeze_(0)

    model.eval()
    fc_out = model(img_tensor)
    probability = F.softmax(fc_out.data,dim=1)
    value, index = torch.topk(probability, k)
    
    
    
    return [class_names[c] for c in index.cpu().data[0].numpy()], value.cpu().data[0].numpy()


@app.route("/post/<int:post_id>/graph1", methods=['GET', 'POST'])
#@login_required
def graph1(post_id):
    post = Post.query.get_or_404(post_id)
    image_file=os.path.join(app.root_path, 'static/uploaded_pics', post.image_file)
    breeds,values=predict_breed_transfer_topk(image_file,model_transfer,5)
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = values
    genre_names = breeds
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Top5 Breeds Prediction ',
                'yaxis': {
                    'title': "Probability"
                },
                'xaxis': {
                    'title': "Breed"
                },
                'orientation':'h'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('graph1.html', title=post.title, post=post,ids=ids, graphJSON=graphJSON)


if __name__ == '__main__':
    app.run(debug=True)