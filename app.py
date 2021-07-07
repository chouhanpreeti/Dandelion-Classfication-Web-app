import numpy as np
import streamlit as st 
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import urllib

classifier = load_model("/home/preeti/Documents/my_app/modelv1.h5")
 

def load_image(image_file):
    img = Image.open(image_file)
    i = str(np.random.randint(100))
    name='image'+i+'.jpg'
    img.save(name)
    image_path='/home/preeti/Documents/my_app/'+name
    return image_path

def predict_flower(input_image):

    img_path=load_image(input_image)

    img = image.load_img(img_path, target_size=(150,150))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    pred = classifier.predict(images)
        
    if pred[0]<0.5:
        prediction='Dandelion'
    else:
        prediction='Grass'
            
    return prediction



@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = "https://github.com/itspreeti25/Dandelion-Classfication-Web-app/blob/main/" + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def main():

    st.title("Streamlit Demo: The Dandelion Classifier")

    html_temp = """
    <div style="background-color:orange;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Based Classifier ML App </h2>
    </div>

    <h3 style="color:green;"> This is a Dandelion Classification Deep Learning based web application built using Streamlit(https://streamlit.io) app and deployed on Heroku.
    We can upload an image and then see if its a Dandelion flower or not. </h3> <br>
    <h3 style="color:green;">The Datatset is from : https://www.kaggle.com/coloradokb/dandelionimages</h3>
    <br>
    <h3 style="color:green;"> The complete demo is [implemented in less than 100 lines of Python](https://github.com/itspreeti25/Dandelion-Classfication-Web-app/edit/main/app.py) and illustrates all the major building blocks of Streamlit.</h3>
    """

    st.markdown(html_temp,unsafe_allow_html=True)

    st.sidebar.title("What to do!")
    app_mode = st.sidebar.selectbox("Choose the app mode",["Show instructions", "Show the source code"])
    
    if app_mode == "Show instructions":
        st.sidebar.success('To continue upload an image.')

    elif app_mode == "Show the source code":
        st.code(get_file_content_as_string("app.py"))


    uploaded_file = st.file_uploader("Upload Files",type=['png','jpg'], accept_multiple_files=False)
    
    result=""

    if st.button("Predict"):
        if uploaded_file is not None:
            result=predict_flower(uploaded_file)
            st.success('The image uploaded is {}'.format(result))
        else:
            st.text('No image uploaded. Please upload an image')

    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()