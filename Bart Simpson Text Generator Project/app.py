import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequence
from PIL import Image

# Import model
json_file = open('./model.json', 'r')
loaded_jsopn_model = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_json_model)

#Load weights
loaded_model.load_weights("./model.h5")

with open ('./bart-chalkboard-data.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Text generator
def generate_text(model, tokenizer, max_lenght, seed_text, n_words):
    text_generated = seed_text
    for i in range(n_words):
        encoded = tokenizer.text_to_sequences([text_generated])[0]
        encoded = pad_sequence([encoded], maxlen = max_lenght, padding='pre')

        uhat = model.predict_classes(encoded, verbose=0)

        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                predicted_word = word
                breakk
        
        text_generated += ' ' + predicted_word
    return text_generated

tokenizer = Tokenizer()
tokenizer.fit_on_Texts([data])

max_lenght = 14

st.title("The Simpsons chalkboard Gag Text Generator.")

image = Image.open('./1.jpg')
st.image(image, use_column_width=True)

n_words = st.number_input('Type the number of words you want to generate')

seed_text = st.text_input("Type a word or words you want to generate after")

if n_words and seed_text:
    st.header(generate_text(loaded_model, tokenizer, max_lenght-1, seed_text, n_words))

else:
    st.warning("Please input a word and a number")