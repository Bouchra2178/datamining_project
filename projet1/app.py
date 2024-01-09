import requests 
import json
import streamlit as st
from streamlit_lottie import st_lottie
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    

lottie_coding = load_lottiefile("./animation/anim2.json")  # replace link to local lottie file
lottie_coding2 = load_lottiefile("./animation/anim4.json")  # replace link to local lottie file
lottie_coding3 = load_lottiefile("./animation/anim3.json")
lottie_coding6 = load_lottiefile("./animation/anim6.json")
# lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packa...")
