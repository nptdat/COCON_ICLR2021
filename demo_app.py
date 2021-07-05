"""
Usage:
```
$ streamlit run demo_app.py
```

Then access `http://localhost:8501`
"""

import os

import streamlit as st
import yaml
import requests

from schema import Config, GenerationRequest, GenerationResponse


# Constants
API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://localhost:8000/api/generate")

def generate(prompt_text, context, length):
    request = GenerationRequest(
        prompt_text=prompt_text,
        context=context,
        length=length
    )

    response = GenerationResponse(
        **requests.post(API_ENDPOINT, json=request.dict()).json()
    )
    return response.generated_text


if __name__ == "__main__":
    lang = st.sidebar.radio("Choose language", ["English", "日本語"])
    length = st.sidebar.selectbox(
        "Length", [2, 5, 10, 15, 20, 30, 50, 100], index=4
    )

    if lang == "日本語":
        st.write("Not support yet")
    else:
        context = st.text_input('Conditioned context', 'finance')
        prompt_text = st.text_input('Prompt text', 'In summary')
        if st.button("Generate text"):
            gen_text = generate(prompt_text, context, length)
            st.write(gen_text)
