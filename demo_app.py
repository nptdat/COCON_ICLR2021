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
# MODEL_NAME2ID = {
#     "English(gpt2-medium)": "en_gpt2_medium",
#     "日本語(rinna-gpt2-medium)": "ja_gpt2_rinna_medium",
#     "日本語(colorfulscoop-gpt2-small)": ""
# }


def generate(model_id, prompt_text, context, length):
    request = GenerationRequest(
        model_id=model_id,
        prompt_text=prompt_text,
        context=context,
        length=length
    )

    response = GenerationResponse(
        **requests.post(API_ENDPOINT, json=request.dict()).json()
    )
    return response.generated_text


if __name__ == "__main__":
    with open("config.yml", "rt") as f:
        cfg = Config(**yaml.load(f, yaml.SafeLoader))
    model_name2id = {
        m.model_display_name: m.model_id
        for m in cfg.models if m.enabled
    }

    model_name = st.sidebar.radio("Choose model", list(model_name2id.keys()))
    length = st.sidebar.selectbox(
        "Length", [2, 5, 10, 15, 20, 30, 50, 100], index=4
    )
    st.sidebar.markdown(
        "※Reference: [COCON paper](https://arxiv.org/abs/2006.03535)",
        unsafe_allow_html=True
    )

    model_id = model_name2id.get(model_name, "")
    if model_id == "":
        st.write("Not support yet")
    else:
        context = st.text_input('Conditioned context/topic', 'finance')
        prompt_text = st.text_input('Prompt text', 'In summary')
        if st.button("Generate text"):
            gen_text = generate(model_id, prompt_text, context, length)
            st.write(gen_text)
