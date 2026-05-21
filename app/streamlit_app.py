import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import tensorflow as tf
import cv2
import matplotlib.cm as cm

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="PneumoVision AI",
    page_icon="🫁",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

body {
    background-color: #050816;
}

.main {
    background: linear-gradient(
        135deg,
        #050816,
        #071126,
        #020617
    );
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.title {
    text-align: center;
    font-size: 68px;
    font-weight: 900;
    color: #00ffd5;
    letter-spacing: 4px;
    text-shadow: 0px 0px 30px #00ffd5;
    margin-bottom: 0px;
}

.subtitle {
    text-align: center;
    color: #b8c1ec;
    font-size: 18px;
    margin-top: -10px;
    margin-bottom: 25px;
}

.glass {
    background: rgba(255,255,255,0.05);
    border-radius: 25px;
    padding: 25px;
    border: 1px solid rgba(0,255,213,0.2);
    box-shadow: 0px 0px 25px rgba(0,255,213,0.15);
}

.result-box {
    text-align: center;
    padding: 20px;
    border-radius: 20px;
    font-size: 30px;
    font-weight: bold;
}

.normal {
    background-color: rgba(0,255,140,0.15);
    color: #00ff99;
    border: 1px solid #00ff99;
}

.pneumonia {
    background-color: rgba(255,0,85,0.15);
    color: #ff4b6e;
    border: 1px solid #ff4b6e;
}

.model-box {
    text-align: center;
    padding: 12px;
    border-radius: 15px;
    background: rgba(255,255,255,0.04);
    margin-top: 15px;
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.markdown(
    """
    <div class="title">
        PNEUMOVISION AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI Powered Chest X-ray Analysis System for Early Pneumonia Detection
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:

    selected_model = option_menu(
        menu_title="AI Model Selection",
        options=[
            "Transfer Learning",
            "Basic CNN"
        ],
        icons=[
            "cpu-fill",
            "diagram-3-fill"
        ],
        default_index=0
    )

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

transfer_model = load_model(
    "models/pneumovision_transfer_model.h5"
)

cnn_model = load_model(
    "models/pneumovision_cnn_model.h5"
)

# ------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------

if selected_model == "Transfer Learning":

    model = transfer_model
    model_name = "MobileNetV2 Transfer Learning"

else:

    model = cnn_model
    model_name = "Basic CNN Model"



# ------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------

def predict_xray(img):

    img = img.convert("RGB")

    img = img.resize((224,224))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    confidence = prediction[0][0]

    if confidence > 0.5:

        result = "PNEUMONIA"

    else:

        result = "NORMAL"

        confidence = 1 - confidence

    return result, confidence

# ------------------------------------------------
# GRAD CAM FUNCTION
# ------------------------------------------------

def generate_gradcam(img, model):


    img = img.convert("RGB")

    img = img.resize((224,224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # access MobileNetV2 base model
    base_model = model.layers[0]

    last_conv_layer = base_model.get_layer(
        "Conv_1"
    )

    grad_model = tf.keras.models.Model(

        inputs=base_model.input,

        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            img_array
        )

        loss = tf.reduce_mean(
            predictions
        )

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(
        heatmap,
        0
    )

    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(
        heatmap,
        (224,224)
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cm.jet(
        heatmap
    )[:,:,:3]

    original = np.array(
        img.resize((224,224))
    ) / 255.0

    superimposed_img = (
    heatmap * 0.4
) + original

    superimposed_img = np.clip(
    superimposed_img,
    0,
    1
)

    return superimposed_img
    
# ------------------------------------------------
# MAIN SECTION
# ------------------------------------------------

left, right = st.columns([1,1])

# ------------------------------------------------
# LEFT SECTION
# ------------------------------------------------

with left:

    st.markdown(
        """
        <div class="glass">
        <h2 style='text-align:center;color:#00ffd5;'>
        Upload Chest X-ray
        </h2>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        " ",
        type=["jpg","jpeg","png","webp"]
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

# ------------------------------------------------
# RIGHT SECTION
# ------------------------------------------------

with right:

    st.markdown(
        f"""
        <div class="glass">

        <h1 style='text-align:center;color:#00ffd5;'>
        AI Diagnostic Dashboard
        </h1>

        <p style='text-align:center;color:#b8c1ec;'>
        Current Model:
        <b>{model_name}</b>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------
# PREDICTION SECTION
# ------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    result, confidence = predict_xray(image)

    col1, col2 = st.columns([1,1.2])

    # --------------------------------------------
    # IMAGE
    # --------------------------------------------

    with col1:

        st.markdown(
            """
            <div class="glass">
            """,
            unsafe_allow_html=True
        )

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            width=350
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # --------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------

    with col2:

        st.markdown(
            """
            <div class="glass">
            """,
            unsafe_allow_html=True
        )

        if result == "PNEUMONIA":

            st.markdown(
                """
                <div class="result-box pneumonia">
                PNEUMONIA DETECTED
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-box normal">
                NORMAL CHEST X-RAY
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------
        # CONFIDENCE GAUGE
        # ----------------------------------------

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(confidence*100),
                title={
                    'text': "Confidence Score"
                },
                gauge={
                    'axis': {
                        'range': [0,100]
                    },

                    'bar': {
                        'color': "#00ffd5"
                    },

                    'steps': [

                        {
                            'range':[0,50],
                            'color':"#3b0a45"
                        },

                        {
                            'range':[50,75],
                            'color':"#2c5364"
                        },

                        {
                            'range':[75,100],
                            'color':"#00ffd5"
                        }
                    ]
                }
            )
        )

        fig.update_layout(
            paper_bgcolor="#0b1120",
            font={
                'color': "white",
                'size': 18
            },
            height=320
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------
        # MODEL INFO
        # ----------------------------------------

        st.markdown(
            f"""
            <div class="model-box">

            <h3>
            Current AI Model
            </h3>

            <h2 style='color:#00ffd5;'>
            {model_name}
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------
        # GRAD CAM
        # ----------------------------------------
    if selected_model == "Transfer Learning":
        st.markdown(
            """
            <h2 style='text-align:center;color:#00ffd5;margin-top:25px;'>
            Grad-CAM Visualization
            </h2>
            """,
            unsafe_allow_html=True
        )

        gradcam_img = generate_gradcam(
            image,
            model
        )

        st.image(
            gradcam_img,
            caption="AI Attention Heatmap",
            use_container_width=500
        )

    else:

        st.info(
        "Grad-CAM visualization available for Transfer Learning model only."
    )

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align:center;color:gray;font-size:15px;'>

    PneumoVision AI • Deep Learning Powered Medical Imaging System

    </div>
    """,
    unsafe_allow_html=True
)

