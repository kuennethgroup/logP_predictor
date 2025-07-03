import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
from sentence_transformers import SentenceTransformer
from psmiles import PolymerSmiles as PS
import os
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st
import sys


# Hauptmodus merken
if "main_mode" not in st.session_state:
    st.session_state["main_mode"] = "Standard Mode (trained monomers)"

# Submodus merken
if "sub_mode" not in st.session_state:
    st.session_state["sub_mode"] = "Single Copolymer"

# Disable GPU for compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# **1. Page Settings**
st.set_page_config(
    page_title="Log P Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# **2. Cache Model Loading**
@st.cache_resource
def load_predictor():
    return TabularPredictor.load('./all_predictions_0601')


@st.cache_resource
def load_polybert():
    return SentenceTransformer('kuelumbus/polyBERT')

predictor = load_predictor()
polyBERT = load_polybert()

# Load scalers from training
with open('./scaler_logP.pkl', 'rb') as f:
    scaler_logP = pickle.load(f)
    
with open('./scaler_nmx.pkl', 'rb') as f:
    scaler_nmx = pickle.load(f)

# Monomer 1 
trained_monomer1 = "CC(C)NC(=O)C(-*)C-*"

# dict_monomer2 automatisch generieren wie im Original-Code
max_x = 10 
dict_monomer2 = []
for x in range(1, max_x+1):
    side_chain = ''
    for i in range(x-1):
        side_chain += 'CNCCN1CCCCC1C(C)O'
    if x == 1:
        chem_struct = '[*]CC([*])C(=O)NCCNC(C)O'
    else:
        chem_struct = f'[*]CC([*])C(=O)NCCN({side_chain})C(C)O'
    dict_monomer2.append({"x": x, "monomer2_psmiles": chem_struct})

# Hilfsfunktion: Copolymer-pSMILES generieren
def build_copolymer_psmiles(psmiles1, psmiles2, n, m):
    try:
        monomer1 = PS(psmiles1)
        monomer2 = PS(psmiles2)
        pattern = [0]*n + [1]*m
        copolymer = monomer1.linear_copolymer(monomer2, pattern=pattern)
        return str(copolymer)
    except Exception as e:
        return None

# **3. Custom 
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
            color: #333;
            font-family: Arial, sans-serif;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stApp {
            max-width: 1000px; 
            margin: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# **4. Sidebar Navigation**
st.sidebar.title("Navigation")
main_mode = st.sidebar.radio(
    "Choose Mode:",
    ["Standard Mode (trained monomers)", "Expert Mode (custom pSMILES)"]
)

sub_mode = st.sidebar.radio(
    "Choose Prediction Type:",
    ["Single Copolymer", "Table (Ranges)"]
)

# **5. Main Content Based on Navigation**
st.title("💊 Drug Delivery")
st.subheader("⚖️ Log P Analysis")
st.markdown("""
This app predicts the Log P value of copolymers based on monomer structure and composition.
""")

if main_mode == "Standard Mode (trained monomers)":
    st.markdown(
    '<p style="font-size:22px;">Standard Mode (trained monomers)</p>',
    unsafe_allow_html=True
    )

    st.info("Monomer 1 and Monomer 2 (for all x) are fixed as in the training set.")
    st.image("./copolymer.png", caption="The two monomers with n, m, x", use_column_width=True)

    if sub_mode == "Single Copolymer":
        st.subheader("Single Copolymer Prediction")
        n = st.number_input("n (number of Monomer 1 units)", min_value=1, max_value=100, value=10)
        m = st.number_input("m (number of Monomer 2 units)", min_value=1, max_value=100, value=10)
        x = st.number_input("x (variant of Monomer 2)", min_value=1, max_value=max_x, value=1)
        if n + m > 100:
            st.error("The sum of n and m must not exceed 100.")
        else:
            if st.button("Predict Log P (Standard)"):
                # Scale n, m, x using training scalers
                nmx_scaled = scaler_nmx.transform([[n, m, x]])  
                psmiles2 = dict_monomer2[x-1]["monomer2_psmiles"]
                copolymer_psmiles = build_copolymer_psmiles(trained_monomer1, psmiles2, n, m)
                
                if copolymer_psmiles:
                    copolymer_fps = polyBERT.encode(copolymer_psmiles)
                    
                    # Create proper input dataframe with scaled features
                    input_data = pd.DataFrame(
                        data=[nmx_scaled.flatten().tolist() + copolymer_fps.tolist()],
                        columns=['n_scaled', 'm_scaled', 'x_scaled'] + [str(i) for i in range(len(copolymer_fps))]
                    )
                    
                    # Predict and inverse transform
                    result_scaled = predictor.predict(input_data)
                    result = scaler_logP.inverse_transform(result_scaled.values.reshape(-1, 1))[0][0]  # Inverse scaling
                    
                    st.success(f"Predicted Log P value: {result:.2f}")
                    st.markdown(f"**Copolymer pSMILES:** `{copolymer_psmiles}`")

    elif sub_mode == "Table (Ranges)":
        st.subheader("Table Prediction (Standard)")
        col1, col2 = st.columns(2)
        with col1:
            start_n = st.number_input("Start value for n", min_value=1, max_value=100, value=1, step=1)
            end_n = st.number_input("End value for n", min_value=start_n, max_value=100, value=5, step=1)
        with col2:
            start_m = st.number_input("Start value for m", min_value=1, max_value=100, value=1, step=1)
            end_m = st.number_input("End value for m", min_value=start_m, max_value=100, value=5, step=1)
        start_x = st.number_input("Start value for x", min_value=1, max_value=max_x, value=1, step=1)
        end_x = st.number_input("End value for x", min_value=start_x, max_value=max_x, value=3, step=1)

        if st.button("Generate Table (Standard)"):
            results = []
            n_list, m_list, x_list, fp_list, smiles_list = [], [], [], [], []   

            for n in range(int(start_n), int(end_n)+1):
                for m in range(int(start_m), int(end_m)+1):
                    if n + m > 100:
                        continue
                    for x in range(int(start_x), int(end_x)+1):
                        psmiles2 = dict_monomer2[x-1]["monomer2_psmiles"]
                        copolymer_psmiles = build_copolymer_psmiles(trained_monomer1, psmiles2, n, m)
                        if copolymer_psmiles:
                            copolymer_fps = polyBERT.encode(copolymer_psmiles)
                            n_list.append(n)            
                            m_list.append(m)            
                            x_list.append(x)            
                            fp_list.append(copolymer_fps)  
                            smiles_list.append(copolymer_psmiles)  
            if n_list:   
                import numpy as np
                # Skalieren
                nmx = np.array(list(zip(n_list, m_list, x_list)))   
                nmx_scaled = scaler_nmx.transform(nmx)              
                fps = np.array(fp_list)                             
                columns = ['n_scaled', 'm_scaled', 'x_scaled'] + [str(i) for i in range(fps.shape[1])]  #!!!
                data = np.hstack([nmx_scaled, fps])                
                batch_df = pd.DataFrame(data, columns=columns)      

                result_scaled = predictor.predict(batch_df)         
                result = scaler_logP.inverse_transform(result_scaled.values.reshape(-1, 1)).flatten()   #!!!

                df_results = pd.DataFrame({
                    "n": n_list,
                    "m": m_list,
                    "x": x_list,
                    "copolymer_pSMILES": smiles_list,
                    "predicted_logP": result                        #!!!
                })
                st.dataframe(df_results)
                csv = df_results.to_csv(index=False)
                st.download_button("Download CSV", csv, "logP_predictions_standard.csv")
            else:
                st.warning("No valid copolymers generated in the given range.")


elif main_mode == "Expert Mode (custom pSMILES)":
    st.markdown(
    '<p style="font-size:22px;">Standard Mode (trained monomers)</p>',
    unsafe_allow_html=True
    )
    st.info("Enter your own pSMILES for both monomers. Predictions outside the training domain may be less reliable.")

    if sub_mode == "Single Copolymer":
        st.subheader("Single Copolymer Prediction")
        # psmiles_input1 = st.text_input("Monomer 1 (pSMILES)")
        # psmiles_input2 = st.text_input("Monomer 2 (pSMILES)")
        psmiles_input1 = st.text_input("Monomer 1 (pSMILES)", key="psmiles1")
        psmiles_input2 = st.text_input("Monomer 2 (pSMILES)", key="psmiles2")

        n = st.number_input("n (number of Monomer 1 units)", min_value=1, max_value=1000, value=10)
        m = st.number_input("m (number of Monomer 2 units)", min_value=1, max_value=1000, value=10)
        x = st.number_input("x (for consistency, any integer)", min_value=1, max_value=100, value=1)
        if n + m > 1000:
            st.error("The sum of n and m must not exceed 1000.")
        else:
            # Modify expert mode prediction
            if st.button("Predict Log P (Expert)"):
                if psmiles_input1 and psmiles_input2:
                    copolymer_psmiles = build_copolymer_psmiles(psmiles_input1, psmiles_input2, n, m)
                    st.write(f"Trying: n={n}, m={m}, x={x}, psmiles1={psmiles_input1}, psmiles2={psmiles_input2}")
                    copolymer_psmiles = build_copolymer_psmiles(psmiles_input1, psmiles_input2, n, m)
                    st.write(f"Resulting copolymer_psmiles: {copolymer_psmiles}")

                    if copolymer_psmiles:
                        # Scale features
                        nmx_scaled = scaler_nmx.transform([[n, m, x]])  #!!! x is required here too
                        
                        # Get fingerprints
                        copolymer_fps = polyBERT.encode(copolymer_psmiles)
                        
                        # Create input dataframe
                        input_data = pd.DataFrame(
                        data=[nmx_scaled.flatten().tolist() + copolymer_fps.tolist()],
                        columns=['n_scaled', 'm_scaled', 'x_scaled'] + [str(i) for i in range(len(copolymer_fps))]
                        )   
                        
                        # Predict and inverse transform
                        result_scaled = predictor.predict(input_data)
                        result = scaler_logP.inverse_transform(result_scaled.values.reshape(-1, 1))[0][0]
                        
                        st.success(f"Predicted Log P value: {result:.2f}")

    elif sub_mode == "Table (Ranges)":
        st.subheader("Table Prediction (Expert)")
        psmiles_input1 = st.text_input("Monomer 1 (pSMILES)")
        psmiles_input2 = st.text_input("Monomer 2 (pSMILES)")
        col1, col2 = st.columns(2)
        with col1:
            start_n = st.number_input("Start value for n", min_value=1, max_value=1000, value=1, step=1, key="exp_n1")
            end_n = st.number_input("End value for n", min_value=start_n, max_value=1000, value=5, step=1, key="exp_n2")
        with col2:
            start_m = st.number_input("Start value for m", min_value=1, max_value=1000, value=1, step=1, key="exp_m1")
            end_m = st.number_input("End value for m", min_value=start_m, max_value=1000, value=5, step=1, key="exp_m2")
        start_x = st.number_input("Start value for x", min_value=1, max_value=100, value=1, step=1, key="exp_x1")
        end_x = st.number_input("End value for x", min_value=start_x, max_value=100, value=3, step=1, key="exp_x2")

        if st.button("Generate Table (Expert)"):
            results = []
            n_list, m_list, x_list, fp_list, smiles_list = [], [], [], [], []   #!!!

            for n in range(int(start_n), int(end_n)+1):
                for m in range(int(start_m), int(end_m)+1):
                    if n + m > 1000:
                        continue
                    for x in range(int(start_x), int(end_x)+1):
                        if not (psmiles_input1 and psmiles_input2):
                            continue
                        copolymer_psmiles = build_copolymer_psmiles(psmiles_input1, psmiles_input2, n, m)
                        st.write(f"Trying: n={n}, m={m}, x={x}, psmiles1={psmiles_input1}, psmiles2={psmiles_input2}")
                        copolymer_psmiles = build_copolymer_psmiles(psmiles_input1, psmiles_input2, n, m)
                        st.write(f"Resulting copolymer_psmiles: {copolymer_psmiles}")

                        if copolymer_psmiles:
                            copolymer_fps = polyBERT.encode(copolymer_psmiles)
                            n_list.append(n)            
                            m_list.append(m)            
                            x_list.append(x)            
                            fp_list.append(copolymer_fps)  
                            smiles_list.append(copolymer_psmiles)  
            if n_list:   
                import numpy as np
                nmx = np.array(list(zip(n_list, m_list, x_list)))   
                nmx_scaled = scaler_nmx.transform(nmx)              
                fps = np.array(fp_list)                             
                columns = ['n_scaled', 'm_scaled', 'x_scaled'] + [str(i) for i in range(fps.shape[1])]  #!!!
                data = np.hstack([nmx_scaled, fps])                 
                batch_df = pd.DataFrame(data, columns=columns)      

                result_scaled = predictor.predict(batch_df)         
                result = scaler_logP.inverse_transform(result_scaled.values.reshape(-1, 1)).flatten()   #!!!

                df_results = pd.DataFrame({
                    "n": n_list,
                    "m": m_list,
                    "x": x_list,
                    "copolymer_pSMILES": smiles_list,
                    "predicted_logP": result                        
                })
                st.dataframe(df_results)
                csv = df_results.to_csv(index=False)
                st.download_button("Download CSV", csv, "logP_predictions_expert.csv")
            else:
                st.warning("No valid copolymers generated in the given range.")

