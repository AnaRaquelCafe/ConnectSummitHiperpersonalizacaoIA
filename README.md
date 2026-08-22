# NOVA+ HyperCommerce: Demo de Hiperpersonalização com IA✨

Projeto hands-on de hiperpersonalização com Machine Learning + IA Generativa.

<img width="500" height="287" alt="GatinhoPiano" src="https://github.com/user-attachments/assets/d42b3c0b-9a26-45e8-ab99-546958b9f589" />


## Arquitetura da demonstração ⚙️

`Histórico → Feature Engineering → XGBoost → Propensão → Gemini → Experiência Streamlit`

O notebook original já usa XGBoost para estimar propensão de conversão e usa o histórico recente para construir a narrativa generativa. O app transforma essa lógica em uma experiência visual de "adaptive commerce". 

## Estrutura do repositório 🗂️

```text
ConnectSummitHiperpersonalizacaoIA/
├── app_hiperpersonalizado.py
├── requirements.txt
├── ecommerce_hiperpersonalizacao_catalogo.csv
└── models/
    ├── propensity_model.joblib
    ├── model_metadata.json
    ├── genai_config.json
└── notebooks/
    ├── Hiperpersonalização_com_Machine_Learning_e_IA_Generativa.ipynb
```

## 1. Criar ambiente 🥳

```powershell
python -m venv meu_venv
.\meu_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Acessar os artefatos do modelo treinado em ambiente do Google Colab e as configurações do agente 📦

A aplicação acessa o binário do modelo clássico:

- `models/propensity_model.joblib`

E os artefatos da camada generativa:

- `genai_config.json`

### Importante sobre o Gemini 🔗

O Gemini usado no notebook é acessado via API (`google-genai`), portanto o modelo generativo não é salvo como um arquivo `.bin` local. O notebook utiliza `gemini-3.6-flash` por API.


## 3. Executar 🥝

```powershell
streamlit run app_hiperpersonalizado.py
```

## 4. IA Generativa ao vivo 🚀

No Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="SUA_CHAVE"
streamlit run app_hiperpersonalizado.py
```

📌 A mensagem principal: 

> **O ML decide o que oferecer. A IA generativa decide como transformar essa decisão em experiência.**
