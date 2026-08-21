# ============================================================
# NOVA+ HYPERCOMMERCE
# ============================================================
#
# Demo de Hiperpersonalização com:
#
#   Machine Learning
#          ↓
#   Propensão de conversão
#          ↓
#   Produto candidato
#          ↓
#   IA Generativa
#          ↓
#   Experiência personalizada
#
# Estrutura da aplicação:
#
# 01. Importações
# 02. Configuração de arquivos
# 03. Features do modelo
# 04. Configuração do Streamlit
# 05. Identidade visual
# 06. Carregamento dos dados
# 07. Carregamento do modelo
# 08. Configuração da IA
# 09. Inicialização
# 10. Funções auxiliares
# 11. IA Generativa
# 12. Simulador de cliente
# 13. Hero
# 14. Abas
# 15. Experiência do cliente
# 16. Motor de hiperpersonalização
# 17. Racional técnico
# 18. Jornada em tempo real
# 19. Propensão
# 20. Pipeline de IA
# 21. Bastidores técnicos
# 22. Footer
#
# ============================================================


# ============================================================
# 01. IMPORTAÇÕES
# ============================================================

from pathlib import Path
import json
import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# GEMINI
# ============================================================
#
# Gemini é opcional.
#
# Caso a biblioteca não esteja instalada, o restante da
# aplicação continua funcionando.
# ============================================================

try:
    from google import genai
except ImportError:
    genai = None


# ============================================================
# 02. CONFIGURAÇÃO DOS ARQUIVOS
# ============================================================

# Diretório onde o app.py está localizado.
ROOT = Path(__file__).resolve().parent


# Dataset do e-commerce.
DATA = ROOT / "ecommerce_hiperpersonalizacao_catalogo.csv"


# Pasta onde estão os modelos e configurações.
MODELS = ROOT / "models"


# ============================================================
# 03. FEATURES UTILIZADAS PELO MODELO
# ============================================================
#
# Essas variáveis precisam estar na mesma estrutura esperada
# pelo modelo salvo em:
#
# models/propensity_model.joblib
#
# ============================================================

FEATURES = [
    "idade",
    "tipo_dispositivo",
    "origem",
    "categoria_principal",
    "paginas_visualizadas",
    "produtos_visualizados",
    "cliques_produto",
    "realizou_busca",
    "qtd_buscas",
    "produtos_carrinho",
    "visitou_checkout",
    "tempo_sessao_min",
    "compras_anteriores",
    "sessoes_anteriores",
    "dias_ultima_compra",
    "recebeu_campanha",
    "utilizou_cupom",
    "preco_medio_produto",
    "valor_carrinho",
    "qtd_eventos_30d",
    "qtd_categorias_interesse_30d",
]


# ============================================================
# 04. CONFIGURAÇÃO DO STREAMLIT
# ============================================================

st.set_page_config(
    page_title="NOVA+ | HyperCommerce",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 05. IDENTIDADE VISUAL
# ============================================================

st.markdown(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);


/* ==========================================================
   CONFIGURAÇÃO GERAL
========================================================== */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}


.stApp {

    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(135, 92, 255, 0.20),
            transparent 30%
        ),

        radial-gradient(
            circle at 10% 15%,
            rgba(0, 210, 255, 0.10),
            transparent 25%
        ),

        #070910;

    color: #F8F9FC;
}


/* ==========================================================
   CONTAINER PRINCIPAL
========================================================== */

.block-container {

    max-width: 1450px;

    padding-top: 3.5rem;

    padding-bottom: 4rem;

    padding-left: 3rem;

    padding-right: 3rem;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {

    background: #090C14;

    border-right:
        1px solid rgba(255,255,255,.06);
}


/* ==========================================================
   TABS
========================================================== */

button[data-baseweb="tab"] {

    font-size: 16px !important;

    font-weight: 700 !important;

    color: #8F98AA !important;
}


button[data-baseweb="tab"][aria-selected="true"] {

    color: #C3B2FF !important;
}


/* ==========================================================
   TEXTOS
========================================================== */

.small-label {

    color: #A98BFF;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 2px;

    text-transform: uppercase;

    margin-bottom: 12px;
}


.big-title {

    font-size: clamp(
        36px,
        4vw,
        52px
    );

    line-height: 1.08;

    font-weight: 800;

    letter-spacing: -1.5px;

    max-width: 1100px;

    overflow: visible;

    white-space: normal;

    word-break: normal;

    margin-bottom: 18px;
}


.subtitle {

    color: #A3ADBE;

    font-size: 16px;

    line-height: 1.6;

    max-width: 850px;
}


/* ==========================================================
   CARDS
========================================================== */

[data-testid="stVerticalBlockBorderWrapper"] {

    border-radius: 20px !important;
}


/* ==========================================================
   BOTÕES
========================================================== */

.stButton > button {

    width: 100%;

    border-radius: 12px;

    min-height: 44px;

    font-weight: 700;

    border:
        1px solid rgba(255,255,255,.12);

    background: #171B28;

    color: white;
}


.stButton > button:hover {

    border-color: #A98BFF;

    color: white;
}


/* ==========================================================
   MÉTRICAS
========================================================== */

[data-testid="stMetric"] {

    background:
        rgba(18, 22, 34, .85);

    border:
        1px solid rgba(255,255,255,.08);

    padding: 18px;

    border-radius: 18px;
}


[data-testid="stMetricLabel"] {

    color: #8F98AA !important;
}


[data-testid="stMetricValue"] {

    color: #F8F9FC !important;
}


/* ==========================================================
   EXPANDER
========================================================== */

.streamlit-expanderHeader {

    font-weight: 700;
}


/* ==========================================================
   DIVISORES
========================================================== */

hr {

    border-color:
        rgba(255,255,255,.08);
}


/* ==========================================================
   RESPONSIVIDADE
========================================================== */

@media (max-width: 900px) {

    .block-container {

        padding-left: 1.2rem;

        padding-right: 1.2rem;

        padding-top: 2rem;
    }


    .big-title {

        font-size: 36px;

        line-height: 1.1;
    }
}


</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 06. CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA)


# ============================================================
# 07. CARREGAMENTO DO MODELO
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODELS / "propensity_model.joblib"
    )


# ============================================================
# 08. CONFIGURAÇÃO DA IA GENERATIVA
# ============================================================
#
# IMPORTANTE:
#
# Não existe mais:
#
# narrative_cache.json
#
# A narrativa é sempre gerada pelo Gemini no momento
# em que o usuário clicar no botão.
#
# O único arquivo opcional aqui é:
#
# models/genai_config.json
#
# Ele serve apenas para informar qual modelo Gemini
# será utilizado.
#
# ============================================================

@st.cache_data
def load_genai_config():

    config_path = (
        MODELS / "genai_config.json"
    )

    config = {}


    if config_path.exists():

        try:

            config = json.loads(
                config_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            config = {}


    return config


# ============================================================
# 09. INICIALIZAÇÃO
# ============================================================

df = load_data()

model = load_model()

genai_config = load_genai_config()


# ============================================================
# 10. FUNÇÕES AUXILIARES
# ============================================================


# ------------------------------------------------------------
# LOG DE ACESSO
# ------------------------------------------------------------

def parse_log(raw):
    """
    Converte o log de acesso armazenado em JSON
    para uma lista Python.

    Caso o conteúdo não seja válido, retorna uma
    lista vazia.
    """

    try:

        return json.loads(raw)

    except Exception:

        return []


# ------------------------------------------------------------
# FORMATAÇÃO MONETÁRIA
# ------------------------------------------------------------

def money(value):
    """
    Converte um número para o formato monetário brasileiro.

    Exemplo:

    149.90 → R$ 149,90
    """

    return (
        f"R$ {float(value):,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


# ------------------------------------------------------------
# RECUPERAÇÃO DO CLIENTE
# ------------------------------------------------------------

def get_customer(cid):
    """
    Retorna todas as observações disponíveis
    para o cliente selecionado.
    """

    return df[
        df["cliente_id"] == cid
    ].copy()


# ------------------------------------------------------------
# PROPENSÃO
# ------------------------------------------------------------

def customer_score(customer):
    """
    Executa o modelo de Machine Learning.

    O modelo retorna a probabilidade da classe positiva
    e calculamos a média das observações daquele cliente.
    """

    return float(
        model.predict_proba(
            customer[FEATURES]
        )[:, 1].mean()
    )


# ------------------------------------------------------------
# PRODUTO CANDIDATO
# ------------------------------------------------------------

def candidate_product(customer):
    """
    Identifica o produto de maior interesse do cliente.

    Utilizamos a moda para encontrar o produto que aparece
    com maior frequência nas observações do cliente.
    """

    values = (
        customer["produto_maior_interesse"]
        .dropna()
        .astype(str)
    )


    if values.empty:

        return "Produto personalizado"


    return values.mode().iloc[0]


# ============================================================
# 11. IA GENERATIVA
# ============================================================
#
# Essa função chama o Gemini diretamente.
#
# NÃO existe cache de narrativa.
#
# Cada clique no botão:
#
#    BOTÃO
#      ↓
#    GEMINI
#      ↓
#    NOVA NARRATIVA
#
# ============================================================

def live_narrative(
    row,
    product,
    probability
):

    # --------------------------------------------------------
    # Recupera a API Key
    # --------------------------------------------------------

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )


    # --------------------------------------------------------
    # Verifica disponibilidade do Gemini
    # --------------------------------------------------------

    if not api_key:

        return None, (
            "A variável de ambiente GEMINI_API_KEY "
            "não está configurada."
        )


    if genai is None:

        return None, (
            "A biblioteca google-genai não está instalada. "
            "Instale com: pip install google-genai"
        )


    # --------------------------------------------------------
    # Recupera os eventos mais recentes
    # --------------------------------------------------------

    events = parse_log(
        row["log_acesso"]
    )[-12:]


    # ========================================================
    # DEFINIÇÃO DO TOM
    # ========================================================

    if probability >= 0.80:

        tone = """
A propensão de conversão é alta.

Seja mais confiante e orientado à conversão.

Apresente o produto como uma escolha bastante
relevante para este momento.

Inclua uma chamada para ação clara.

Não seja agressivo, apelativo ou manipulativo.
"""

    else:

        tone = """
A propensão de conversão ainda não é alta.

Use um tom consultivo, leve e exploratório.

Apresente o produto como uma possibilidade relevante,
sem pressionar o cliente a comprar.

Não use chamadas de ação agressivas.
"""


    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
Você é o concierge de uma experiência
de e-commerce premium chamada NOVA+.

Sua função é explicar de forma natural e elegante
por que um determinado produto está sendo destacado
para este cliente neste momento.

Produto destacado:
"{product}"

Sinais observados:

Categoria principal:
{row["categoria_principal"]}

Eventos nos últimos 30 dias:
{row["qtd_eventos_30d"]}

Categorias exploradas nos últimos 30 dias:
{row["qtd_categorias_interesse_30d"]}

Propensão estimada de conversão:
{probability:.0%}

Histórico recente de navegação:
{json.dumps(events, ensure_ascii=False, indent=2)}

TOM DA COMUNICAÇÃO:

{tone}

REGRAS:

- O produto candidato é fixo.
- Não troque o produto.
- Use somente os fatos observados.
- Não invente informações sobre o produto.
- Não mencione algoritmo.
- Não mencione Machine Learning.
- Não mencione modelo.
- Não mencione Gemini.
- Não mencione JSON.
- Não mencione dados pessoais.
- Não diga que está analisando o cliente.
- Escreva em Português do Brasil.
- Seja natural e sofisticado.
- A resposta deve ter exatamente duas frases.
"""


    # ========================================================
    # CHAMADA AO GEMINI
    # ========================================================

    try:

        client = genai.Client(
            api_key=api_key
        )


        response = client.models.generate_content(

            model=genai_config.get(
                "model",
                "gemini-2.5-flash"
            ),

            contents=prompt,
        )


        generated_text = (
            response.text or ""
        ).strip()


        if not generated_text:

            return None, (
                "O Gemini não retornou conteúdo."
            )


        return generated_text, None


    except Exception as e:

        return None, (
            f"Erro ao gerar a narrativa: {str(e)}"
        )


# ============================================================
# 12. CUSTOMER SIMULATOR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # Identidade
    # --------------------------------------------------------

    st.markdown(
        "## ✦ NOVA+"
    )

    st.caption(
        "HyperCommerce · Customer Simulator"
    )


    st.divider()


    # --------------------------------------------------------
    # Lista de clientes
    # --------------------------------------------------------

    customer_ids = sorted(
        df["cliente_id"].unique()
    )


    # --------------------------------------------------------
    # Seleção do cliente
    # --------------------------------------------------------

    cid = int(
        st.selectbox(
            "Escolha um cliente",
            customer_ids
        )
    )


    # --------------------------------------------------------
    # Recupera dados do cliente
    # --------------------------------------------------------

    customer = get_customer(cid)

    row = customer.iloc[-1]


    # --------------------------------------------------------
    # Calcula propensão
    # --------------------------------------------------------

    probability = customer_score(
        customer
    )


    # --------------------------------------------------------
    # Identifica produto candidato
    # --------------------------------------------------------

    product = candidate_product(
        customer
    )


    st.divider()


    # ========================================================
    # CONTEXTO DETECTADO
    # ========================================================

    st.markdown(
        "**Contexto detectado**"
    )


    st.caption(
        f"◉ Dispositivo · {row['tipo_dispositivo']}"
    )


    st.caption(
        f"◉ Origem · {row['origem']}"
    )


    st.caption(
        f"◉ Categoria · {row['categoria_principal']}"
    )


    st.caption(
        f"◉ Eventos 30d · {row['qtd_eventos_30d']}"
    )


    st.caption(
        f"◉ Categorias · {row['qtd_categorias_interesse_30d']}"
    )


    st.divider()


    # ========================================================
    # MODO PALESTRA
    # ========================================================

    st.caption(
        "MODO PALESTRA"
    )


    st.info(
        "Troque o cliente e observe "
        "como a experiência muda."
    )


# ============================================================
# 13. HERO
# ============================================================

st.markdown(
    """
    <div class="small-label">
        NOVA+ · HYPERCOMMERCE
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="big-title">
        Uma loja que não mostra produtos.
        <br>
        Ela entende o momento.
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="subtitle">
        Machine Learning identifica intenção.
        IA Generativa transforma sinais comportamentais
        em uma experiência que parece feita para você.
    </div>
    """,
    unsafe_allow_html=True,
)


st.write("")


# ============================================================
# 14. ABAS PRINCIPAIS
# ============================================================

tab_experience, tab_rationale = st.tabs(
    [
        "✦  Experiência",
        "◈  Racional",
    ]
)


# ============================================================
# 15. EXPERIÊNCIA DO CLIENTE
# ============================================================

with tab_experience:

    st.markdown(
        "## Para você, agora"
    )


    st.caption(
        "Uma seleção adaptada ao que você está explorando neste momento."
    )


    st.write("")


    # ========================================================
    # PRODUTOS RECOMENDADOS
    # ========================================================

    price = (
        float(
            row["preco_medio_produto"]
        )
        if float(
            row["preco_medio_produto"]
        ) > 0
        else 99.90
    )


    # --------------------------------------------------------
    # Catálogo demonstrativo
    # --------------------------------------------------------

    recommendations = [

        (
            product,
            str(
                row["categoria_principal"]
            ),
            price,
            "SIGNAL MATCH"
        ),

        (
            "VoltCharge 65W",
            "Eletrônicos",
            89.90,
            "MAIS EXPLORADO"
        ),

        (
            "AeroRun Pro",
            "Esportes",
            399.90,
            "TENDÊNCIA"
        ),

        (
            "GlowSkin Kit",
            "Beleza",
            149.90,
            "DESCOBERTA"
        ),

    ]


    # --------------------------------------------------------
    # Cria quatro colunas
    # --------------------------------------------------------

    cols = st.columns(4)


    # --------------------------------------------------------
    # Renderiza produtos
    # --------------------------------------------------------

    for i, (
        name,
        category,
        price_value,
        badge
    ) in enumerate(
        recommendations
    ):

        with cols[i]:

            with st.container(
                border=True
            ):

                st.caption(
                    f"✦ {badge}"
                )


                st.subheader(
                    name
                )


                st.caption(
                    f"{category} · experiência contextual"
                )


                st.markdown(
                    f"### {money(price_value)}"
                )


                if i == 0:

                    st.caption(
                        "Escolha calculada para este momento."
                    )

                else:

                    st.caption(
                        "Explore uma nova possibilidade."
                    )


                st.write("")


                # ------------------------------------------------
                # Botão de interação
                # ------------------------------------------------

                if st.button(
                    "Explorar",
                    key=f"explore_{cid}_{i}"
                ):

                    st.toast(
                        f"{name} entrou na sua jornada ✦"
                    )


    st.write("")
    st.write("")


    # ========================================================
    # 16. HYPERPERSONALIZATION ENGINE
    # ========================================================

    st.markdown(
        "### ✦ Hyperpersonalization Engine"
    )


    st.markdown(
        f"""
        ## Por que **{product}** apareceu para você?
        """
    )


    st.caption(
        "Descubra o racional personalizado por trás desta recomendação."
    )


    st.write("")


    # ========================================================
    # BOTÃO DE ATIVAÇÃO DA IA
    # ========================================================

    ai_clicked = st.button(
        "⚡ Ativar IA Generativa",
        key=f"ai_{cid}",
        use_container_width=True
    )


    # ========================================================
    # GERAÇÃO DA NARRATIVA
    # ========================================================
    #
    # IMPORTANTE:
    #
    # A cada clique:
    #
    #     botão
    #       ↓
    #     live_narrative()
    #       ↓
    #     Gemini
    #       ↓
    #     nova narrativa
    #
    # Não existe narrative_cache.json.
    #
    # ========================================================

    generated = None
    generation_error = None


    if ai_clicked:

        with st.spinner(
            "A NOVA+ está construindo sua experiência..."
        ):

            generated, generation_error = live_narrative(
                row,
                product,
                probability
            )


    # ========================================================
    # TRATAMENTO DA RESPOSTA
    # ========================================================

    if generation_error:

        st.error(
            generation_error
        )


    elif generated:

        st.write("")


        # ----------------------------------------------------
        # Comunicação baseada na propensão
        # ----------------------------------------------------

        if probability >= 0.80:

            st.success(
                "✦ Alta intenção detectada · "
                "Comunicação orientada à conversão"
            )

        else:

            st.info(
                "✦ Sugestão personalizada · "
                "Comunicação consultiva"
            )


        # ----------------------------------------------------
        # Card da narrativa
        # ----------------------------------------------------

        with st.container(
            border=True
        ):

            st.markdown(
                f"### ✦ {product}"
            )


            st.markdown(
                f"> {generated}"
            )


# ============================================================
# 18. RACIONAL
# ============================================================

with tab_rationale:

    st.markdown(
        "### ◈ Racional da experiência"
    )


    st.caption(
        "Aqui está o que acontece por trás da experiência que o cliente vê."
    )


    st.write("")


    # ========================================================
    # 19. SINAIS UTILIZADOS
    # ========================================================

    st.markdown(
        "#### Sinais utilizados pelo sistema"
    )


    c1, c2, c3, c4 = st.columns(4)


    # --------------------------------------------------------
    # INTENÇÃO
    # --------------------------------------------------------

    with c1:

        st.metric(
            "INTENÇÃO",
            f"{probability:.0%}",
            "propensão estimada"
        )


    # --------------------------------------------------------
    # ATIVIDADE
    # --------------------------------------------------------

    with c2:

        st.metric(
            "ATIVIDADE",
            int(
                row[
                    "qtd_eventos_30d"
                ]
            ),
            "eventos · últimos 30d"
        )


    # --------------------------------------------------------
    # EXPLORAÇÃO
    # --------------------------------------------------------

    with c3:

        st.metric(
            "EXPLORAÇÃO",
            int(
                row[
                    "qtd_categorias_interesse_30d"
                ]
            ),
            "categorias observadas"
        )


    # --------------------------------------------------------
    # CANDIDATO
    # --------------------------------------------------------

    with c4:

        st.metric(
            "CANDIDATO",
            product,
            "produto mais aderente"
        )


    st.write("")

    st.divider()

    st.write("")


    # ========================================================
    # 20. JORNADA EM TEMPO REAL
    # ========================================================

    left, right = st.columns(
        [1.05, 1]
    )


    # ========================================================
    # LADO ESQUERDO
    # ========================================================

    with left:

        st.markdown(
            "### A jornada acontecendo em tempo real"
        )


        st.caption(
            "Os últimos sinais observados no comportamento."
        )


        # ----------------------------------------------------
        # Recupera todos os eventos
        # ----------------------------------------------------

        events = parse_log(
            row["log_acesso"]
        )


        if events:

            recent = events[-10:]


            for ev in reversed(recent):

                action = str(
                    ev.get(
                        "acao",
                        "interagiu"
                    )
                )


                prod = str(
                    ev.get(
                        "produto",
                        "conteúdo"
                    )
                )


                category = str(
                    ev.get(
                        "categoria",
                        ""
                    )
                )


                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"**{action.upper()}**"
                    )


                    st.caption(
                        f"{prod} · {category}"
                    )


        else:

            st.info(
                "Nenhum evento disponível."
            )


    # ========================================================
    # 21. PROPENSÃO
    # ========================================================

    with right:

        st.markdown(
            "### O que aconteceu por trás"
        )


        st.caption(
            "A propensão estimada orienta a intensidade da comunicação."
        )


        st.write("")


        # ====================================================
        # GAUGE
        # ====================================================

        fig = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability * 100,


                number={
                    "suffix": "%",
                    "font": {
                        "size": 42
                    }
                },


                title={
                    "text": "Propensão de conversão",
                    "font": {
                        "size": 16
                    }
                },


                gauge={

                    "axis": {
                        "range": [0, 100]
                    },


                    "bar": {
                        "thickness": 0.28
                    },


                    "steps": [

                        {
                            "range": [0, 40]
                        },

                        {
                            "range": [40, 70]
                        },

                        {
                            "range": [70, 100]
                        },

                    ],
                },
            )
        )


        # ====================================================
        # CONFIGURAÇÃO VISUAL DO GAUGE
        # ====================================================

        fig.update_layout(

            height=290,

            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            font={
                "color": "white"
            },

            template="plotly_dark",
        )


        # ----------------------------------------------------
        # Renderiza gráfico
        # ----------------------------------------------------

        st.plotly_chart(

            fig,

            use_container_width=True,

            config={
                "displayModeBar": False
            }
        )


        # ====================================================
        # INTERPRETAÇÃO DA PROPENSÃO
        # ====================================================

        if probability >= 0.80:

            st.success(
                f"""
**ALTA PROPENSÃO · {probability:.0%}**

A IA recebe uma orientação para ser mais
incisiva e orientada à conversão.
"""
            )

        else:

            st.info(
                f"""
**PROPENSÃO MODERADA · {probability:.0%}**

A IA recebe uma orientação mais consultiva,
leve e exploratória.
"""
            )


    st.write("")

    st.divider()

    st.write("")


    # ========================================================
    # 22. PIPELINE
    # ========================================================

    st.markdown(
        "### Observe → Predict → Generate → Act"
    )


    pipeline = st.columns(4)


    # --------------------------------------------------------
    # 01 - OBSERVE
    # --------------------------------------------------------

    with pipeline[0]:

        st.markdown(
            "#### 01 · Observe"
        )


        st.caption(
            "Histórico de navegação, "
            "interações e contexto."
        )


    # --------------------------------------------------------
    # 02 - PREDICT
    # --------------------------------------------------------

    with pipeline[1]:

        st.markdown(
            "#### 02 · Predict"
        )


        st.caption(
            "Machine Learning estima "
            "a propensão de conversão."
        )


    # --------------------------------------------------------
    # 03 - GENERATE
    # --------------------------------------------------------

    with pipeline[2]:

        st.markdown(
            "#### 03 · Generate"
        )


        st.caption(
            "IA Generativa transforma "
            "os sinais em narrativa."
        )


    # --------------------------------------------------------
    # 04 - ACT
    # --------------------------------------------------------

    with pipeline[3]:

        st.markdown(
            "#### 04 · Act"
        )


        st.caption(
            "A experiência apresenta "
            "o próximo melhor conteúdo."
        )


    st.write("")

    st.write("")


    # ========================================================
    # 23. BASTIDORES TÉCNICOS
    # ========================================================

    with st.expander(
        "🧠 Bastidores técnicos da solução"
    ):

        st.markdown(
            """
### Camada 1 · Dados

- histórico de navegação;
- comportamento de sessão;
- categoria principal;
- produtos visualizados;
- eventos dos últimos 30 dias.

### Camada 2 · Machine Learning

- XGBoost;
- pipeline de pré-processamento;
- imputação;
- padronização;
- One-Hot Encoding;
- probabilidade de conversão.

### Camada 3 · IA Generativa

- produto candidato definido pela camada analítica;
- sinais comportamentais enviados para a IA;
- narrativa personalizada;
- Gemini como camada generativa;
- geração sob demanda;
- nenhuma narrativa pré-gerada.

### Camada 4 · Experience

- recomendação;
- explicação;
- jornada;
- interação;
- adaptação da comunicação.

---

### Regra de personalização

**Propensão ≥ 80%**

Comunicação mais incisiva e orientada à conversão.

**Propensão < 80%**

Comunicação consultiva e exploratória.
"""
        )


# ============================================================
# 24. FOOTER
# ============================================================

st.write("")

st.write("")


st.caption(
    "NOVA+ HyperCommerce · Adaptive Experience · Demo para palestra"
)