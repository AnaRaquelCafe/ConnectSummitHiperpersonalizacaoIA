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
# HISTÓRICO COMPLETO DO CLIENTE (TODAS AS SESSÕES)
# ------------------------------------------------------------

def all_customer_events(customer):
    """
    Reúne os eventos de log_acesso de TODAS as sessões
    (linhas) do cliente selecionado, não apenas da última.

    Cada linha do dataset representa uma sessão distinta,
    então olhar só a última sessão perde parte do histórico
    real de navegação do cliente.

    Os eventos são ordenados cronologicamente pelo timestamp,
    para que a recência possa ser usada como critério de peso.
    """

    all_events = []

    for raw in customer["log_acesso"].dropna():

        all_events.extend(
            parse_log(raw)
        )

    all_events.sort(
        key=lambda ev: ev.get("timestamp", "")
    )

    return all_events


# ------------------------------------------------------------
# ATIVIDADE POR PRODUTO (VISUALIZAÇÕES X CLIQUES)
# ------------------------------------------------------------

def build_product_activity(events):
    """
    Agrega os eventos do log de acesso por produto,
    separando visualizações de cliques.

    Eventos cuja ação não seja identificável como
    visualização ou clique são contabilizados como
    visualização (comportamento mais conservador).
    """

    activity = {}

    for ev in events:

        prod = str(
            ev.get(
                "produto",
                "Produto"
            )
        )

        acao = str(
            ev.get(
                "acao",
                ""
            )
        ).lower()

        if "cli" in acao:

            tipo = "Clicado"

        else:

            tipo = "Visualizado"

        activity.setdefault(
            prod,
            {"Visualizado": 0, "Clicado": 0}
        )

        activity[prod][tipo] += 1

    return activity


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
# PESO POR TIPO DE AÇÃO
# ------------------------------------------------------------
#
# Ações com maior intenção de compra pesam mais na escolha
# do produto candidato.
# ------------------------------------------------------------

ACTION_WEIGHTS = {
    "adicionou_carrinho": 4.0,
    "clicou": 2.0,
    "pesquisou": 1.5,
    "visualizou": 1.0,
}


# ------------------------------------------------------------
# PRODUTO CANDIDATO
# ------------------------------------------------------------

def candidate_product(events):
    """
    Identifica o produto mais relevante com base no histórico
    real de navegação do cliente (log_acesso), em vez de uma
    coluna auxiliar desacoplada do comportamento observado.

    Cada evento contribui para a pontuação do produto de
    acordo com:

    - o tipo de ação (carrinho > clique > busca > visualização);
    - a recência (eventos mais recentes pesam mais).

    O produto e a categoria retornados vêm do MESMO evento,
    então o resultado é sempre coerente com o catálogo
    (ex.: nunca retorna um produto de "moda" com categoria
    "eletrônicos").
    """

    if not events:

        return "Produto personalizado", ""


    scores = {}

    last_seen = {}

    n = len(events)


    for i, ev in enumerate(events):

        produto = str(
            ev.get("produto", "")
        ).strip()

        if not produto:

            continue


        acao = str(
            ev.get("acao", "")
        ).lower()

        peso_acao = ACTION_WEIGHTS.get(
            acao,
            1.0
        )

        # Recência: eventos mais recentes (índice maior,
        # já que a lista está ordenada cronologicamente)
        # recebem um peso maior, entre 1x e 2x.
        peso_recencia = 1 + (
            i / max(n - 1, 1)
        )

        scores[produto] = (
            scores.get(produto, 0.0)
            + peso_acao * peso_recencia
        )

        # Guarda o evento mais recente de cada produto, para
        # recuperar a categoria correta depois.
        last_seen[produto] = ev


    if not scores:

        return "Produto personalizado", ""


    produto_escolhido = max(
        scores,
        key=scores.get
    )

    categoria = str(
        last_seen[produto_escolhido].get(
            "categoria",
            ""
        )
    )

    return produto_escolhido, categoria


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
    probability,
    events=None,
    product_categoria=None,
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
    #
    # Se o histórico completo do cliente (todas as sessões)
    # for informado, usamos ele. Caso contrário, caímos de
    # volta para o log da última sessão (compatibilidade).
    # --------------------------------------------------------

    if events is None:

        events = parse_log(
            row["log_acesso"]
        )

    events = events[-12:]


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

Categoria do produto destacado:
{product_categoria or row["categoria_principal"]}

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
    # Histórico completo de navegação (todas as sessões)
    # --------------------------------------------------------

    customer_events = all_customer_events(
        customer
    )


    # --------------------------------------------------------
    # Identifica produto candidato
    # --------------------------------------------------------
    #
    # O produto e a categoria vêm do mesmo evento de log,
    # garantindo coerência entre eles.
    # --------------------------------------------------------

    product, product_categoria = candidate_product(
        customer_events
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
                probability,
                events=customer_events,
                product_categoria=product_categoria,
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
            (
                f"categoria: {product_categoria}"
                if product_categoria
                else "produto mais aderente"
            )
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
        # Recupera os eventos de todas as sessões do cliente
        # ----------------------------------------------------

        events = customer_events


        # ----------------------------------------------------
        # Agrega produtos visualizados e clicados
        # ----------------------------------------------------

        recent_events = events[-30:]

        activity = build_product_activity(
            recent_events
        )


        if activity:

            products = list(
                activity.keys()
            )

            visual_counts = [
                activity[p]["Visualizado"]
                for p in products
            ]

            click_counts = [
                activity[p]["Clicado"]
                for p in products
            ]


            fig_events = go.Figure()


            fig_events.add_trace(
                go.Bar(
                    x=products,
                    y=visual_counts,
                    name="Visualizado",
                    marker_color="#00D2FF",
                )
            )


            fig_events.add_trace(
                go.Bar(
                    x=products,
                    y=click_counts,
                    name="Clicado",
                    marker_color="#A98BFF",
                )
            )


            fig_events.update_layout(

                barmode="group",

                height=320,

                margin=dict(
                    l=10,
                    r=10,
                    t=40,
                    b=10
                ),

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                font={
                    "color": "white"
                },

                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),

                xaxis=dict(
                    tickangle=-30,
                    gridcolor="rgba(255,255,255,.06)",
                ),

                yaxis=dict(
                    gridcolor="rgba(255,255,255,.06)",
                    title="Ocorrências",
                ),

                template="plotly_dark",
            )


            st.plotly_chart(

                fig_events,

                use_container_width=True,

                config={
                    "displayModeBar": False
                }
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
        # COR DINÂMICA DA PROPENSÃO
        # ====================================================
        #
        # Verde  → propensão ≥ 80%
        # Amarelo → propensão < 80%
        # ====================================================

        prop_color = (
            "#22C55E"
            if probability >= 0.80
            else "#F5B942"
        )


        # ====================================================
        # ANEL COM A PROPENSÃO CENTRALIZADA
        # ====================================================

        fig = go.Figure(

            go.Pie(

                values=[
                    probability,
                    1 - probability
                ],

                hole=0.72,

                marker={
                    "colors": [
                        prop_color,
                        "rgba(255,255,255,.08)"
                    ]
                },

                textinfo="none",

                hoverinfo="skip",

                sort=False,

                direction="clockwise",

                rotation=0,
            )
        )


        # ----------------------------------------------------
        # Número da propensão centralizado no gráfico
        # ----------------------------------------------------

        fig.add_annotation(

            text=f"<b>{probability:.0%}</b>",

            x=0.5,
            y=0.54,

            xref="paper",
            yref="paper",

            showarrow=False,

            font={
                "size": 46,
                "color": prop_color,
            },
        )


        fig.add_annotation(

            text="Propensão de conversão",

            x=0.5,
            y=0.40,

            xref="paper",
            yref="paper",

            showarrow=False,

            font={
                "size": 13,
                "color": "#8F98AA",
            },
        )


        # ====================================================
        # CONFIGURAÇÃO VISUAL DO ANEL
        # ====================================================

        fig.update_layout(

            height=290,

            margin=dict(
                l=20,
                r=20,
                t=40,
                b=20
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

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

            st.warning(
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