"""
Checagem Documental Comex — Pedidos de Compra (Produtos) vs Invoices (Fornecedor)

Consolida os PartNumbers (coluna D) de pedidos de compra e busca seus pares em
invoices. Rastreia origem de cada item e de cada match.

3 camadas de checagem:
  1. PartNumber exato / fuzzy / sufixo crítico
  2. Checagem semântica da descrição (tokens sensíveis: 5700S vs 5700)
  3. Detecção automática de OCR (escaneado, CJK ou camada fantasma)

Extração de PartNumbers em 4 camadas:
  - Labels explícitos (P/N:, MPN:, Model name:)
  - Padrão com separador (DS-XXXXX, ABC-123-XYZ)
  - Padrão alfanumérico SEM separador (ST4000VX016, BX8071512100)
  - Padrão legado Hikvision

Deploy: Streamlit Community Cloud
"""

import io
import re
import tempfile
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd
import pdfplumber
import pytesseract
import streamlit as st
from pdf2image import convert_from_path
from rapidfuzz import fuzz

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Checagem Documental Comex",
    page_icon="🔍",
    layout="wide",
)

LIMITE_CARACTERES_POR_PAGINA = 50
LIMITE_DIVERGENTE = 70

PADRAO_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
PADRAO_CODIGO = re.compile(r"\b[A-Z]{2,4}(?:\s*[-/]\s*[A-Z0-9()]+)+")

# Labels de PartNumber em invoices internacionais
LABELS_PARTNUMBER = re.compile(
    r"(?:P/?N|PART\s*(?:NO|NUMBER|#)?|MPN|MODEL(?:\s*NAME)?|MODELO|SKU)"
    r"\s*[:#]?\s*([A-Z0-9][A-Z0-9\-/_.]{4,})",
    re.IGNORECASE,
)

# Padrão com separador: DS-XXXXX, ABC-123-XYZ, BX-8071-5121
PADRAO_PARTNUMBER = re.compile(
    r"\b(?=[A-Z0-9\-/_.]*[A-Z])(?=[A-Z0-9\-/_.]*\d)"
    r"[A-Z0-9]{2,}[-/_.][A-Z0-9\-/_.]{3,}\b"
)

# NOVO: códigos alfanuméricos SEM separador (ST4000VX016, BX8071512100)
# Formato: 2-4 letras + 4+ dígitos + até 10 chars alfanuméricos
# Comprimento total: 8-16 caracteres
PADRAO_ALFANUM_SEM_SEP = re.compile(
    r"\b(?=[A-Z0-9]{8,16}\b)"
    r"[A-Z]{2,4}"
    r"[0-9]{4,}"
    r"[A-Z0-9]{0,10}"
    r"\b"
)

# Sufixos que mudam a identidade do produto
SUFIXOS_CRITICOS = [
    r"/[A-Z]\b",
    r"-[A-Z]\d?\b",
    r"-R\d+\b",
    r"-V\d+\b",
    r"-\bREV\b",
    r"-\bMK\d+\b",
]

# Tokens sensíveis: quando divergem entre descrições, geram alerta amarelo
TOKENS_SENSIVEIS = re.compile(
    r"\b(\d{3,4}S?|i[3579]|ryzen\s*[3579]|core\s*i[3579]|"
    r"zen\s*[234]|alder\s*lake|raptor\s*lake)\b",
    re.IGNORECASE,
)

BLACKLIST_TOKENS = {
    "SHIP", "DATE", "TOTAL", "USD", "CNPJ", "ORDER", "PALLET",
    "BOXES", "CARTON", "GROSS", "NET", "WEIGHT", "VOLUME",
    "TERMS", "FREIGHT", "PREPAID", "COLLECT", "PAYMENT",
    "BANK", "SWIFT", "BENEFICIARY", "MANUFACTURER",
    "INVOICE", "COMMERCIAL", "PACKING", "LIST", "CUSTOMER",
    "DELIVERY", "DESCRIPTION", "QUANTITY", "UNIT", "PRICE",
}


# ============================================================
# EXTRAÇÃO DE PDF
# ============================================================
def _extrair_texto_nativo(file_bytes: bytes) -> tuple[str, int]:
    """Extração vetorial via pdfplumber."""
    partes = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        n = len(pdf.pages)
        for p in pdf.pages:
            partes.append(p.extract_text() or "")
    return "\n".join(partes), n


def _tem_cjk(texto: str) -> bool:
    return bool(PADRAO_CJK.search(texto))


def _extraiu_lixo(texto: str) -> bool:
    """
    Detecta camadas de texto fantasma:
      - token dominante (ex.: '1 1 1 1 1...')
      - maioria de tokens curtos
      - maioria de tokens numéricos (numeração sequencial), com salva-vidas
    """
    from collections import Counter
    tokens = texto.split()
    if len(tokens) < 30:
        return False

    # Caso 1: um token domina o texto
    mais_comum, freq = Counter(tokens).most_common(1)[0]
    if freq / len(tokens) > 0.6:
        return True

    # Caso 2: maioria dos tokens é curto (lixo tipo '1 1 1 1')
    curtos = sum(1 for t in tokens if len(t) <= 2)
    if curtos / len(tokens) > 0.85:
        return True

    # Caso 3: maioria puramente numérico (numeração sequencial '2 3 4...')
    # Salva-vidas: se há PartNumbers plausíveis, NÃO é lixo.
    numericos = sum(1 for t in tokens if t.isdigit())
    if numericos / len(tokens) > 0.85:
        if extrair_partnumbers_invoice(texto):
            return False
        return True

    return False


def _extrair_texto_ocr(file_bytes: bytes, dpi: int = 300) -> str:
    """OCR via Tesseract."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    imagens = convert_from_path(tmp_path, dpi=dpi)
    linhas = []
    for img in imagens:
        linhas.append(pytesseract.image_to_string(img, lang="eng"))
    return "\n".join(linhas)


def extrair_texto(file_bytes: bytes, modo_ocr: str = "auto",
                  dpi: int = 300) -> dict:
    """modo_ocr: 'auto' | 'sempre' | 'nunca'."""
    texto, n = _extrair_texto_nativo(file_bytes)
    media = len(texto.strip()) / n if n else 0

    usar_ocr, motivo = False, "PDF nativo confiável"
    if modo_ocr == "sempre":
        usar_ocr, motivo = True, "OCR forçado pelo usuário"
    elif modo_ocr == "nunca":
        usar_ocr, motivo = False, "Modo vetorial forçado"
    else:
        if media < LIMITE_CARACTERES_POR_PAGINA:
            usar_ocr = True
            motivo = f"PDF escaneado (média {media:.0f} chars/pág.)"
        elif _tem_cjk(texto):
            usar_ocr = True
            motivo = "PDF com fonte CJK (risco de confundir I↔1)"
        elif _extraiu_lixo(texto):
            usar_ocr = True
            motivo = "Extração vetorial retornou lixo (camada fantasma)"
        else:
            motivo = f"PDF nativo confiável (média {media:.0f} chars/pág.)"

    if usar_ocr:
        texto = _extrair_texto_ocr(file_bytes, dpi=dpi)

    return {
        "texto": texto,
        "metodo": "ocr" if usar_ocr else "nativo",
        "paginas": n,
        "motivo": motivo,
    }


# ============================================================
# NORMALIZAÇÃO E EXTRAÇÃO DE CÓDIGOS
# ============================================================
def normalizar(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    txt = txt.strip().upper()
    txt = re.sub(r"[\s_\.]+", "-", txt)
    txt = re.sub(r"-{2,}", "-", txt)
    return txt.strip("-")


def _limpar_codigo(m: str) -> str:
    return re.sub(r"\s*([-/])\s*", r"\1", m).strip("-/")


def _parece_partnumber(tok: str) -> bool:
    if len(tok) < 6 or len(tok) > 40:
        return False
    if tok.isdigit():
        return False
    if sum(c.isalpha() for c in tok) < 1:
        return False
    if sum(c.isdigit() for c in tok) < 2:
        return False
    if tok.upper() in BLACKLIST_TOKENS:
        return False
    return True


def extrair_partnumbers_invoice(texto: str) -> list[str]:
    """Extração em camadas de PartNumbers da invoice."""
    if not isinstance(texto, str):
        return []
    texto_up = texto.upper()
    achados = []

    # 1) Labels explícitos (P/N:, MPN:, Model name:, SKU:)
    for m in LABELS_PARTNUMBER.finditer(texto_up):
        tok = _limpar_codigo(m.group(1))
        if _parece_partnumber(tok):
            achados.append(tok)

    # 2) Padrão com separador (DS-XXXXX, ABC-123-XYZ)
    for m in PADRAO_PARTNUMBER.findall(texto_up):
        tok = _limpar_codigo(m)
        if _parece_partnumber(tok):
            achados.append(tok)

    # 3) Alfanumérico SEM separador (ST4000VX016, BX8071512100)
    for m in PADRAO_ALFANUM_SEM_SEP.findall(texto_up):
        tok = m.strip()
        if _parece_partnumber(tok):
            achados.append(tok)

    # 4) Padrão legado (Hikvision-like)
    for m in PADRAO_CODIGO.findall(texto_up):
        tok = _limpar_codigo(m)
        if _parece_partnumber(tok):
            achados.append(tok)

    return list(dict.fromkeys(achados))


# ============================================================
# COMPARAÇÃO
# ============================================================
def _remover_sufixo_critico(txt: str) -> str:
    for padrao in SUFIXOS_CRITICOS:
        novo = re.sub(padrao + r"$", "", txt)
        if novo != txt:
            return novo
    return txt


def diferenca_e_sufixo_critico(a: str, b: str) -> bool:
    na, nb = normalizar(a), normalizar(b)
    if na == nb:
        return False
    if len(na) > len(nb) and _remover_sufixo_critico(na) == nb:
        return True
    if len(nb) > len(na) and _remover_sufixo_critico(nb) == na:
        return True
    return False


def destacar_diferenca(a: str, b: str) -> str:
    sm = SequenceMatcher(None, a, b)
    partes = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            partes.append(a[i1:i2])
        else:
            if i2 > i1:
                partes.append(f"[-{a[i1:i2]}-]")
            if j2 > j1:
                partes.append(f"[+{b[j1:j2]}+]")
    return " ".join(partes)


def comparar(codigo_excel: str, codigos_invoice: list[str]) -> dict:
    """Compara um PartNumber do Excel contra todos das invoices."""
    n_excel = normalizar(codigo_excel)
    melhor = {
        "codigo_excel": codigo_excel,
        "codigo_invoice": "",
        "score": 0.0,
        "status": "NAO_ENCONTRADO",
        "sufixo_critico": False,
        "diferenca": "",
    }
    for inv in codigos_invoice:
        n_inv = normalizar(inv)
        score = fuzz.ratio(n_excel, n_inv)
        if score < melhor["score"]:
            continue
        sufixo = diferenca_e_sufixo_critico(codigo_excel, inv)
        if n_excel == n_inv:
            status = "EXATO"
        elif score >= LIMITE_DIVERGENTE:
            status = "DIVERGENTE"
        else:
            status = "NAO_ENCONTRADO"
        if sufixo and status != "EXATO":
            status = "DIVERGENTE"
        melhor = {
            "codigo_excel": codigo_excel,
            "codigo_invoice": inv,
            "score": round(score, 2),
            "status": status,
            "sufixo_critico": sufixo,
            "diferenca": destacar_diferenca(n_excel, n_inv),
        }
    return melhor


# ============================================================
# CHECAGEM SEMÂNTICA
# ============================================================
def checar_semantica(desc_excel: str, desc_invoice: str) -> dict:
    """Compara tokens sensíveis (5700S vs 5700, i5 vs i3, etc.)."""
    if not desc_excel or not desc_invoice:
        return {"alerta": False, "so_no_excel": set(), "so_na_invoice": set()}

    tx = {m.group(0).upper() for m in TOKENS_SENSIVEIS.finditer(desc_excel)}
    ti = {m.group(0).upper() for m in TOKENS_SENSIVEIS.finditer(desc_invoice)}

    def _norm(s):
        return {t.replace(" ", "") for t in s}

    so_excel = _norm(tx) - _norm(ti)
    so_inv = _norm(ti) - _norm(tx)
    return {
        "alerta": bool(so_excel or so_inv),
        "so_no_excel": so_excel,
        "so_na_invoice": so_inv,
    }


# ============================================================
# LEITURA DO EXCEL
# ============================================================
def _achar_linha_cabecalho(file_bytes: bytes, max_linhas: int = 15) -> int:
    df = pd.read_excel(io.BytesIO(file_bytes), header=None, nrows=max_linhas)
    for i in range(len(df)):
        valores = [str(v).strip().lower() for v in df.iloc[i] if pd.notna(v)]
        joined = " ".join(valores)
        if "pedido" in joined and ("partnumber" in joined
                                    or "part number" in joined
                                    or "descri" in joined):
            return i
    return 0


def ler_excel(file_bytes: bytes, nome_arquivo: str) -> tuple[list[dict], list[str]]:
    """Lê um Excel e retorna itens já com origem + lista de falhas."""
    header_row = _achar_linha_cabecalho(file_bytes)
    df = pd.read_excel(io.BytesIO(file_bytes), header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    def _achar(*nomes: str) -> Optional[str]:
        for alvo in nomes:
            for c in df.columns:
                if c.lower().strip() == alvo.lower():
                    return c
        for alvo in nomes:
            for c in df.columns:
                if alvo.lower() in c.lower():
                    return c
        return None

    col_pedido = _achar("Pedido")
    col_pn = _achar("PartNumber", "Part Number", "Partnumber", "PN", "P/N")
    col_desc = _achar("Descrição", "Descricao")
    col_duimp = _achar("Descrição DUIMP", "Descricao DUIMP")

    if col_pn is None:
        raise ValueError(
            f"Coluna 'PartNumber' não encontrada em '{nome_arquivo}'. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    itens, falhas = [], []
    for _, row in df.iterrows():
        pn = row[col_pn]
        if pd.isna(pn) or not str(pn).strip():
            continue
        pn = str(pn).strip()
        itens.append({
            "partnumber": pn,
            "pedido": str(row[col_pedido]).strip()
            if col_pedido and pd.notna(row[col_pedido]) else "",
            "descricao": str(row[col_desc]).strip()
            if col_desc and pd.notna(row[col_desc]) else "",
            "descricao_duimp": str(row[col_duimp]).strip()
            if col_duimp and pd.notna(row[col_duimp]) else "",
            "arquivo_excel": nome_arquivo,
        })
    return itens, falhas


# ============================================================
# UI
# ============================================================
st.title("🔍 Checagem Documental Comex")
st.caption(
    "Compara os **PartNumbers (coluna D)** de Pedidos de Compra "
    "contra invoices do fornecedor. Detecta divergências de sufixo "
    "(ex.: `DS-3E0526P-E/M` vs `DS-3E0526P-EI/M`) e alertas semânticos "
    "(ex.: `5700S` vs `5700`)."
)

with st.sidebar:
    st.header("⚙️ Configurações")
    modo_ocr = st.radio(
        "Modo de leitura dos PDFs",
        options=["auto", "sempre", "nunca"],
        index=0,
        format_func=lambda x: {
            "auto": "Automático (recomendado)",
            "sempre": "Forçar OCR sempre",
            "nunca": "Nunca usar OCR (só vetorial)",
        }[x],
        help=(
            "Automático: usa OCR só quando o PDF é escaneado, tem fonte "
            "CJK (Hikvision) ou retorna camada fantasma. Forçar: usa OCR "
            "sempre (mais lento). Nunca: só vetorial."
        ),
    )
    dpi = st.slider("DPI do OCR", 150, 400, 300, step=50,
                    help="300 é ideal. Menos que 250 pode borrar códigos.")

    st.divider()
    st.caption(
        "**Dica:** o expander `🔎 PartNumbers detectados` mostra tudo "
        "que a ferramenta extraiu de cada invoice. Use-o para diagnosticar "
        "códigos não encontrados."
    )

st.subheader("1. Suba os arquivos")
col1, col2 = st.columns(2)
with col1:
    up_excels = st.file_uploader(
        "📊 Excels de pedidos (.xlsx) — pode subir vários",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )
with col2:
    up_pdfs = st.file_uploader(
        "📄 Invoices do fornecedor (.pdf) — pode subir várias",
        type=["pdf"],
        accept_multiple_files=True,
    )

if up_excels and up_pdfs:
    st.subheader("2. Arquivos carregados")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"📊 **{len(up_excels)} Excel(s):**")
        for e in up_excels:
            st.write(f"  - `{e.name}` ({e.size/1024:.0f} KB)")
    with col_b:
        st.write(f"📄 **{len(up_pdfs)} invoice(s):**")
        for p in up_pdfs:
            st.write(f"  - `{p.name}` ({p.size/1024:.0f} KB)")

    if st.button("🚀 Comparar documentos", type="primary"):
        # ---- EXCELS ----
        itens_consolidados: list[dict] = []
        falhas_excel: list[dict] = []
        erros_excel: list[dict] = []

        progresso_e = st.progress(0.0, text="Lendo Excels...")
        for i, excel in enumerate(up_excels):
            try:
                itens, falhas = ler_excel(excel.getvalue(), excel.name)
                itens_consolidados.extend(itens)
                for f in falhas:
                    falhas_excel.append(
                        {"arquivo": excel.name, "descricao": f}
                    )
            except Exception as e:
                erros_excel.append({"arquivo": excel.name, "erro": str(e)})
            progresso_e.progress((i + 1) / len(up_excels),
                                 text=f"Lendo Excels... "
                                      f"({i+1}/{len(up_excels)})")
        progresso_e.empty()

        if erros_excel:
            st.error("❌ Alguns Excels falharam:")
            st.dataframe(pd.DataFrame(erros_excel),
                         use_container_width=True, hide_index=True)

        if not itens_consolidados:
            st.error(
                "Nenhum PartNumber encontrado em nenhum dos Excels."
            )
            st.stop()

        st.success(
            f"✅ {len(itens_consolidados)} PartNumbers consolidados de "
            f"{len(up_excels)} Excel(s)"
        )

        # ---- PDFS ----
        mapa_pns: dict[str, list[str]] = {}
        textos_consolidados = []
        logs_pdf = []
        erros_pdf = []

        progresso_p = st.progress(0.0, text="Processando invoices...")
        for i, pdf in enumerate(up_pdfs):
            nome = pdf.name
            try:
                extra = extrair_texto(pdf.getvalue(),
                                      modo_ocr=modo_ocr, dpi=dpi)
                pns = extrair_partnumbers_invoice(extra["texto"])
                for pn in pns:
                    mapa_pns.setdefault(pn, [])
                    if nome not in mapa_pns[pn]:
                        mapa_pns[pn].append(nome)
                textos_consolidados.append(extra["texto"])
                logs_pdf.append({
                    "arquivo": nome,
                    "metodo": extra["metodo"],
                    "motivo": extra["motivo"],
                    "paginas": extra["paginas"],
                    "partnumbers": len(pns),
                })
            except Exception as e:
                erros_pdf.append({"arquivo": nome, "erro": str(e)})
            progresso_p.progress((i + 1) / len(up_pdfs),
                                 text=f"Processando invoices... "
                                      f"({i+1}/{len(up_pdfs)})")
        progresso_p.empty()

        if erros_pdf:
            st.warning("⚠️ Alguns PDFs falharam:")
            st.dataframe(pd.DataFrame(erros_pdf),
                         use_container_width=True, hide_index=True)

        pns_invoice = list(mapa_pns.keys())
        if not pns_invoice:
            st.error(
                "Nenhum PartNumber encontrado nas invoices. "
                "Tente 'Forçar OCR sempre' ou verifique se as invoices "
                "têm labels como 'P/N:', 'MPN:', 'Model'."
            )
            st.stop()

        st.success(
            f"✅ {len(pns_invoice)} PartNumbers únicos detectados em "
            f"{len(logs_pdf)} invoice(s)"
        )

        with st.expander("📋 Log de processamento dos PDFs"):
            st.dataframe(pd.DataFrame(logs_pdf),
                         use_container_width=True, hide_index=True)

        with st.expander(f"🔎 {len(pns_invoice)} PartNumbers detectados"):
            st.write(pns_invoice)

        if falhas_excel:
            with st.expander(
                f"⚠️ {len(falhas_excel)} linhas de Excel sem PartNumber"
            ):
                st.dataframe(pd.DataFrame(falhas_excel),
                             use_container_width=True, hide_index=True)

        # ---- COMPARAÇÃO ----
        texto_consolidado = "\n".join(textos_consolidados)

        with st.spinner(f"🧮 Comparando {len(itens_consolidados)} itens..."):
            resultados = []
            for it in itens_consolidados:
                r = comparar(it["partnumber"], pns_invoice)
                r["partnumber_excel"] = it["partnumber"]
                r["descricao_excel"] = it["descricao"]
                r["pedido"] = it["pedido"]
                r["excel_origem"] = it["arquivo_excel"]

                if r["codigo_invoice"] in mapa_pns:
                    r["invoices_origem"] = ", ".join(
                        mapa_pns[r["codigo_invoice"]]
                    )
                else:
                    r["invoices_origem"] = ""

                sem = checar_semantica(it["descricao"], texto_consolidado)
                r["alerta_semantico"] = sem["alerta"]
                r["tokens_divergentes"] = (
                    ", ".join(sorted(sem["so_no_excel"]
                                      | sem["so_na_invoice"]))
                    if sem["alerta"] else ""
                )
                resultados.append(r)
            df_res = pd.DataFrame(resultados)

        divergentes = df_res[df_res["status"] == "DIVERGENTE"].copy()
        exatos = df_res[df_res["status"] == "EXATO"].copy()
        nao_enc = df_res[df_res["status"] == "NAO_ENCONTRADO"].copy()
        alerta_sem = df_res[df_res["alerta_semantico"]].copy()

        divergentes = divergentes.sort_values(
            by=["sufixo_critico", "score"], ascending=[False, True]
        )

        usados = set(df_res["codigo_invoice"].dropna().tolist())
        sobra = [c for c in pns_invoice if c not in usados]

        # ---- RESULTADO ----
        st.subheader("3. Resultado")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ Match exato", len(exatos))
        m2.metric(
            "🚨 Divergentes", len(divergentes),
            delta=f"{len(divergentes)} p/ revisar"
            if len(divergentes) else None,
            delta_color="inverse",
        )
        m3.metric("❓ Não encontrados", len(nao_enc))
        m4.metric("🟡 Alerta semântico", len(alerta_sem))

        cols = ["pedido", "excel_origem", "partnumber_excel",
                "descricao_excel", "codigo_invoice", "invoices_origem",
                "score", "status", "sufixo_critico", "diferenca",
                "tokens_divergentes"]

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🚨 Divergentes", "🟡 Alertas semânticos", "✅ Match exato",
             "❓ Não encontrados", "➕ Sobra nas invoices"]
        )
        with tab1:
            if len(divergentes):
                st.dataframe(
                    divergentes[[c for c in cols if c in divergentes.columns]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success("Nenhuma divergência de PartNumber! 🎉")
        with tab2:
            if len(alerta_sem):
                st.dataframe(
                    alerta_sem[[c for c in cols if c in alerta_sem.columns]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("Sem alertas semânticos.")
        with tab3:
            st.dataframe(
                exatos[[c for c in cols if c in exatos.columns]],
                use_container_width=True, hide_index=True,
            )
        with tab4:
            st.dataframe(
                nao_enc[[c for c in cols if c in nao_enc.columns]],
                use_container_width=True, hide_index=True,
            )
        with tab5:
            if sobra:
                sobra_df = pd.DataFrame({
                    "partnumber_sobra": sobra,
                    "invoices": [", ".join(mapa_pns[p]) for p in sobra],
                })
                st.dataframe(sobra_df, use_container_width=True,
                             hide_index=True)
            else:
                st.info("Nenhuma sobra nas invoices.")

        # ---- DOWNLOAD ----
        st.subheader("4. Baixar relatório")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            resumo = pd.DataFrame({
                "Métrica": [
                    "Total de Excels processados",
                    "Total de invoices processadas",
                    "Total de PartNumbers nos Excels",
                    "PartNumbers únicos nas invoices",
                    "Match exato",
                    "Divergentes",
                    "Não encontrados",
                    "Alerta semântico",
                    "Sobra nas invoices",
                ],
                "Valor": [
                    len(up_excels), len(logs_pdf),
                    len(df_res), len(pns_invoice),
                    len(exatos), len(divergentes), len(nao_enc),
                    len(alerta_sem), len(sobra),
                ],
            })
            resumo.to_excel(writer, sheet_name="RESUMO", index=False)
            divergentes.to_excel(writer, sheet_name="DIVERGENTES",
                                 index=False)
            alerta_sem.to_excel(writer, sheet_name="ALERTA_SEMANTICO",
                                index=False)
            exatos.to_excel(writer, sheet_name="MATCH_EXATO", index=False)
            nao_enc.to_excel(writer, sheet_name="NAO_ENCONTRADOS",
                             index=False)
            pd.DataFrame({
                "partnumber_sobra": sobra,
                "invoices": [", ".join(mapa_pns[p]) for p in sobra],
            }).to_excel(writer, sheet_name="SOBRA_INVOICE", index=False)
            pd.DataFrame(logs_pdf).to_excel(writer,
                                            sheet_name="LOG_PDFS",
                                            index=False)
            if falhas_excel:
                pd.DataFrame(falhas_excel).to_excel(
                    writer, sheet_name="LINHAS_SEM_PN", index=False)

        nome_base = (
            f"{len(up_excels)}excels_{len(up_pdfs)}invoices"
            if len(up_excels) > 1 or len(up_pdfs) > 1
            else up_excels[0].name.replace(".xlsx", "")
        )
        st.download_button(
            "📥 Baixar relatório .xlsx",
            data=buffer.getvalue(),
            file_name=f"checagem_{nome_base}.xlsx",
            mime=("application/vnd.openxmlformats-officedocument"
                  ".spreadsheetml.sheet"),
            type="primary",
        )
else:
    st.info(
        "⬆️ Suba ao menos um Excel de pedido e uma invoice PDF "
        "para começar."
    )
