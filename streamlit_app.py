"""
Checagem Documental Comex — Excel (Produtos) vs Invoice (Fornecedor)
Versão Streamlit — deploy no Streamlit Community Cloud.
"""

import io
import re
import tempfile
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
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
SUFIXOS_CRITICOS = [
    r"/[A-Z]\b", r"-[A-Z]\d?\b", r"-R\d+\b",
    r"-V\d+\b", r"-\bREV\b", r"-\bMK\d+\b",
]


# ============================================================
# EXTRAÇÃO DE PDF
# ============================================================
def _extrair_texto_nativo(file_bytes: bytes) -> tuple[str, int]:
    partes = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        n = len(pdf.pages)
        for p in pdf.pages:
            partes.append(p.extract_text() or "")
    return "\n".join(partes), n


def _tem_cjk(texto: str) -> bool:
    return bool(PADRAO_CJK.search(texto))


def _extrair_texto_ocr(file_bytes: bytes, dpi: int = 300) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    imagens = convert_from_path(tmp_path, dpi=dpi)
    linhas = []
    for img in imagens:
        texto = pytesseract.image_to_string(img, lang="eng")
        linhas.append(texto)
    return "\n".join(linhas)


def extrair_texto(file_bytes: bytes, modo_ocr: str = "auto",
                  dpi: int = 300) -> dict:
    """
    modo_ocr: 'auto' | 'sempre' | 'nunca'
    """
    texto, n = _extrair_texto_nativo(file_bytes)
    media = len(texto.strip()) / n if n else 0

    usar_ocr, motivo = False, "PDF nativo confiável"
    if modo_ocr == "sempre":
        usar_ocr, motivo = True, "OCR forçado pelo usuário"
    elif modo_ocr == "nunca":
        usar_ocr, motivo = False, "Modo vetorial forçado pelo usuário"
    else:  # auto
        if media < LIMITE_CARACTERES_POR_PAGINA:
            usar_ocr = True
            motivo = f"PDF escaneado (média {media:.0f} chars/pág.)"
        elif _tem_cjk(texto):
            usar_ocr = True
            motivo = "PDF com fonte CJK (risco de confundir I↔1, O↔0)"
        else:
            motivo = f"PDF nativo confiável (média {media:.0f} chars/pág.)"

    if usar_ocr:
        texto = _extrair_texto_ocr(file_bytes, dpi=dpi)

    return {"texto": texto, "metodo": "ocr" if usar_ocr else "nativo",
            "paginas": n, "motivo": motivo}


# ============================================================
# COMPARAÇÃO
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


def extrair_codigos(texto: str) -> list[str]:
    if not isinstance(texto, str):
        return []
    achados = PADRAO_CODIGO.findall(texto.upper())
    validos = []
    for m in achados:
        cod = _limpar_codigo(m)
        if (len(cod) >= 6 and re.search(r"[A-Z]", cod)
                and re.search(r"\d", cod)):
            validos.append(cod)
    return list(dict.fromkeys(validos))


def extrair_codigo_principal(texto: str) -> Optional[str]:
    codigos = extrair_codigos(texto)
    return max(codigos, key=len) if codigos else None


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
    n_excel = normalizar(codigo_excel)
    melhor = {
        "codigo_excel": codigo_excel, "codigo_invoice": "",
        "score": 0.0, "status": "NAO_ENCONTRADO",
        "sufixo_critico": False, "diferenca": "",
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
            "codigo_excel": codigo_excel, "codigo_invoice": inv,
            "score": round(score, 2), "status": status,
            "sufixo_critico": sufixo,
            "diferenca": destacar_diferenca(n_excel, n_inv),
        }
    return melhor


# ============================================================
# LEITURA DO EXCEL
# ============================================================
def _achar_linha_cabecalho(file_bytes: bytes, max_linhas: int = 15) -> int:
    df = pd.read_excel(io.BytesIO(file_bytes), header=None,
                       nrows=max_linhas)
    for i in range(len(df)):
        valores = [str(v).strip().lower()
                   for v in df.iloc[i] if pd.notna(v)]
        joined = " ".join(valores)
        if "pedido" in joined and ("partnumber" in joined
                                    or "part number" in joined
                                    or "descri" in joined):
            return i
    return 0


def ler_excel(file_bytes: bytes, coluna: str) -> tuple[list[dict], list[str], list[str]]:
    header_row = _achar_linha_cabecalho(file_bytes)
    df = pd.read_excel(io.BytesIO(file_bytes), header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    alvo = None
    for c in df.columns:
        if c.lower() == coluna.lower():
            alvo = c
            break
    if alvo is None:
        for c in df.columns:
            if coluna.lower() in c.lower():
                alvo = c
                break
    if alvo is None:
        raise ValueError(
            f"Coluna '{coluna}' não encontrada. "
            f"Disponíveis: {list(df.columns)}"
        )

    itens, falhas = [], []
    for _, row in df.iterrows():
        d = row[alvo]
        if pd.isna(d) or not str(d).strip():
            continue
        d = str(d).strip()
        cod = extrair_codigo_principal(d)
        if not cod:
            falhas.append(d)
            continue
        itens.append({"descricao_excel": d, "codigo_excel": cod})
    return itens, falhas, list(df.columns)


# ============================================================
# UI
# ============================================================
st.title("🔍 Checagem Documental Comex")
st.caption(
    "Compare a descrição de produto do setor de Produtos com a Invoice "
    "do fornecedor antes de emitir a DUIMP. Detecta divergências sutis "
    "de sufixo (ex.: `DS-3E0526P-E/M` vs `DS-3E0526P-EI/M`)."
)

with st.sidebar:
    st.header("⚙️ Configurações")
    coluna_nome = st.text_input(
        "Coluna do Excel com a descrição",
        value="Descrição",
        help="Nome exato ou parcial da coluna que contém os códigos de produto."
    )
    modo_ocr = st.radio(
        "Modo de leitura do PDF",
        options=["auto", "sempre", "nunca"],
        index=0,
        format_func=lambda x: {
            "auto": "Automático (recomendado)",
            "sempre": "Forçar OCR sempre",
            "nunca": "Nunca usar OCR (só vetorial)",
        }[x],
        help=(
            "Automático: usa OCR só quando o PDF é escaneado ou tem fonte "
            "CJK (ex.: Hikvision). Forçar: usa OCR sempre. Nunca: "
            "só extração vetorial (mais rápido, mas falha em PDF escaneado)."
        ),
    )
    dpi = st.slider("DPI do OCR", 150, 400, 300, step=50,
                    help="300 é o ideal. Menos que 250 pode borrar códigos.")

st.subheader("1. Suba os arquivos")
col1, col2 = st.columns(2)
with col1:
    up_excel = st.file_uploader(
        "📊 Excel do setor de Produtos (.xlsx)",
        type=["xlsx", "xls"],
    )
with col2:
    up_pdf = st.file_uploader(
        "📄 Invoice do fornecedor (.pdf)",
        type=["pdf"],
    )

if up_excel and up_pdf:
    st.subheader("2. Configuração")
    st.write(
        f"- Excel: `{up_excel.name}` ({up_excel.size/1024:.0f} KB)\n"
        f"- Invoice: `{up_pdf.name}` ({up_pdf.size/1024:.0f} KB)"
    )

    if st.button("🚀 Comparar documentos", type="primary"):
        excel_bytes = up_excel.getvalue()
        pdf_bytes = up_pdf.getvalue()

        with st.spinner("📄 Lendo Excel..."):
            try:
                itens, falhas, colunas = ler_excel(excel_bytes, coluna_nome)
            except Exception as e:
                st.error(f"Erro ao ler Excel: {e}")
                st.stop()

        if not itens:
            st.error(
                f"Nenhum código de produto encontrado na coluna "
                f"'{coluna_nome}'. Colunas disponíveis: {colunas}"
            )
            st.stop()

        st.success(f"✅ {len(itens)} itens com código extraído do Excel")
        if falhas:
            with st.expander(f"⚠️ {len(falhas)} linhas sem código identificável"):
                for f in falhas:
                    st.text(f[:200])

        with st.spinner(f"🔍 Lendo PDF (modo: {modo_ocr})..."):
            try:
                extra = extrair_texto(pdf_bytes, modo_ocr=modo_ocr, dpi=dpi)
            except Exception as e:
                st.error(f"Erro ao processar PDF: {e}")
                st.stop()

        st.info(
            f"**Método:** {extra['metodo'].upper()} — {extra['motivo']}  \n"
            f"**{extra['paginas']} páginas**, "
            f"{len(extra['texto'])} caracteres extraídos"
        )

        invoice_codes = extrair_codigos(extra["texto"])
        if not invoice_codes:
            st.error(
                "Nenhum código de produto encontrado na invoice. "
                "Tente mudar o modo de OCR para 'Forçar OCR sempre'."
            )
            st.stop()

        with st.spinner(f"🧮 Comparando {len(itens)} itens..."):
            resultados = []
            for it in itens:
                r = comparar(it["codigo_excel"], invoice_codes)
                r["descricao_excel"] = it["descricao_excel"]
                resultados.append(r)
            df_res = pd.DataFrame(resultados)

        # ---------- RESULTADOS ----------
        divergentes = df_res[df_res["status"] == "DIVERGENTE"].copy()
        exatos = df_res[df_res["status"] == "EXATO"].copy()
        nao_enc = df_res[df_res["status"] == "NAO_ENCONTRADO"].copy()

        divergentes = divergentes.sort_values(
            by=["sufixo_critico", "score"], ascending=[False, True]
        )

        usados = set(df_res["codigo_invoice"].dropna().tolist())
        sobra = [c for c in invoice_codes if c not in usados]

        st.subheader("3. Resultado")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("✅ Match exato", len(exatos))
        m2.metric("🚨 Divergentes", len(divergentes),
                  delta=f"{len(divergentes)} p/ revisar"
                  if len(divergentes) else None,
                  delta_color="inverse")
        m3.metric("❓ Não encontrados", len(nao_enc))
        m4.metric("➕ Sobra na invoice", len(sobra))

        cols = ["descricao_excel", "codigo_excel", "codigo_invoice",
                "score", "status", "sufixo_critico", "diferenca"]

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🚨 Divergentes", "✅ Match exato",
             "❓ Não encontrados", "➕ Sobra invoice"]
        )
        with tab1:
            if len(divergentes):
                st.dataframe(
                    divergentes[[c for c in cols if c in divergentes.columns]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success("Nenhuma divergência! 🎉")
        with tab2:
            st.dataframe(
                exatos[[c for c in cols if c in exatos.columns]],
                use_container_width=True, hide_index=True,
            )
        with tab3:
            st.dataframe(
                nao_enc[[c for c in cols if c in nao_enc.columns]],
                use_container_width=True, hide_index=True,
            )
        with tab4:
            if sobra:
                st.dataframe(pd.DataFrame({"codigo_invoice_sem_par": sobra}),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Nenhuma sobra na invoice.")

        # ---------- DOWNLOAD ----------
        st.subheader("4. Baixar relatório")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            resumo = pd.DataFrame({
                "Métrica": [
                    "Total de itens no Excel",
                    "Match exato", "Divergentes",
                    "Não encontrados", "Sobra na invoice",
                    "Linhas sem código", "Método PDF", "Motivo",
                ],
                "Valor": [
                    len(df_res), len(exatos), len(divergentes),
                    len(nao_enc), len(sobra), len(falhas),
                    extra["metodo"], extra["motivo"],
                ],
            })
            resumo.to_excel(writer, sheet_name="RESUMO", index=False)
            divergentes.to_excel(writer, sheet_name="DIVERGENTES", index=False)
            exatos.to_excel(writer, sheet_name="MATCH_EXATO", index=False)
            nao_enc.to_excel(writer, sheet_name="NAO_ENCONTRADOS", index=False)
            pd.DataFrame({"sobra": sobra}).to_excel(
                writer, sheet_name="SOBRA_INVOICE", index=False)
            if falhas:
                pd.DataFrame({"sem_codigo": falhas}).to_excel(
                    writer, sheet_name="SEM_CODIGO", index=False)

        st.download_button(
            "📥 Baixar relatório .xlsx",
            data=buffer.getvalue(),
            file_name=f"checagem_{up_pdf.name.replace('.pdf', '')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
else:
    st.info("⬆️ Suba o Excel e o PDF para começar.")