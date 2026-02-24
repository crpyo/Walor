"""
╔══════════════════════════════════════════════════════════════════════╗
║         InfraInvestor Bot v3 — by Matheus Facco                      ║
║         Machine Learning + Backtesting + Comandos Telegram           ║
║         Gestão de Risco + Correlação + Notícias + Dashboard          ║
╚══════════════════════════════════════════════════════════════════════╝

pip install -r requirements.txt
python bot.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json, time, logging, requests, os, math, threading
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

# ML
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_DISPONIVEL = True
except ImportError:
    ML_DISPONIVEL = False
    logging.warning("scikit-learn não instalado — ML desativado. pip install scikit-learn joblib")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════
CONFIG = {
    "TELEGRAM_TOKEN":    os.getenv("TELEGRAM_TOKEN", "SEU_TOKEN_AQUI"),
    "TELEGRAM_CHAT_ID":  os.getenv("TELEGRAM_CHAT_ID", "SEU_CHAT_ID_AQUI"),
    "ALPACA_API_KEY":    os.getenv("ALPACA_API_KEY", ""),
    "ALPACA_API_SECRET": os.getenv("ALPACA_API_SECRET", ""),

    # ✅ SIMULAÇÃO
    "ALPACA_BASE_URL":   "https://paper-api.alpaca.markets",
    # 🚀 REAL — só mudar isso
    # "ALPACA_BASE_URL": "https://api.alpaca.markets",

    "INTERVALO_ANALISE": 3600,       # 1 hora
    "META_RENDA_MENSAL": 500.0,      # $500/mês — meta de renda passiva
    "DASHBOARD_PORT":    8080,       # porta do dashboard web
    "BEAR_MARKET_LIMITE":-0.10,      # queda > 10% no S&P = bear market
    "MAX_DRAWDOWN":      -0.15,      # para operações se carteira cair 15%
    "PAUSADO":           False,      # controlado por /pausar e /retomar
}

ATIVOS = {
    "NEE":  {"nome": "NextEra Energy",  "setor": "Energia Renovavel",   "frequencia": "trimestral", "prioridade": 1},
    "DUK":  {"nome": "Duke Energy",     "setor": "Energia Distribuicao","frequencia": "trimestral", "prioridade": 1},
    "PCG":  {"nome": "PGE Corp",        "setor": "Energia California",  "frequencia": "trimestral", "prioridade": 2},
    "AMT":  {"nome": "American Tower",  "setor": "Torres Telecom",      "frequencia": "trimestral", "prioridade": 1},
    "EQIX": {"nome": "Equinix",         "setor": "Data Centers",        "frequencia": "trimestral", "prioridade": 1},
    "O":    {"nome": "Realty Income",   "setor": "REIT Mensal",         "frequencia": "mensal",     "prioridade": 1},
    "STAG": {"nome": "STAG Industrial", "setor": "REIT Galpoes",        "frequencia": "mensal",     "prioridade": 2},
}

CRITERIOS = {
    "rsi_sobrevenda":    35,
    "rsi_sobrecompra":   72,
    "queda_media_3m":   -0.05,
    "dy_minimo":         0.015,
    "pe_maximo":         65,
    "meta_lucro_venda":  0.22,
    "max_pct_carteira":  0.30,
    "score_minimo":      55,
    "score_minimo_ml":   0.60,   # probabilidade mínima do ML para comprar
}

# Palavras críticas em notícias que pausam operações do ativo
PALAVRAS_CRITICAS = [
    "dividend cut", "dividend suspended", "SEC investigation",
    "bankruptcy", "fraud", "earnings miss", "CEO resign",
    "regulatory fine", "lawsuit", "downgrade",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("infra_investor.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Estado global compartilhado
ESTADO = {
    "ultimo_ciclo":    None,
    "analises":        [],
    "metas":           {},
    "renda_atual":     0.0,
    "patrimonio":      0.0,
    "bear_market":     False,
    "ativos_pausados": set(),   # ativos com notícia crítica
}


# ═══════════════════════════════════════════════════════════════════════
# ALPACA CLIENT
# ═══════════════════════════════════════════════════════════════════════
class AlpacaClient:
    def __init__(self):
        self.base    = CONFIG["ALPACA_BASE_URL"]
        self.headers = {
            "APCA-API-KEY-ID":     CONFIG["ALPACA_API_KEY"],
            "APCA-API-SECRET-KEY": CONFIG["ALPACA_API_SECRET"],
            "Content-Type":        "application/json",
        }

    def _get(self, ep):
        try:
            r = requests.get(f"{self.base}{ep}", headers=self.headers, timeout=10)
            r.raise_for_status(); return r.json()
        except Exception as e:
            log.error(f"GET {ep}: {e}"); return {}

    def _post(self, ep, payload):
        try:
            r = requests.post(f"{self.base}{ep}", headers=self.headers, json=payload, timeout=10)
            return r.json()
        except Exception as e:
            log.error(f"POST {ep}: {e}"); return {}

    def conta(self):      return self._get("/v2/account")
    def saldo(self):      return float(self.conta().get("buying_power", 0))
    def patrimonio(self): return float(self.conta().get("portfolio_value", 0))

    def posicoes(self):
        resp = self._get("/v2/positions")
        return {p["symbol"]: p for p in resp} if isinstance(resp, list) else {}

    def comprar(self, ticker, qtd, motivo=""):
        if qtd <= 0: return False
        resp = self._post("/v2/orders", {
            "symbol": ticker, "qty": str(qtd),
            "side": "buy", "type": "market", "time_in_force": "gtc",
        })
        ok = bool(resp.get("id"))
        log.info(f"{'✅ COMPRA OK' if ok else '❌ ERRO'} | {ticker} x{qtd} | {motivo}")
        return ok

    def vender(self, ticker, qtd, motivo=""):
        if qtd <= 0: return False
        resp = self._post("/v2/orders", {
            "symbol": ticker, "qty": str(qtd),
            "side": "sell", "type": "market", "time_in_force": "gtc",
        })
        ok = bool(resp.get("id"))
        log.info(f"{'💰 VENDA OK' if ok else '❌ ERRO'} | {ticker} x{qtd} | {motivo}")
        return ok


# ═══════════════════════════════════════════════════════════════════════
# CARTEIRA LOCAL
# ═══════════════════════════════════════════════════════════════════════
class Carteira:
    def __init__(self, arquivo="carteira.json"):
        self.arquivo = arquivo
        if os.path.exists(arquivo):
            with open(arquivo) as f: self.dados = json.load(f)
        else:
            self.dados = {
                "posicoes":          {},
                "historico":         [],
                "metas":             {},
                "snapshots":         [],
                "operacoes_ml":      [],   # dados para treinar ML
                "patrimonio_inicial": 0.0,
            }

    def salvar(self):
        with open(self.arquivo, "w") as f:
            json.dump(self.dados, f, indent=2, default=str)

    def registrar_compra(self, ticker, preco, qtd, motivo, features_ml=None):
        pos = self.dados["posicoes"].setdefault(
            ticker, {"quantidade": 0.0, "preco_medio": 0.0, "total_gasto": 0.0}
        )
        custo = preco * qtd
        pos["preco_medio"]  = (pos["quantidade"] * pos["preco_medio"] + custo) / (pos["quantidade"] + qtd)
        pos["quantidade"]  += qtd
        pos["total_gasto"] += custo

        entrada = {
            "id":          f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "data":        datetime.now().isoformat(),
            "tipo":        "COMPRA",
            "ticker":      ticker,
            "preco":       preco,
            "qtd":         qtd,
            "motivo":      motivo,
            "features_ml": features_ml or {},
            "resultado":   None,   # preenchido depois pela ML
        }
        self.dados["historico"].append(entrada)
        if features_ml:
            self.dados["operacoes_ml"].append(entrada)
        self.salvar()

    def registrar_venda(self, ticker, preco, qtd, motivo):
        pos = self.dados["posicoes"].get(ticker)
        if not pos: return 0
        lucro = (preco - pos["preco_medio"]) * qtd
        pos["quantidade"] -= qtd
        if pos["quantidade"] <= 0.001: del self.dados["posicoes"][ticker]

        # Marca resultado nas operações ML abertas desse ticker
        for op in self.dados.get("operacoes_ml", []):
            if op["ticker"] == ticker and op["resultado"] is None:
                op["resultado"] = 1 if lucro > 0 else 0   # 1=lucro, 0=prejuízo

        self.dados["historico"].append({
            "data": datetime.now().isoformat(), "tipo": "VENDA",
            "ticker": ticker, "preco": preco, "qtd": qtd,
            "lucro": round(lucro, 2), "motivo": motivo,
        })
        self.salvar()
        return lucro

    def salvar_snapshot(self, patrimonio, renda):
        if self.dados["patrimonio_inicial"] == 0:
            self.dados["patrimonio_inicial"] = patrimonio
        self.dados["snapshots"].append({
            "data": datetime.now().isoformat(),
            "patrimonio": round(patrimonio, 2),
            "renda_mensal": round(renda, 2),
        })
        self.dados["snapshots"] = self.dados["snapshots"][-180:]
        self.salvar()

    def atualizar_metas(self, metas):
        self.dados["metas"] = metas; self.salvar()

    def preco_medio(self, t):  return self.dados["posicoes"].get(t, {}).get("preco_medio", 0.0)
    def quantidade(self, t):   return self.dados["posicoes"].get(t, {}).get("quantidade", 0.0)
    def meta(self, t):         return self.dados["metas"].get(t, {})

    def drawdown_atual(self, patrimonio_atual):
        pi = self.dados.get("patrimonio_inicial", 0)
        if pi <= 0: return 0
        return (patrimonio_atual - pi) / pi


# ═══════════════════════════════════════════════════════════════════════
# MACHINE LEARNING — aprende com cada operação
# ═══════════════════════════════════════════════════════════════════════
class ModeloML:
    def __init__(self, arquivo="modelo_ml.pkl"):
        self.arquivo  = arquivo
        self.scaler   = StandardScaler() if ML_DISPONIVEL else None
        self.modelo   = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
        ) if ML_DISPONIVEL else None
        self.treinado = False
        self._carregar()

    def _carregar(self):
        if not ML_DISPONIVEL: return
        if os.path.exists(self.arquivo):
            try:
                dados = joblib.load(self.arquivo)
                self.modelo   = dados["modelo"]
                self.scaler   = dados["scaler"]
                self.treinado = True
                log.info("✅ Modelo ML carregado")
            except Exception as e:
                log.error(f"Erro ao carregar ML: {e}")

    def salvar(self):
        if not ML_DISPONIVEL or not self.treinado: return
        joblib.dump({"modelo": self.modelo, "scaler": self.scaler}, self.arquivo)

    def extrair_features(self, a):
        """Extrai vetor de features de uma análise de ativo"""
        freq_num = {"mensal": 3, "trimestral": 2, "semestral": 1, "anual": 0}.get(a["frequencia"], 1)
        pos_52 = 0.5
        if a.get("52w_high", 0) > a.get("52w_low", 0):
            pos_52 = (a["preco"] - a["52w_low"]) / (a["52w_high"] - a["52w_low"])

        return [
            a["rsi"],
            a["dy"],
            a["pe"] if a["pe"] < 999 else 100,
            a["var_media"],
            a.get("upside", 0),
            freq_num,
            pos_52,
            ATIVOS.get(a["ticker"], {}).get("prioridade", 3),
            a.get("tendencia_num", 0),   # 1=alta, -1=baixa
        ]

    def treinar(self, operacoes_ml):
        """Treina com histórico de operações que já têm resultado conhecido"""
        if not ML_DISPONIVEL: return False
        dados_treino = [op for op in operacoes_ml if op.get("resultado") is not None and op.get("features_ml")]
        if len(dados_treino) < 10:
            log.info(f"ML: {len(dados_treino)} amostras — precisa de 10 para treinar")
            return False

        X = [list(op["features_ml"].values()) for op in dados_treino]
        y = [op["resultado"] for op in dados_treino]

        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)
        self.treinado = True
        self.salvar()

        acuracia = self.modelo.score(X_scaled, y)
        log.info(f"✅ ML treinado | {len(dados_treino)} amostras | Acurácia: {acuracia:.1%}")
        return True

    def prever(self, features):
        """Retorna probabilidade de sucesso (0 a 1)"""
        if not ML_DISPONIVEL or not self.treinado: return None
        try:
            X = self.scaler.transform([features])
            prob = self.modelo.predict_proba(X)[0][1]
            return round(prob, 3)
        except Exception as e:
            log.error(f"ML predict: {e}"); return None

    def importancia_features(self):
        if not self.treinado: return {}
        nomes = ["RSI","DY","PE","Var3m","Upside","Freq","Pos52s","Prioridade","Tendencia"]
        return dict(zip(nomes, self.modelo.feature_importances_))


# ═══════════════════════════════════════════════════════════════════════
# ANÁLISE DE NOTÍCIAS
# ═══════════════════════════════════════════════════════════════════════
def verificar_noticias(ticker, nome):
    """
    Busca notícias recentes e detecta palavras críticas.
    Usa RSS do Yahoo Finance — gratuito, sem API key.
    """
    try:
        url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200: return False, []

        texto    = resp.text.lower()
        alertas  = [p for p in PALAVRAS_CRITICAS if p in texto]

        if alertas:
            log.warning(f"⚠️ Notícia crítica {ticker}: {alertas}")
            return True, alertas
        return False, []
    except Exception as e:
        log.debug(f"Notícias {ticker}: {e}"); return False, []


# ═══════════════════════════════════════════════════════════════════════
# DETECÇÃO DE BEAR MARKET
# ═══════════════════════════════════════════════════════════════════════
def detectar_bear_market():
    """Verifica se S&P 500 está em queda significativa"""
    try:
        spy  = yf.Ticker("SPY")
        hist = spy.history(period="3mo")
        if hist.empty: return False
        preco_atual  = hist["Close"].iloc[-1]
        preco_3m_atras = hist["Close"].iloc[0]
        queda = (preco_atual - preco_3m_atras) / preco_3m_atras
        em_bear = queda < CONFIG["BEAR_MARKET_LIMITE"]
        if em_bear:
            log.warning(f"🐻 Bear market detectado | S&P caiu {queda:.1%} em 3 meses")
        return em_bear
    except Exception as e:
        log.error(f"Bear market check: {e}"); return False


# ═══════════════════════════════════════════════════════════════════════
# ANÁLISE DE CORRELAÇÃO
# ═══════════════════════════════════════════════════════════════════════
def calcular_correlacoes(tickers):
    """Calcula matriz de correlação entre ativos — evita concentração correlacionada"""
    try:
        dados = {}
        for t in tickers:
            hist = yf.Ticker(t).history(period="6mo")
            if not hist.empty:
                dados[t] = hist["Close"].pct_change().dropna()
        if len(dados) < 2: return {}
        df    = pd.DataFrame(dados).dropna()
        corr  = df.corr()
        return corr.to_dict()
    except Exception as e:
        log.error(f"Correlação: {e}"); return {}

def penalidade_correlacao(ticker, correlacoes, posicoes_atuais):
    """
    Retorna penalidade de 0 a 20 pontos se o ativo
    é muito correlacionado com ativos que já temos posição
    """
    if not correlacoes or ticker not in correlacoes: return 0
    penalidade = 0
    for outro_ticker, qtd in posicoes_atuais.items():
        if outro_ticker == ticker or qtd <= 0: continue
        corr_val = correlacoes.get(ticker, {}).get(outro_ticker, 0)
        if corr_val > 0.85:
            penalidade += 15
        elif corr_val > 0.70:
            penalidade += 8
    return min(penalidade, 20)


# ═══════════════════════════════════════════════════════════════════════
# BACKTESTING
# ═══════════════════════════════════════════════════════════════════════
def rodar_backtest(periodo_anos=3):
    """
    Simula a estratégia nos últimos N anos com dados históricos.
    Retorna resultado e métricas.
    """
    log.info(f"🔄 Rodando backtest de {periodo_anos} anos...")
    resultado = {
        "periodo":        f"{periodo_anos} anos",
        "capital_inicial": 10000,
        "capital_final":   0,
        "retorno_total":   0,
        "max_drawdown":    0,
        "operacoes":       0,
        "taxa_acerto":     0,
        "ativos":          {},
    }

    try:
        capital    = resultado["capital_inicial"]
        posicoes   = {}
        historico  = []
        max_cap    = capital
        min_cap    = capital

        periodo    = f"{periodo_anos}y"

        for ticker in list(ATIVOS.keys())[:4]:   # top 4 para não demorar
            t    = yf.Ticker(ticker)
            hist = t.history(period=periodo)
            if hist.empty: continue

            # Simula compra e venda baseado em RSI simples
            close  = hist["Close"]
            delta  = close.diff()
            ganho  = delta.clip(lower=0).rolling(14).mean()
            perda  = (-delta.clip(upper=0)).rolling(14).mean()
            rsi    = 100 - (100 / (1 + ganho / perda.replace(0, np.nan)))

            lucros = []
            preco_compra = None

            for i in range(20, len(close)):
                preco  = close.iloc[i]
                rsi_v  = rsi.iloc[i]

                if rsi_v < 35 and preco_compra is None and capital >= preco:
                    qtd          = int((capital * 0.15) / preco)
                    if qtd > 0:
                        preco_compra = preco
                        capital     -= qtd * preco
                        posicoes[ticker] = {"qtd": qtd, "pm": preco}
                        resultado["operacoes"] += 1

                elif preco_compra and (
                    (preco / preco_compra - 1) >= 0.20 or rsi_v > 72
                ):
                    qtd    = posicoes.get(ticker, {}).get("qtd", 0)
                    lucro  = (preco - preco_compra) * qtd
                    capital += qtd * preco
                    lucros.append(1 if lucro > 0 else 0)
                    preco_compra = None
                    del posicoes[ticker]
                    resultado["operacoes"] += 1

            acertos = sum(lucros) / len(lucros) if lucros else 0
            resultado["ativos"][ticker] = f"{acertos:.0%} acerto"
            max_cap = max(max_cap, capital)
            min_cap = min(min_cap, capital)

        # Valor final incluindo posições abertas
        for ticker, pos in posicoes.items():
            t    = yf.Ticker(ticker)
            hist = t.history(period="1d")
            if not hist.empty:
                capital += pos["qtd"] * hist["Close"].iloc[-1]

        resultado["capital_final"]  = round(capital, 2)
        resultado["retorno_total"]  = round((capital / resultado["capital_inicial"] - 1) * 100, 1)
        resultado["max_drawdown"]   = round((min_cap / resultado["capital_inicial"] - 1) * 100, 1)

        log.info(f"✅ Backtest concluído | Retorno: {resultado['retorno_total']}% | Operações: {resultado['operacoes']}")
    except Exception as e:
        log.error(f"Backtest erro: {e}")

    return resultado


# ═══════════════════════════════════════════════════════════════════════
# ANÁLISE PROFUNDA
# ═══════════════════════════════════════════════════════════════════════
def calcular_rsi(series, n=14):
    d = series.diff()
    g = d.clip(lower=0).rolling(n).mean()
    p = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + g / p.replace(0, np.nan)))

def frequencia_para_vezes(freq):
    return {"mensal": 12, "trimestral": 4, "semestral": 2, "anual": 1}.get(freq, 4)

def analisar_ativo(ticker):
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="6mo")
        info = t.info
        if hist.empty: return None

        preco    = round(float(hist["Close"].iloc[-1]), 4)
        media3m  = round(float(hist["Close"].tail(63).mean()), 4)
        rsi_val  = calcular_rsi(hist["Close"]).iloc[-1]
        rsi_val  = round(float(rsi_val), 1) if not np.isnan(rsi_val) else 50.0

        mm20 = hist["Close"].tail(20).mean()
        mm50 = hist["Close"].tail(50).mean()
        tend_num = 1 if mm20 > mm50 else -1

        dy_dec        = info.get("dividendYield") or 0
        div_acao      = round(info.get("lastDividendValue") or 0, 4)
        freq          = ATIVOS[ticker]["frequencia"]
        vezes_ano     = frequencia_para_vezes(freq)
        div_anual     = round(div_acao * vezes_ano, 4)
        renda_mensal  = round(div_anual / 12, 6)

        prox_div = info.get("exDividendDate")
        try:    prox_div = datetime.fromtimestamp(prox_div).strftime("%d/%m/%Y") if prox_div else "—"
        except: prox_div = "—"

        preco_alvo = info.get("targetMeanPrice") or 0
        upside     = round((preco_alvo - preco) / preco * 100, 1) if preco_alvo > 0 else 0

        return {
            "ticker":            ticker,
            "nome":              ATIVOS[ticker]["nome"],
            "setor":             ATIVOS[ticker]["setor"],
            "frequencia":        freq,
            "preco":             preco,
            "media3m":           media3m,
            "rsi":               rsi_val,
            "tendencia":         "ALTA" if tend_num == 1 else "BAIXA",
            "tendencia_num":     tend_num,
            "dy":                round(dy_dec * 100, 2),
            "div_por_acao":      div_acao,
            "div_anual":         div_anual,
            "renda_mensal_acao": renda_mensal,
            "prox_dividendo":    prox_div,
            "pe":                round(info.get("trailingPE") or 999, 1),
            "var_media":         round((preco - media3m) / media3m * 100, 2),
            "upside":            upside,
            "preco_alvo":        round(preco_alvo, 2),
            "52w_high":          round(info.get("fiftyTwoWeekHigh") or 0, 2),
            "52w_low":           round(info.get("fiftyTwoWeekLow") or 0, 2),
            "market_cap_bi":     round((info.get("marketCap") or 0) / 1e9, 1),
        }
    except Exception as e:
        log.error(f"Erro {ticker}: {e}"); return None

def calcular_score(a, correlacoes=None, posicoes_atuais=None):
    score   = 0
    razoes  = []

    # RSI (0-40)
    if a["rsi"] < 30:    score += 40; razoes.append(f"RSI {a['rsi']} extremo ↑+40")
    elif a["rsi"] < 35:  score += 30; razoes.append(f"RSI {a['rsi']} sobrevenda ↑+30")
    elif a["rsi"] < 45:  score += 20; razoes.append(f"RSI {a['rsi']} ↑+20")
    elif a["rsi"] < 55:  score += 10; razoes.append(f"RSI {a['rsi']} neutro ↑+10")
    elif a["rsi"] > 72:  score -= 20; razoes.append(f"RSI {a['rsi']} sobrecompra ↓-20")

    # Variação vs média (0-25)
    if a["var_media"] < -8:   score += 25; razoes.append(f"Queda forte {a['var_media']}% ↑+25")
    elif a["var_media"] < -5: score += 18; razoes.append(f"Queda {a['var_media']}% ↑+18")
    elif a["var_media"] < -2: score += 10; razoes.append(f"Queda leve {a['var_media']}% ↑+10")
    elif a["var_media"] > 10: score -= 10; razoes.append(f"Alta forte ↓-10")

    # Posição 52 semanas (0-15)
    if a["52w_high"] > 0 and (a["52w_high"] - a["52w_low"]) > 0:
        p52 = (a["preco"] - a["52w_low"]) / (a["52w_high"] - a["52w_low"])
        if p52 < 0.25:   score += 15; razoes.append("Próximo mínima 52s ↑+15")
        elif p52 < 0.40: score += 8;  razoes.append("Abaixo metade 52s ↑+8")
        elif p52 > 0.90: score -= 10; razoes.append("Próximo máxima 52s ↓-10")

    # DY (0-20)
    if a["dy"] >= 5.0:   score += 20; razoes.append(f"DY {a['dy']}% excelente ↑+20")
    elif a["dy"] >= 3.5: score += 15; razoes.append(f"DY {a['dy']}% bom ↑+15")
    elif a["dy"] >= 2.0: score += 8;  razoes.append(f"DY {a['dy']}% ok ↑+8")
    elif a["dy"] < 1.5:  score -= 5;  razoes.append(f"DY {a['dy']}% baixo ↓-5")

    # P/L (0-15)
    if a["pe"] < 20:    score += 15; razoes.append(f"P/L {a['pe']}x barato ↑+15")
    elif a["pe"] < 35:  score += 8;  razoes.append(f"P/L {a['pe']}x ok ↑+8")
    elif a["pe"] > 60:  score -= 10; razoes.append(f"P/L {a['pe']}x caro ↓-10")

    # Prioridade e frequência (0-25)
    p = ATIVOS.get(a["ticker"], {}).get("prioridade", 3)
    if p == 1:    score += 15; razoes.append("Ativo prioritário ↑+15")
    elif p == 2:  score += 8;  razoes.append("Ativo secundário ↑+8")
    if ATIVOS.get(a["ticker"], {}).get("frequencia") == "mensal":
        score += 10; razoes.append("Pagamento mensal ↑+10")

    # Penalidade de correlação
    if correlacoes and posicoes_atuais:
        pen = penalidade_correlacao(a["ticker"], correlacoes, posicoes_atuais)
        if pen > 0:
            score -= pen
            razoes.append(f"Alta correlação ↓-{pen}")

    return max(0, min(100, score)), razoes

def calcular_metas(analises, meta_mensal):
    metas     = {}
    total_peso = 0
    pesos     = {}
    for a in analises:
        t     = a["ticker"]
        p     = ATIVOS[t]["prioridade"]
        fb    = 1.5 if a["frequencia"] == "mensal" else 1.0
        db    = min(a["dy"] / 3.0, 2.0) if a["dy"] > 0 else 0.5
        pesos[t] = max((4 - p) * fb * db, 0.1)
        total_peso += pesos[t]

    for a in analises:
        t      = a["ticker"]
        prop   = pesos[t] / total_peso
        alvo   = meta_mensal * prop
        un     = math.ceil(alvo / a["renda_mensal_acao"]) if a["renda_mensal_acao"] > 0 else 0
        metas[t] = {
            "unidades_meta":     un,
            "renda_alvo_mensal": round(alvo, 2),
            "custo_total_meta":  round(un * a["preco"], 2),
            "proporcao":         round(prop * 100, 1),
        }
    return metas

def calcular_qtd_inteligente(a, saldo, patrimonio, qtd_atual, meta_un, score):
    if saldo < a["preco"]: return 0
    if score >= 80:    fator = 1.0
    elif score >= 65:  fator = 0.70
    elif score >= 55:  fator = 0.40
    else:              return 0

    progresso = qtd_atual / meta_un if meta_un > 0 else 0.5
    if progresso < 0.25:    urgencia = 1.0
    elif progresso < 0.50:  urgencia = 0.75
    elif progresso < 0.75:  urgencia = 0.50
    elif progresso < 1.00:  urgencia = 0.30
    else:                   urgencia = 0.10

    max_val   = patrimonio * CRITERIOS["max_pct_carteira"]
    val_atual = qtd_atual * a["preco"]
    espaco    = max(0, max_val - val_atual)
    capital   = min(saldo * 0.40, espaco) * fator * urgencia
    return max(0, int(capital / a["preco"]))

def projetar_meses(qtd_atual, meta_un, aporte_mes, preco, renda_acao):
    if meta_un <= 0 or qtd_atual >= meta_un: return 0
    un_mes = (aporte_mes / preco) + (qtd_atual * renda_acao / preco)
    if un_mes <= 0: return 999
    return math.ceil((meta_un - qtd_atual) / un_mes)


# ═══════════════════════════════════════════════════════════════════════
# TELEGRAM — mensagens E comandos
# ═══════════════════════════════════════════════════════════════════════
def enviar_telegram(msg):
    token, chat = CONFIG["TELEGRAM_TOKEN"], CONFIG["TELEGRAM_CHAT_ID"]
    if "SEU_TOKEN" in token:
        log.info(f"[TG]\n{msg[:300]}"); return
    try:
        for i in range(0, len(msg), 4000):
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": msg[i:i+4000], "parse_mode": "Markdown"},
                timeout=10,
            )
            time.sleep(0.3)
    except Exception as e:
        log.error(f"TG: {e}")

def processar_comando(texto, carteira, alpaca, modelo_ml):
    """Processa comandos recebidos pelo Telegram"""
    cmd = texto.strip().lower().split()
    if not cmd: return

    if cmd[0] == "/status":
        pat = alpaca.patrimonio()
        sal = alpaca.saldo()
        rd  = ESTADO["renda_atual"]
        pct = rd / CONFIG["META_RENDA_MENSAL"] * 100
        bear = "🐻 SIM" if ESTADO["bear_market"] else "🟢 NÃO"
        pausado = "⏸ SIM" if CONFIG["PAUSADO"] else "▶️ NÃO"
        msg = (f"📊 *Status InfraInvestor*\n"
               f"💼 Patrimônio: ${pat:,.2f}\n"
               f"💵 Saldo: ${sal:,.2f}\n"
               f"💰 Renda: ${rd:.2f}/mês ({pct:.1f}% da meta)\n"
               f"🐻 Bear market: {bear}\n"
               f"⏸ Pausado: {pausado}\n"
               f"🤖 ML treinado: {'✅' if modelo_ml.treinado else '❌'}\n"
               f"🕐 Último ciclo: {ESTADO['ultimo_ciclo'] or 'Nunca'}")
        enviar_telegram(msg)

    elif cmd[0] == "/pausar":
        CONFIG["PAUSADO"] = True
        enviar_telegram("⏸ *Bot pausado* — posições mantidas, sem novas operações")

    elif cmd[0] == "/retomar":
        CONFIG["PAUSADO"] = False
        enviar_telegram("▶️ *Bot retomado* — operações ativas")

    elif cmd[0] == "/meta" and len(cmd) > 1:
        try:
            nova = float(cmd[1])
            CONFIG["META_RENDA_MENSAL"] = nova
            enviar_telegram(f"🎯 *Meta atualizada:* ${nova:.0f}/mês")
        except:
            enviar_telegram("❌ Uso: /meta 1000")

    elif cmd[0] == "/radar":
        if not ESTADO["analises"]:
            enviar_telegram("⏳ Aguardando próximo ciclo de análise...")
            return
        msg = "📡 *Radar atual:*\n"
        for a in sorted(ESTADO["analises"], key=lambda x: -calcular_score(x)[0]):
            s, _ = calcular_score(a)
            ico  = "🟢" if s >= 65 else ("🟡" if s >= 45 else "🔴")
            msg += f"{ico} *{a['ticker']}* Score:{s} RSI:{a['rsi']} DY:{a['dy']}%\n"
        enviar_telegram(msg)

    elif cmd[0] == "/historico":
        ops = carteira.dados.get("historico", [])[-10:]
        if not ops:
            enviar_telegram("📋 Nenhuma operação registrada ainda")
            return
        msg = "📋 *Últimas 10 operações:*\n"
        for op in reversed(ops):
            e   = "🟢" if op["tipo"] == "COMPRA" else "🔴"
            dt  = op["data"][:10]
            msg += f"{e} {dt} {op['tipo']} {op['ticker']} x{op.get('qtd','?')} @ ${op.get('preco','?')}\n"
        enviar_telegram(msg)

    elif cmd[0] == "/backtest":
        enviar_telegram("⏳ Rodando backtest de 3 anos... pode demorar 1-2 minutos")
        r = rodar_backtest(3)
        msg = (f"📈 *Backtest {r['periodo']}*\n"
               f"💰 Capital inicial: ${r['capital_inicial']:,}\n"
               f"💼 Capital final: ${r['capital_final']:,}\n"
               f"📊 Retorno total: *{r['retorno_total']}%*\n"
               f"📉 Max drawdown: {r['max_drawdown']}%\n"
               f"🔄 Operações: {r['operacoes']}\n\n"
               f"*Por ativo:*\n")
        for t, v in r["ativos"].items():
            msg += f"  {t}: {v}\n"
        enviar_telegram(msg)

    elif cmd[0] == "/ml":
        if not ML_DISPONIVEL:
            enviar_telegram("❌ scikit-learn não instalado")
            return
        ops = carteira.dados.get("operacoes_ml", [])
        com_resultado = [o for o in ops if o.get("resultado") is not None]
        if modelo_ml.treinado:
            imp = modelo_ml.importancia_features()
            msg = (f"🤖 *Machine Learning*\n"
                   f"Status: ✅ Treinado\n"
                   f"Amostras: {len(com_resultado)}\n\n"
                   f"*Importância das features:*\n")
            for feat, val in sorted(imp.items(), key=lambda x: -x[1])[:5]:
                bar = "█" * int(val * 20)
                msg += f"  {feat}: {bar} {val:.1%}\n"
        else:
            msg = (f"🤖 *Machine Learning*\n"
                   f"Status: ❌ Não treinado\n"
                   f"Amostras com resultado: {len(com_resultado)}/10\n"
                   f"Precisa de 10 operações finalizadas para treinar")
        enviar_telegram(msg)

    elif cmd[0] == "/ajuda":
        msg = ("*Comandos disponíveis:*\n"
               "/status — patrimônio e estado atual\n"
               "/radar — score de todos os ativos\n"
               "/historico — últimas 10 operações\n"
               "/meta 1000 — muda meta para $1000/mês\n"
               "/pausar — pausa operações\n"
               "/retomar — retoma operações\n"
               "/backtest — testa estratégia em 3 anos\n"
               "/ml — status do machine learning\n"
               "/ajuda — esta mensagem")
        enviar_telegram(msg)

def ouvir_telegram(carteira, alpaca, modelo_ml):
    """Thread que fica ouvindo comandos do Telegram"""
    token  = CONFIG["TELEGRAM_TOKEN"]
    if "SEU_TOKEN" in token: return
    offset = 0
    log.info("🎧 Ouvindo comandos Telegram...")
    while True:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )
            updates = r.json().get("result", [])
            for upd in updates:
                offset = upd["update_id"] + 1
                msg    = upd.get("message", {})
                texto  = msg.get("text", "")
                if texto.startswith("/"):
                    log.info(f"Comando recebido: {texto}")
                    processar_comando(texto, carteira, alpaca, modelo_ml)
        except Exception as e:
            log.debug(f"TG poll: {e}")
        time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════
# DASHBOARD WEB
# ═══════════════════════════════════════════════════════════════════════
def gerar_html_dashboard(carteira):
    snaps    = carteira.dados.get("snapshots", [])
    labels   = json.dumps([s["data"][:10] for s in snaps[-30:]])
    valores  = json.dumps([s["patrimonio"] for s in snaps[-30:]])
    rendas   = json.dumps([s["renda_mensal"] for s in snaps[-30:]])
    analises = ESTADO["analises"]
    metas    = ESTADO["metas"]
    renda_at = ESTADO["renda_atual"]
    pat      = ESTADO["patrimonio"]
    pct_meta = renda_at / CONFIG["META_RENDA_MENSAL"] * 100 if CONFIG["META_RENDA_MENSAL"] > 0 else 0

    rows_ativos = ""
    for a in analises:
        score, _ = calcular_score(a)
        qtd   = carteira.quantidade(a["ticker"])
        meta  = metas.get(a["ticker"], {}).get("unidades_meta", 0)
        pct   = qtd / meta * 100 if meta > 0 else 0
        renda = qtd * a["renda_mensal_acao"]
        cor   = "#00e676" if score >= 65 else ("#ffd600" if score >= 45 else "#ff3d57")
        rows_ativos += f"""
        <tr>
          <td><b>{a['ticker']}</b><br><small style="color:#546e7a">{a['nome']}</small></td>
          <td>${a['preco']}</td>
          <td style="color:{cor}">{score}/100</td>
          <td>{a['rsi']}</td>
          <td>{a['dy']}%</td>
          <td>{a['frequencia'][:3].upper()}</td>
          <td>{qtd:.1f}/{meta}</td>
          <td>
            <div style="background:#1a2535;border-radius:4px;height:8px;width:100px">
              <div style="background:{cor};height:8px;border-radius:4px;width:{min(pct,100):.0f}px"></div>
            </div>
            <small>{pct:.0f}%</small>
          </td>
          <td style="color:#00e676">${renda:.2f}/mês</td>
          <td>{a['prox_dividendo']}</td>
        </tr>"""

    ops_html = ""
    for op in reversed(carteira.dados.get("historico", [])[-15:]):
        cor = "#00e676" if op["tipo"] == "COMPRA" else "#ff3d57"
        ops_html += f"""
        <tr>
          <td style="color:#546e7a">{op['data'][:16]}</td>
          <td style="color:{cor}">{op['tipo']}</td>
          <td><b>{op['ticker']}</b></td>
          <td>{op.get('qtd','—')}</td>
          <td>${op.get('preco','—')}</td>
          <td style="color:{cor}">${op.get('lucro','—')}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>InfraInvestor Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#05080d;color:#b0bec5;font-family:'Segoe UI',sans-serif;padding:20px}}
  h1{{color:#00e676;font-size:22px;margin-bottom:4px}}
  .sub{{color:#546e7a;font-size:12px;margin-bottom:24px}}
  .kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
  .kpi{{background:#0a0f18;border:1px solid #1a2535;padding:20px;border-radius:4px}}
  .kpi-label{{font-size:10px;letter-spacing:2px;color:#546e7a;text-transform:uppercase;margin-bottom:8px}}
  .kpi-val{{font-size:28px;font-weight:800;color:#fff}}
  .kpi-val.g{{color:#00e676}} .kpi-val.b{{color:#2979ff}} .kpi-val.o{{color:#ff9100}}
  .card{{background:#0a0f18;border:1px solid #1a2535;padding:20px;border-radius:4px;margin-bottom:16px}}
  .card h2{{font-size:11px;letter-spacing:2px;color:#546e7a;text-transform:uppercase;margin-bottom:16px}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{color:#546e7a;font-size:9px;letter-spacing:1px;padding:8px;border-bottom:1px solid #1a2535;text-align:left;text-transform:uppercase}}
  td{{padding:10px 8px;border-bottom:1px solid rgba(26,37,53,0.5)}}
  .charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
  @media(max-width:800px){{.kpis{{grid-template-columns:1fr 1fr}}.charts{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>⚡ InfraInvestor</h1>
<div class="sub">Dashboard • Atualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')} • {'📄 PAPER' if 'paper' in CONFIG['ALPACA_BASE_URL'] else '🚨 REAL'}</div>

<div class="kpis">
  <div class="kpi"><div class="kpi-label">Patrimônio</div><div class="kpi-val">${pat:,.0f}</div></div>
  <div class="kpi"><div class="kpi-label">Renda Mensal</div><div class="kpi-val g">${renda_at:.2f}</div></div>
  <div class="kpi"><div class="kpi-label">Meta</div><div class="kpi-val b">{pct_meta:.1f}%</div></div>
  <div class="kpi"><div class="kpi-label">Bear Market</div><div class="kpi-val {'o' if ESTADO['bear_market'] else 'g'}">{'⚠️ SIM' if ESTADO['bear_market'] else '✅ NÃO'}</div></div>
</div>

<div class="charts">
  <div class="card"><h2>Patrimônio (30 dias)</h2><canvas id="c1" height="200"></canvas></div>
  <div class="card"><h2>Renda Mensal (30 dias)</h2><canvas id="c2" height="200"></canvas></div>
</div>

<div class="card">
  <h2>Radar de Ativos</h2>
  <table><thead><tr>
    <th>Ativo</th><th>Preço</th><th>Score</th><th>RSI</th><th>DY</th>
    <th>Freq</th><th>Unidades</th><th>Progresso</th><th>Renda/Mês</th><th>Próx Div</th>
  </tr></thead><tbody>{rows_ativos}</tbody></table>
</div>

<div class="card">
  <h2>Histórico de Operações</h2>
  <table><thead><tr><th>Data</th><th>Tipo</th><th>Ativo</th><th>Qtd</th><th>Preço</th><th>Lucro</th></tr></thead>
  <tbody>{ops_html}</tbody></table>
</div>

<script>
const opts = {{
  plugins:{{legend:{{labels:{{color:'#546e7a',font:{{size:10}}}}}}}},
  scales:{{x:{{grid:{{color:'rgba(26,37,53,0.8)'}},ticks:{{color:'#546e7a',font:{{size:9}},maxTicksLimit:6}}}},
           y:{{grid:{{color:'rgba(26,37,53,0.8)'}},ticks:{{color:'#546e7a',font:{{size:9}}}}}}}}
}};
new Chart(document.getElementById('c1'),{{type:'line',data:{{labels:{labels},
  datasets:[{{label:'Patrimônio',data:{valores},borderColor:'#2979ff',backgroundColor:'rgba(41,121,255,0.05)',tension:0.4,fill:true,pointRadius:2}}]}},options:opts}});
new Chart(document.getElementById('c2'),{{type:'line',data:{{labels:{labels},
  datasets:[{{label:'Renda/mês',data:{rendas},borderColor:'#00e676',backgroundColor:'rgba(0,230,118,0.05)',tension:0.4,fill:true,pointRadius:2}}]}},options:opts}});
setTimeout(()=>location.reload(), 300000);
</script>
</body></html>"""

class DashboardHandler(BaseHTTPRequestHandler):
    carteira = None
    def do_GET(self):
        html = gerar_html_dashboard(self.carteira).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html)
    def log_message(self, *args): pass

def iniciar_dashboard(carteira):
    DashboardHandler.carteira = carteira
    server = HTTPServer(("0.0.0.0", CONFIG["DASHBOARD_PORT"]), DashboardHandler)
    log.info(f"🌐 Dashboard: http://localhost:{CONFIG['DASHBOARD_PORT']}")
    server.serve_forever()


# ═══════════════════════════════════════════════════════════════════════
# RELATÓRIO TELEGRAM
# ═══════════════════════════════════════════════════════════════════════
def barra(atual, meta, n=10):
    if meta <= 0: return "─" * n
    pct = min(atual / meta, 1.0)
    return "█" * int(pct * n) + "░" * (n - int(pct * n))

def montar_relatorio(modo, patrimonio, saldo, acoes, analises, carteira, metas, renda, bear, ml_ativo):
    pct_meta = renda / CONFIG["META_RENDA_MENSAL"] * 100
    msg  = f"📊 *InfraInvestor [{modo}]*\n"
    msg += f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
    if bear: msg += f"⚠️ *BEAR MARKET DETECTADO — modo defensivo*\n"
    if ml_ativo: msg += f"🤖 ML ativo\n"
    msg += f"{'─'*30}\n\n"
    msg += f"💼 *${patrimonio:,.2f}* | 💵 Livre: ${saldo:,.2f}\n"
    msg += f"💰 Renda: *${renda:.2f}/mês* ({pct_meta:.1f}% da meta ${CONFIG['META_RENDA_MENSAL']:.0f})\n\n"

    if acoes:
        msg += "⚡ *Operações:*\n"
        for op in acoes:
            e = "🟢" if op["tipo"] == "COMPRA" else "🔴"
            msg += f"{e} *{op['tipo']}* {op['ticker']} x{op['qtd']} @ ${op['preco']:.2f}\n"
            msg += f"   Score:{op['score']}/100"
            if op.get("prob_ml"): msg += f" ML:{op['prob_ml']:.0%}"
            msg += f"\n   _{op['motivo']}_\n"
        msg += "\n"
    else:
        msg += "⏳ Aguardando sinal suficiente...\n\n"

    msg += "📦 *Progresso de Unidades:*\n"
    for a in analises:
        t     = a["ticker"]
        qtd   = carteira.quantidade(t)
        meta  = metas.get(t, {}).get("unidades_meta", 0)
        b     = barra(qtd, meta)
        pct   = qtd / meta * 100 if meta > 0 else 0
        rend  = round(qtd * a["renda_mensal_acao"], 2)
        meses = projetar_meses(qtd, meta, 100, a["preco"], a["renda_mensal_acao"])
        msg  += f"\n*{t}*: {b} {qtd:.0f}/{meta} ({pct:.0f}%)\n"
        msg  += f"  💰 ${rend:.2f}/mês"
        if t in ESTADO["ativos_pausados"]: msg += " ⚠️ NOTÍCIA CRÍTICA"
        if meses > 0 and meses < 999: msg += f" | ~{meses}m para meta"
        msg  += "\n"

    return msg


# ═══════════════════════════════════════════════════════════════════════
# CICLO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════
def ciclo(alpaca, carteira, modelo_ml):
    log.info("=" * 75)

    if CONFIG["PAUSADO"]:
        log.info("⏸ Bot pausado — pulando ciclo")
        return

    saldo       = alpaca.saldo()
    patrimonio  = alpaca.patrimonio()
    posicoes    = alpaca.posicoes()
    ESTADO["patrimonio"] = patrimonio

    # Verifica drawdown
    dd = carteira.drawdown_atual(patrimonio)
    if dd < CONFIG["MAX_DRAWDOWN"]:
        log.warning(f"⚠️ Drawdown {dd:.1%} — pausando operações")
        CONFIG["PAUSADO"] = True
        enviar_telegram(f"⚠️ *Drawdown {dd:.1%} atingido — operações pausadas*\nUse /retomar para continuar")
        return

    # Bear market
    ESTADO["bear_market"] = detectar_bear_market()

    # Correlações
    log.info("Calculando correlações...")
    correlacoes = calcular_correlacoes(list(ATIVOS.keys()))

    # Analisa empresas
    log.info("Analisando empresas...")
    analises = []
    for ticker in ATIVOS:
        # Verifica notícias críticas
        critico, alertas = verificar_noticias(ticker, ATIVOS[ticker]["nome"])
        if critico:
            ESTADO["ativos_pausados"].add(ticker)
            enviar_telegram(f"⚠️ *Notícia crítica {ticker}:* {', '.join(alertas)}\nOperações pausadas para este ativo")
        else:
            ESTADO["ativos_pausados"].discard(ticker)

        a = analisar_ativo(ticker)
        if a:
            analises.append(a)
        time.sleep(0.5)

    if not analises: return

    ESTADO["analises"] = analises

    # Metas
    metas = calcular_metas(analises, CONFIG["META_RENDA_MENSAL"])
    carteira.atualizar_metas(metas)
    ESTADO["metas"] = metas

    # Treina ML se tiver dados suficientes
    if ML_DISPONIVEL:
        modelo_ml.treinar(carteira.dados.get("operacoes_ml", []))

    # Sincroniza posições
    for ticker, pos_alp in posicoes.items():
        if ticker in ATIVOS and carteira.quantidade(ticker) == 0:
            carteira.registrar_compra(
                ticker,
                float(pos_alp.get("avg_entry_price", 0)),
                float(pos_alp.get("qty", 0)),
                "Sync Alpaca",
            )

    # Posições atuais para correlação
    pos_atuais = {t: carteira.quantidade(t) for t in ATIVOS}

    # Decisões — melhores scores primeiro
    acoes       = []
    renda_total = 0.0
    analises_ord = sorted(analises, key=lambda x: -calcular_score(x, correlacoes, pos_atuais)[0])

    for a in analises_ord:
        ticker    = a["ticker"]
        score, razoes = calcular_score(a, correlacoes, pos_atuais)
        pm        = carteira.preco_medio(ticker)
        qtd_atual = carteira.quantidade(ticker)
        pos_alp   = posicoes.get(ticker)
        meta_un   = metas.get(ticker, {}).get("unidades_meta", 0)
        renda_total += qtd_atual * a["renda_mensal_acao"]

        log.info(f"  {ticker:5s} | ${a['preco']:>8.2f} | RSI={a['rsi']:>5.1f} | Score={score:>3}/100 | {a['tendencia']}")

        # VENDA
        if pm > 0:
            ret = (a["preco"] - pm) / pm
            if ret >= CRITERIOS["meta_lucro_venda"]:
                qtd_v  = max(1, int(qtd_atual * 0.5))
                motivo = f"Lucro +{ret*100:.1f}% | Vende 50%"
                if pos_alp and alpaca.vender(ticker, qtd_v, motivo):
                    lucro = carteira.registrar_venda(ticker, a["preco"], qtd_v, motivo)
                    acoes.append({"tipo":"VENDA","ticker":ticker,"qtd":qtd_v,
                                  "preco":a["preco"],"score":score,"motivo":motivo,"lucro":lucro})
                    saldo += qtd_v * a["preco"]
                continue

        # COMPRA — pula se bear market ou ativo com notícia crítica
        if ESTADO["bear_market"] and ATIVOS[ticker]["prioridade"] > 1:
            continue
        if ticker in ESTADO["ativos_pausados"]:
            continue
        if score < CRITERIOS["score_minimo"]:
            continue

        # Verificação ML
        features     = modelo_ml.extrair_features(a)
        prob_ml      = modelo_ml.prever(features)
        if prob_ml is not None and prob_ml < CRITERIOS["score_minimo_ml"]:
            log.info(f"  {ticker}: ML bloqueou compra (prob={prob_ml:.1%})")
            continue

        if saldo < a["preco"]: continue

        qtd = calcular_qtd_inteligente(a, saldo, patrimonio, qtd_atual, meta_un, score)
        if qtd > 0 and alpaca.comprar(ticker, qtd, " | ".join(razoes[:2])):
            feats_dict = {f"f{i}": v for i, v in enumerate(features)}
            carteira.registrar_compra(ticker, a["preco"], qtd, " | ".join(razoes[:2]), feats_dict)
            acoes.append({"tipo":"COMPRA","ticker":ticker,"qtd":qtd,
                          "preco":a["preco"],"score":score,
                          "motivo":" | ".join(razoes[:2]),"prob_ml":prob_ml})
            saldo -= qtd * a["preco"]
            pos_atuais[ticker] = carteira.quantidade(ticker)

        time.sleep(0.3)

    ESTADO["renda_atual"]  = renda_total
    ESTADO["ultimo_ciclo"] = datetime.now().strftime("%d/%m %H:%M")

    carteira.salvar_snapshot(patrimonio, renda_total)

    # Log tabela
    log.info("")
    log.info(f"  {'TICKER':<6} {'PREÇO':>8} {'RSI':>5} {'SCORE':>5} {'DY':>5} {'ATUAL':>7} {'META':>6} {'RENDA/MES':>10}")
    for a in analises:
        s, _ = calcular_score(a, correlacoes, pos_atuais)
        qtd  = carteira.quantidade(a["ticker"])
        meta = metas.get(a["ticker"], {}).get("unidades_meta", 0)
        r    = qtd * a["renda_mensal_acao"]
        log.info(f"  {a['ticker']:<6} ${a['preco']:>7.2f} {a['rsi']:>5.1f} {s:>5} {a['dy']:>4.1f}% {qtd:>7.1f} {meta:>6} ${r:>8.2f}/mês")
    log.info(f"  {'─'*65}")
    log.info(f"  Renda total: ${renda_total:.2f}/mês | Meta: ${CONFIG['META_RENDA_MENSAL']:.0f}/mês ({renda_total/CONFIG['META_RENDA_MENSAL']*100:.1f}%)")
    log.info("")

    modo = "📄 PAPER" if "paper" in CONFIG["ALPACA_BASE_URL"] else "🚨 REAL"
    msg  = montar_relatorio(modo, patrimonio, saldo, acoes, analises, carteira, metas, renda_total,
                            ESTADO["bear_market"], modelo_ml.treinado)
    enviar_telegram(msg)
    log.info("Ciclo finalizado.")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    modo = "PAPER TRADING 📄" if "paper" in CONFIG["ALPACA_BASE_URL"] else "🚨 REAL"
    log.info("╔════════════════════════════════════════════════════╗")
    log.info("║   InfraInvestor Bot v3 — ML + Backtest + Risk     ║")
    log.info("╚════════════════════════════════════════════════════╝")
    log.info(f"Modo: {modo} | Meta: ${CONFIG['META_RENDA_MENSAL']:.0f}/mês")
    log.info(f"ML disponível: {ML_DISPONIVEL}")

    alpaca    = AlpacaClient()
    carteira  = Carteira()
    modelo_ml = ModeloML() if ML_DISPONIVEL else type("FakeML", (), {
        "treinado": False, "treinar": lambda *a: None,
        "prever": lambda *a: None, "extrair_features": lambda *a: [],
        "importancia_features": lambda *a: {},
    })()

    conta = alpaca.conta()
    if not conta.get("id"):
        log.error("❌ Falha Alpaca — verifique as chaves"); return

    log.info(f"✅ Alpaca OK | Patrimônio: ${float(conta.get('portfolio_value',0)):,.2f}")

    # Inicia dashboard em thread separada
    t_dash = threading.Thread(target=iniciar_dashboard, args=(carteira,), daemon=True)
    t_dash.start()

    # Inicia listener de comandos Telegram em thread separada
    t_tg = threading.Thread(target=ouvir_telegram, args=(carteira, alpaca, modelo_ml), daemon=True)
    t_tg.start()

    enviar_telegram(
        f"🚀 *InfraInvestor v3 iniciado!*\n"
        f"Modo: *{modo}*\n"
        f"🎯 Meta: *${CONFIG['META_RENDA_MENSAL']:.0f}/mês*\n"
        f"🤖 ML: {'✅ ativo' if ML_DISPONIVEL else '❌ instale scikit-learn'}\n"
        f"🌐 Dashboard: http://localhost:{CONFIG['DASHBOARD_PORT']}\n"
        f"💬 Digite /ajuda para ver os comandos"
    )

    while True:
        try:
            ciclo(alpaca, carteira, modelo_ml)
            log.info(f"⏳ Próximo ciclo em {CONFIG['INTERVALO_ANALISE']//60} minutos...")
            time.sleep(CONFIG["INTERVALO_ANALISE"])
        except KeyboardInterrupt:
            log.info("Bot encerrado."); break
        except Exception as e:
            log.error(f"Erro: {e}"); time.sleep(60)

if __name__ == "__main__":
    main()