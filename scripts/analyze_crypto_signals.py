import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pandas.api.types import is_datetime64_any_dtype as is_datetime

# Ajustar estilo do matplotlib (pode ser alterado conforme preferência)
plt.style.use('ggplot')

# =============================================================================
# CONFIGURAÇÕES DE LOG
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÕES GERAIS
# =============================================================================
DB_PATH = "data/crypto_data.db"       # Caminho do banco SQLite
PLOTS_FOLDER = "reports"      # Pasta onde serão salvos os gráficos
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# =============================================================================
# CLASSE DE LEITURA E CRIAÇÃO DE TABELAS
# =============================================================================
class CryptoDataManager:
    """
    - Carrega todos os dados OHLCV da tabela 'ohlcv_data' (todas as criptos, dados diários).
    - Cria a tabela 'best_trades' para salvar as melhores oportunidades de compra.
    """
    def __init__(self, db_path=DB_PATH):
        self.db_engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self._create_best_trades_table()

    def _create_best_trades_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS best_trades (
            symbol TEXT,
            timeframe TEXT,
            timestamp DATETIME,
            close REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            adx REAL,
            plus_di REAL,
            minus_di REAL,
            squeeze_on BOOLEAN,
            signal INT,
            PRIMARY KEY(symbol, timeframe, timestamp)
        );
        """
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text(sql))
            logger.info("Tabela 'best_trades' criada/verificada com sucesso.")
        except SQLAlchemyError as e:
            logger.error(f"Erro ao criar/verificar tabela 'best_trades': {e}")

    def load_all_ohlcv(self):
        sql = """
        SELECT symbol, timeframe, timestamp, open, high, low, close, volume
        FROM ohlcv_data
        ORDER BY symbol, timeframe, timestamp ASC
        """
        try:
            df = pd.read_sql_query(sql, con=self.db_engine)
            if not df.empty and not is_datetime(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Foram carregadas {len(df)} linhas da tabela 'ohlcv_data'.")
            return df
        except SQLAlchemyError as e:
            logger.error(f"Erro ao carregar OHLCV do banco: {e}")
            return pd.DataFrame()

    def save_best_trades(self, df_best):
        if df_best.empty:
            logger.warning("Nenhum 'best trade' para salvar.")
            return

        try:
            with self.db_engine.begin() as conn:
                for _, row in df_best.iterrows():
                    sql = text("""
                        INSERT OR REPLACE INTO best_trades (
                            symbol, timeframe, timestamp,
                            close,
                            macd, macd_signal, macd_hist,
                            adx, plus_di, minus_di,
                            squeeze_on, signal
                        )
                        VALUES (
                            :symbol, :timeframe, :timestamp,
                            :close,
                            :macd, :macd_signal, :macd_hist,
                            :adx, :plus_di, :minus_di,
                            :sqz_on, :signal
                        );
                    """)
                    conn.execute(sql, {
                        'symbol':       row['symbol'],
                        'timeframe':    row['timeframe'],
                        'timestamp':    row['timestamp'].isoformat(),
                        'close':        float(row['close']),
                        'macd':         float(row['macd']),
                        'macd_signal':  float(row['macd_signal']),
                        'macd_hist':    float(row['macd_hist']),
                        'adx':          float(row['adx']),
                        'plus_di':      float(row['plus_di']),
                        'minus_di':     float(row['minus_di']),
                        'sqz_on':       bool(row['squeeze_on']),
                        'signal':       int(row['signal'])
                    })
            logger.info(f"{len(df_best)} registros inseridos/atualizados em 'best_trades'.")
        except SQLAlchemyError as e:
            logger.error(f"Erro ao salvar best_trades no banco: {e}")

# =============================================================================
# FUNÇÕES DE INDICADORES (SQUEEZE, MACD, ADX/DMI)
# =============================================================================
def sma(series, length):
    return series.rolling(window=length).mean()

def stdev(series, length):
    return series.rolling(window=length).std()

def true_range(df):
    shifted_close = df['close'].shift(1)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - shifted_close).abs()
    low_close = (df['low'] - shifted_close).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

def linreg(series, length):
    def _lr(vals):
        x = np.arange(len(vals))
        y = vals.values
        if np.isnan(y).any():
            return np.nan
        b = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
        a = y.mean() - b * x.mean()
        return a + b * (length - 1)
    return series.rolling(window=length).apply(_lr, raw=False)

def squeeze_momentum_lazybear(df, lengthBB=20, multBB=2.0,
                              lengthKC=20, multKC=1.5, useTR=True):
    """
    Calcula o indicador Squeeze Momentum e adiciona as colunas:
      'squeeze_on', 'squeeze_off', 'no_squeeze' e 'squeeze_value'
    Retorna o DataFrame original com os indicadores adicionados.
    """
    df = df.copy()
    source = df['close']
    basis = sma(source, lengthBB)
    dev = multBB * stdev(source, lengthBB)
    upperBB = basis + dev
    lowerBB = basis - dev

    ma_kc = sma(source, lengthKC)
    if useTR:
        rng = true_range(df)
    else:
        rng = df['high'] - df['low']
    range_ma = rng.rolling(window=lengthKC).mean()
    upperKC = ma_kc + range_ma * multKC
    lowerKC = ma_kc - range_ma * multKC

    df['squeeze_on'] = (lowerBB > lowerKC) & (upperBB < upperKC)
    df['squeeze_off'] = (lowerBB < lowerKC) & (upperBB > upperKC)
    df['no_squeeze'] = ~(df['squeeze_on'] | df['squeeze_off'])

    highest_high = df['high'].rolling(window=lengthKC).max()
    lowest_low = df['low'].rolling(window=lengthKC).min()
    avg_hilo = (highest_high + lowest_low) / 2.0
    avg_sma = sma(source, lengthKC)
    inner_avg = (avg_hilo + avg_sma) / 2.0
    series_for_linreg = source - inner_avg
    df['squeeze_value'] = linreg(series_for_linreg, lengthKC)
    return df

def macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': macd_signal_line,
        'macd_hist': macd_hist
    })

def adx_dmi(df, period=14):
    tr = true_range(df)
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()

    return pd.DataFrame({
        'adx': adx_val,
        'plus_di': plus_di,
        'minus_di': minus_di
    })

# =============================================================================
# FUNÇÕES DE CÁLCULO E FILTRO DAS MELHORES OPORTUNIDADES
# =============================================================================
def calculate_indicators_for_all(df_all):
    """
    Agrupa os dados por (symbol, timeframe), calcula os indicadores e gera o sinal de compra.
    Retorna o DataFrame resultante com as colunas adicionais.
    """
    if df_all.empty:
        logger.warning("DataFrame vazio em 'calculate_indicators_for_all'.")
        return df_all

    def _calc_group(group):
        group = group.sort_values('timestamp').reset_index(drop=True)
        group = squeeze_momentum_lazybear(group)  # agora retorna o df completo com as novas colunas
        macd_df = macd(group)
        group = pd.concat([group, macd_df], axis=1)
        adx_df = adx_dmi(group)
        group = pd.concat([group, adx_df], axis=1)
        group['signal'] = 0
        cond_buy = (
            (group['macd'] > group['macd_signal']) &
            (group['adx'] > 25) &
            (group['plus_di'] > group['minus_di']) &
            (group['squeeze_on'] == True)
        )
        group.loc[cond_buy, 'signal'] = 1
        return group

    df_result = df_all.groupby(['symbol', 'timeframe'], group_keys=False).apply(_calc_group)
    return df_result

def filter_best_trades(df_result):
    """
    Seleciona, para cada (symbol, timeframe), a última barra onde 'signal' == 1.
    Ordena por adx, plus_di e macd (descendente) e retorna as melhores oportunidades.
    """
    if df_result.empty:
        return pd.DataFrame()

    df_last = df_result.groupby(['symbol', 'timeframe'], group_keys=False).tail(1)
    df_buy = df_last[df_last['signal'] == 1].copy()
    if df_buy.empty:
        return df_buy
    df_buy.sort_values(by=['adx', 'plus_di', 'macd'], ascending=False, inplace=True)
    return df_buy

# =============================================================================
# FUNÇÕES DE PLOTAGEM
# =============================================================================
def plot_single_crypto(df_crypto, save_path):
    """
    Plota um gráfico composto em 4 subplots:
      1) Candlestick com Bollinger Bands e Keltner Channels, com background squeeze.
      2) Histograma do Squeeze Momentum.
      3) MACD (linhas e histograma).
      4) ADX, +DI e -DI.
    O background (squeeze_on/squeeze_off) é aplicado em todos os subplots.
    """
    if df_crypto.empty:
        return
    symbol = df_crypto['symbol'].iloc[0]
    timeframe = df_crypto['timeframe'].iloc[0]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{symbol} - {timeframe}", fontsize=16)

    # Subplot 1: Candlestick + Bollinger + Keltner
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.set_title("Candlestick + Bollinger + Keltner")
    df_mpf = df_crypto.set_index('timestamp')[['open', 'high', 'low', 'close', 'volume']].copy()
    lengthBB = 20
    multBB = 2.0
    basis = df_mpf['close'].rolling(lengthBB).mean()
    dev = multBB * df_mpf['close'].rolling(lengthBB).std()
    upperBB = basis + dev
    lowerBB = basis - dev

    lengthKC = 20
    multKC = 1.5
    tr_serie = (df_mpf['high'] - df_mpf['low']).abs()
    range_ma = tr_serie.rolling(lengthKC).mean()
    ma_kc = df_mpf['close'].rolling(lengthKC).mean()
    upperKC = ma_kc + range_ma * multKC
    lowerKC = ma_kc - range_ma * multKC

    mpf.plot(df_mpf, type='candle', ax=ax1, style='charles', show_nontrading=True)
    ax1.plot(upperBB.index, upperBB, color='blue', linewidth=0.8, label='BB Up')
    ax1.plot(lowerBB.index, lowerBB, color='blue', linewidth=0.8, label='BB Down')
    ax1.plot(upperKC.index, upperKC, color='orange', linewidth=0.8, label='KC Up')
    ax1.plot(lowerKC.index, lowerKC, color='orange', linewidth=0.8, label='KC Down')
    ax1.legend(loc='upper left', fontsize=8)

    # Subplot 2: Squeeze Momentum (histograma)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax2.set_title("Squeeze Momentum (LazyBear)")
    val = df_crypto['squeeze_value']
    color_array = []
    for i in range(len(val)):
        if i == 0 or pd.isna(val[i-1]):
            color_array.append('gray')
        else:
            if val[i] > 0:
                if val[i] > val[i-1]:
                    color_array.append('lime')
                else:
                    color_array.append('green')
            else:
                if val[i] < val[i-1]:
                    color_array.append('red')
                else:
                    color_array.append('maroon')
    ax2.bar(df_crypto['timestamp'], val, color=color_array, width=0.6)
    ax2.axhline(0, color='white', linewidth=1)

    # Subplot 3: MACD
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax3.set_title("MACD")
    ax3.plot(df_crypto['timestamp'], df_crypto['macd'], label='MACD', color='blue')
    ax3.plot(df_crypto['timestamp'], df_crypto['macd_signal'], label='Signal', color='red')
    hist_colors = df_crypto['macd_hist'].apply(lambda x: 'green' if x >= 0 else 'red')
    ax3.bar(df_crypto['timestamp'], df_crypto['macd_hist'], color=hist_colors, width=0.6)
    ax3.legend(loc='upper left', fontsize=8)

    # Subplot 4: ADX / +DI / -DI
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
    ax4.set_title("ADX / +DI / -DI")
    ax4.plot(df_crypto['timestamp'], df_crypto['adx'], label='ADX', color='magenta')
    ax4.plot(df_crypto['timestamp'], df_crypto['plus_di'], label='+DI', color='green')
    ax4.plot(df_crypto['timestamp'], df_crypto['minus_di'], label='-DI', color='red')
    ax4.axhline(25, color='gray', linewidth=1, linestyle='--')
    ax4.legend(loc='upper left', fontsize=8)

    # Aplicar background squeeze em todos os subplots
    sqz_on = df_crypto['squeeze_on'].values
    sqz_off = df_crypto['squeeze_off'].values
    times = df_crypto['timestamp'].values
    for ax in [ax1, ax2, ax3, ax4]:
        for i in range(1, len(df_crypto)):
            if sqz_on[i]:
                ax.axvspan(times[i-1], times[i], color='black', alpha=0.05)
            elif sqz_off[i]:
                ax.axvspan(times[i-1], times[i], color='green', alpha=0.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_best_cryptos(df_best, df_result, folder=PLOTS_FOLDER):
    """
    Para cada (symbol, timeframe) em df_best (onde signal == 1),
    gera um gráfico a partir do histórico presente em df_result.
    """
    if df_best.empty:
        logger.info("Nenhuma cripto para plotar (df_best vazio).")
        return

    for idx, row in df_best.iterrows():
        sym = row['symbol']
        tf = row['timeframe']
        df_pair = df_result[(df_result['symbol'] == sym) & (df_result['timeframe'] == tf)]
        if df_pair.empty:
            continue
        fname = f"{sym}_{tf}.png"
        fpath = os.path.join(PLOTS_FOLDER, fname)
        plot_single_crypto(df_pair, save_path=fpath)
        logger.info(f"Plot salvo em {fpath}.")

def plot_lazybear_squeeze_with_crosses(df, save_path=None):
    """
    Plota o histograma do indicador 'squeeze_value' com cores definidas em 'hist_color'
    e, na linha zero, desenha um símbolo (marker '+') para cada candle com a cor em 'zero_line_color',
    simulando o estilo "cross" do TradingView.
    """
    if df.empty:
        logger.warning("DataFrame vazio; nada para plotar.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Squeeze Momentum Indicator [LazyBear]")

    ax.bar(df['timestamp'], df['squeeze_value'], color=df['hist_color'], width=0.6)
    ax.scatter(df['timestamp'], [0]*len(df), color=df['zero_line_color'], marker='+', s=100)
    ax.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Gráfico salvo em {save_path}.")
    else:
        plt.show()
    plt.close(fig)

# =============================================================================
# SCRIPT PRINCIPAL COM INTERAÇÃO
# =============================================================================
if __name__ == "__main__":
    manager = CryptoDataManager(DB_PATH)
    df_all = manager.load_all_ohlcv()
    if df_all.empty:
        logger.warning("Nenhum dado OHLCV encontrado no DB. Encerrando.")
        sys.exit()

    df_result = calculate_indicators_for_all(df_all)
    if df_result.empty:
        logger.warning("Erro ou DataFrame vazio após cálculo de indicadores.")
        sys.exit()

    print("\n=== Escolha uma opção ===")
    print("E: Analisar uma criptomoeda específica")
    print("M: Encontrar as melhores oportunidades entre todas")
    choice = input("Digite 'E' ou 'M': ").strip().upper()

    if choice == 'E':
        unique_symbols = sorted(df_all['symbol'].unique())
        print("\nCriptomoedas disponíveis:")
        for idx, sym in enumerate(unique_symbols):
            print(f"{idx+1}: {sym}")
        selected = input("\nDigite o número correspondente à criptomoeda desejada ou o símbolo: ").strip()
        try:
            index = int(selected) - 1
            chosen_symbol = unique_symbols[index]
        except:
            chosen_symbol = selected.lower()  # assume minúsculo

        # Como os dados são diários, o timeframe é fixo em "1d"
        timeframe_choice = "1d"
        print(f"\nUsando o timeframe '1d' para a criptomoeda {chosen_symbol}.")

        df_symbol = df_result[(df_result['symbol'] == chosen_symbol) & (df_result['timeframe'] == timeframe_choice)]
        if df_symbol.empty:
            logger.warning("Não há dados para a criptomoeda selecionada.")
            sys.exit()
        output_path = os.path.join(PLOTS_FOLDER, f"{chosen_symbol}_{timeframe_choice}_chart.png")
        plot_single_crypto(df_symbol, save_path=output_path)
        print(f"Gráfico composto salvo em {output_path}.")

        # Gráfico do indicador Squeeze com crosses
        df_lazy = squeeze_momentum_lazybear(df_symbol)
        # Definindo as cores para o histograma e para os crosses (ajuste conforme sua lógica)
        df_lazy['hist_color'] = ['lime' if x > 0 else 'red' for x in df_lazy['squeeze_value']]
        df_lazy['zero_line_color'] = ['black' if y else 'blue' for y in df_lazy['squeeze_on']]
        lazy_path = os.path.join(PLOTS_FOLDER, f"{chosen_symbol}_{timeframe_choice}_lazybear.png")
        plot_lazybear_squeeze_with_crosses(df_lazy, save_path=lazy_path)
        print(f"Gráfico do Squeeze (LazyBear) salvo em {lazy_path}.")

    elif choice == 'M':
        df_best = filter_best_trades(df_result)
        if df_best.empty:
            logger.info("Nenhuma criptomoeda gerou sinal de compra na última barra.")
        else:
            df_best = df_best.head(10)  # opcional: limitar às top 10
        manager.save_best_trades(df_best)
        plot_best_cryptos(df_best, df_result, folder=PLOTS_FOLDER)
        print("Análise das melhores oportunidades concluída. Resultados salvos e gráficos gerados.")
    else:
        print("Opção inválida. Encerrando.")
        sys.exit()

    logger.info("Processo concluído.")
