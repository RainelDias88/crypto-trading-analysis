#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
import requests
import pandas as pd
import ccxt
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

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

DB_PATH = "data/crypto_data.db"     # Caminho do banco de dados SQLite
CSV_PATH = "data/all_data.csv"      # Arquivo único onde TUDO será salvo antes de ir para o DB

# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class CryptoDataCollector:
    """
    Classe responsável por:
     - Criar/verificar tabelas no banco
     - Buscar top cryptos e OHLCV
     - Salvar tudo em um único CSV
     - Carregar o CSV no banco posteriormente
    """

    def __init__(self, db_path=DB_PATH):
        self.db_engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.exchange = ccxt.binance()
        self._create_tables()

    def _create_tables(self):
        """
        Cria as tabelas necessárias no banco de dados, se não existirem.
        """
        create_top_cryptos_sql = """
        CREATE TABLE IF NOT EXISTS top_cryptos (
            id TEXT PRIMARY KEY,
            symbol TEXT,
            name TEXT,
            last_update DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        create_ohlcv_sql = """
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            symbol TEXT,
            timeframe TEXT,
            timestamp DATETIME,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY(symbol, timeframe, timestamp)
        );
        """
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text(create_top_cryptos_sql))
                conn.execute(text(create_ohlcv_sql))
            logger.info("Tabelas criadas/verificadas com sucesso.")
        except SQLAlchemyError as e:
            logger.error(f"Erro ao criar/verificar tabelas: {e}")

    # -------------------------------------------------------------------------
    # 1) BUSCA E SALVA TOP CRYPTOS EM CSV
    # -------------------------------------------------------------------------
    def fetch_top_cryptos(self, limit=300):
        """
        Busca os top criptoativos pelo market cap na CoinGecko
        e salva no CSV unificado (all_data.csv).
        """
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 250,
            'page': 1
        }
        crypto_list = []
        total_pages = (limit // 250) + (1 if limit % 250 != 0 else 0)

        # Paginação
        for page in range(1, total_pages + 1):
            params['page'] = page
            response = requests.get(url, params=params)
            if response.status_code == 200:
                crypto_list.extend(response.json())
            else:
                logger.error(f"Erro na API CoinGecko: {response.status_code}")
            time.sleep(1)

        # DataFrame com as colunas principais
        df = pd.DataFrame(crypto_list)[['id', 'symbol', 'name']]
        if df.empty:
            logger.warning("Nenhum dado retornado pela CoinGecko para top cryptos.")
            return

        # Para salvar TUDO em um único CSV, criamos colunas extras para padronizar com OHLCV
        df['data_type'] = 'top_cryptos'
        df['timeframe'] = None
        df['timestamp'] = None
        df['open'] = None
        df['high'] = None
        df['low'] = None
        df['close'] = None
        df['volume'] = None

        # Reordena para uma estrutura consistente
        df = df[['data_type', 'id', 'symbol', 'name',
                 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Salva/Anexa no CSV
        self._append_to_csv(df, CSV_PATH)
        logger.info(f"{len(df)} top cryptos salvos em {CSV_PATH} (modo append).")

    def _append_to_csv(self, df, csv_path):
        """
        Salva (em modo append) o DataFrame para o CSV, criando cabeçalho
        apenas se o arquivo ainda não existir.
        """
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False, mode='a', header=False)

    # -------------------------------------------------------------------------
    # 2) BUSCA E SALVA OHLCV EM CSV
    # -------------------------------------------------------------------------
    def validate_symbol(self, symbol):
        """
        Verifica se o par symbol/USDT está disponível na Binance.
        """
        try:
            markets = self.exchange.load_markets()
            pair = f"{symbol.upper()}/USDT"
            if pair in markets:
                return pair
            else:
                logger.warning(f"Par {pair} não encontrado na Binance.")
                return None
        except Exception as e:
            logger.error(f"Erro ao carregar mercados da Binance: {e}")
            return None

    def fetch_ohlcv(self, symbol, timeframe="1d", months=12):
        """
        Busca dados OHLCV na Binance e salva no CSV unificado (all_data.csv).
        (Simples: busca sempre 'months' meses atrás, sem checar incremental.)
        """
        pair = self.validate_symbol(symbol)
        if not pair:
            return

        # Data inicial (agora - months)
        dt_start = pd.Timestamp.utcnow() - pd.DateOffset(months=months)
        since = int(dt_start.timestamp() * 1000)  # em milissegundos

        all_ohlcv = []
        while True:
            try:
                batch = self.exchange.fetch_ohlcv(pair, timeframe, since=since, limit=500)
            except ccxt.NetworkError as e:
                logger.error(f"Erro de rede ao buscar OHLCV ({symbol}): {e}")
                time.sleep(5)
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Erro da exchange ao buscar OHLCV ({symbol}): {e}")
                break

            if not batch:
                break

            all_ohlcv.extend(batch)
            since = batch[-1][0] + 1  # pronto para próxima página

            # Pequena pausa para evitar rate-limit
            time.sleep(0.2)

        if not all_ohlcv:
            logger.info(f"Nenhum dado OHLCV encontrado para {symbol}.")
            return

        # Montar o DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Ajustar para formato padronizado no CSV
        df['data_type'] = 'ohlcv'
        df['id'] = None
        df['name'] = None
        df['timeframe'] = timeframe

        # Crie a coluna 'symbol' ANTES de reordenar as colunas
        df['symbol'] = symbol

        # Agora podemos reordenar
        df = df[['data_type', 'id', 'symbol', 'name', 'timeframe',
                 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Salva/Anexa no CSV
        self._append_to_csv(df, CSV_PATH)
        logger.info(f"{len(df)} registros OHLCV para {symbol} salvos em {CSV_PATH}.")

    # -------------------------------------------------------------------------
    # 3) CARREGAR O CSV NO BANCO DE DADOS
    # -------------------------------------------------------------------------
    def load_csv_into_db(self, csv_path=CSV_PATH):
        """
        Lê o CSV unificado e carrega cada tipo de dado (top_cryptos, ohlcv)
        na tabela correspondente do banco (top_cryptos e ohlcv_data).
        """
        if not os.path.exists(csv_path):
            logger.warning(f"CSV {csv_path} não existe. Nada a carregar.")
            return

        df_all = pd.read_csv(csv_path)
        if df_all.empty:
            logger.warning(f"{csv_path} está vazio. Nada a carregar.")
            return

        # Separa top_cryptos
        df_top = df_all[df_all['data_type'] == 'top_cryptos'].copy()
        if not df_top.empty:
            self._save_top_cryptos_to_db(df_top)

        # Separa OHLCV
        df_ohlcv = df_all[df_all['data_type'] == 'ohlcv'].copy()
        if not df_ohlcv.empty:
            self._save_ohlcv_to_db(df_ohlcv)

    def _save_top_cryptos_to_db(self, df_top):
        """
        Insere/atualiza registros na tabela top_cryptos.
        Espera colunas: [id, symbol, name].
        """
        try:
            with self.db_engine.connect() as conn:
                for _, row in df_top.iterrows():
                    sql = text("""
                        INSERT OR REPLACE INTO top_cryptos (id, symbol, name)
                        VALUES (:id, :symbol, :name);
                    """)
                    conn.execute(sql, {
                        'id': row['id'],
                        'symbol': row['symbol'],
                        'name': row['name']
                    })
            logger.info(f"{len(df_top)} registros de top_cryptos inseridos/atualizados no DB.")
        except SQLAlchemyError as e:
            logger.error(f"Erro ao salvar top_cryptos no DB: {e}")

    def _save_ohlcv_to_db(self, df_ohlcv):
        """
        Insere/atualiza registros na tabela ohlcv_data.
        Espera colunas: [symbol, timeframe, timestamp, open, high, low, close, volume].
        """
        if df_ohlcv.empty:
            return

        try:
            # Converter a coluna timestamp para datetime, se necessário
            if not pd.api.types.is_datetime64_any_dtype(df_ohlcv['timestamp']):
                df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'])

            with self.db_engine.begin() as conn:
                for _, row in df_ohlcv.iterrows():
                    sql = text("""
                        INSERT OR REPLACE INTO ohlcv_data
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume);
                    """)
                    conn.execute(sql, {
                        'symbol':    row['symbol'],
                        'timeframe': row['timeframe'],
                        'timestamp': row['timestamp'].isoformat(),
                        'open':      float(row['open']),
                        'high':      float(row['high']),
                        'low':       float(row['low']),
                        'close':     float(row['close']),
                        'volume':    float(row['volume'])
                    })
            logger.info(f"{len(df_ohlcv)} registros de OHLCV inseridos/atualizados no DB.")
        except SQLAlchemyError as e:
            logger.error(f"Erro ao salvar OHLCV no DB: {e}")

# =============================================================================
# SCRIPT PRINCIPAL (Exemplo de Uso)
# =============================================================================
if __name__ == "__main__":
    collector = CryptoDataCollector(db_path=DB_PATH)

    # 1) Busca Top 300 e salva no CSV
    collector.fetch_top_cryptos(limit=300)

    # 2) Exemplo: busca OHLCV para os símbolos listados em top_cryptos
    if os.path.exists(CSV_PATH):
        df_all = pd.read_csv(CSV_PATH)
        df_top = df_all[df_all['data_type'] == 'top_cryptos'].dropna(subset=['symbol'])
        unique_symbols = df_top['symbol'].unique()

        for symbol in unique_symbols[:300]:
            collector.fetch_ohlcv(symbol, timeframe="1d", months=6)
    else:
        logger.warning(f"Arquivo {CSV_PATH} não encontrado para buscar os símbolos.")

    # 3) Após tudo salvo no CSV, carrega o CSV no banco de dados
    collector.load_csv_into_db(csv_path=CSV_PATH)
