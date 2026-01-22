# -*- coding: utf-8 -*-
"""
===================================
AkshareFetcher - 主数据源 (Priority 1)
===================================
数据来源：东方财富爬虫（通过 akshare 库） + Tushare fallback
特点：免费、无需 Token、数据全面
风险：爬虫机制易被反爬封禁
防封禁策略：
1. 每次请求前随机休眠 5-30 秒（增强）
2. 随机轮换 User-Agent
3. 使用 tenacity 实现指数退避重试（3次）
增强数据：
- 实时行情：量比、换手率、市盈率、市净率、总市值、流通市值
- 筹码分布：获利比例、平均成本、筹码集中度
"""
import logging
import random
import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import tushare as ts
from .base import BaseFetcher, DataFetchError, RateLimitError, STANDARD_COLUMNS
from config import get_config  # 假设你有 config.py 读取 TUSHARE_TOKEN

logger = logging.getLogger(__name__)

# User-Agent 池
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

# 缓存实时行情数据（延长 TTL 到 5 分钟）
_realtime_cache: Dict[str, Any] = {
    'data': None,
    'timestamp': 0,
    'ttl': 300  # 300秒 = 5分钟
}

# ETF 实时行情缓存
_etf_realtime_cache: Dict[str, Any] = {
    'data': None,
    'timestamp': 0,
    'ttl': 300
}

def _is_etf_code(stock_code: str) -> bool:
    etf_prefixes = ('51', '52', '56', '58', '15', '16', '18')
    return stock_code.startswith(etf_prefixes) and len(stock_code) == 6

def _is_hk_code(stock_code: str) -> bool:
    code = stock_code.lower()
    if code.startswith('hk'):
        numeric_part = code[2:]
        return numeric_part.isdigit() and 1 <= len(numeric_part) <= 5
    return code.isdigit() and len(code) == 5

@dataclass
class RealtimeQuote:
    code: str
    name: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    change_amount: float = 0.0
    volume_ratio: float = 0.0
    turnover_rate: float = 0.0
    amplitude: float = 0.0
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    total_mv: float = 0.0
    circ_mv: float = 0.0
    change_60d: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'price': self.price,
            'change_pct': self.change_pct,
            'volume_ratio': self.volume_ratio,
            'turnover_rate': self.turnover_rate,
            'amplitude': self.amplitude,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'total_mv': self.total_mv,
            'circ_mv': self.circ_mv,
            'change_60d': self.change_60d,
        }

@dataclass
class ChipDistribution:
    code: str
    date: str = ""
    profit_ratio: float = 0.0
    avg_cost: float = 0.0
    cost_90_low: float = 0.0
    cost_90_high: float = 0.0
    concentration_90: float = 0.0
    cost_70_low: float = 0.0
    cost_70_high: float = 0.0
    concentration_70: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'date': self.date,
            'profit_ratio': self.profit_ratio,
            'avg_cost': self.avg_cost,
            'cost_90_low': self.cost_90_low,
            'cost_90_high': self.cost_90_high,
            'concentration_90': self.concentration_90,
            'concentration_70': self.concentration_70,
        }

    def get_chip_status(self, current_price: float) -> str:
        status_parts = []
        if self.profit_ratio >= 0.9:
            status_parts.append("获利盘极高(>90%)")
        elif self.profit_ratio >= 0.7:
            status_parts.append("获利盘较高(70-90%)")
        # ... 其他逻辑保持不变 ...
        return "，".join(status_parts)

class AkshareFetcher(BaseFetcher):
    name = "AkshareFetcher"
    priority = 1

    def __init__(self, sleep_min: float = 5.0, sleep_max: float = 30.0):
        self.sleep_min = sleep_min
        self.sleep_max = sleep_max
        self._last_request_time: Optional[float] = None

    def _set_random_user_agent(self) -> None:
        random_ua = random.choice(USER_AGENTS)
        logger.debug(f"设置随机 User-Agent: {random_ua[:50]}...")

    def _enforce_rate_limit(self) -> None:
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.sleep_min:
                additional = self.sleep_min - elapsed
                logger.debug(f"补充休眠 {additional:.2f} 秒")
                time.sleep(additional)
        # 随机延时
        jitter = random.uniform(self.sleep_min, self.sleep_max)
        logger.debug(f"执行随机延时 {jitter:.2f} 秒")
        time.sleep(jitter)
        self._last_request_time = time.time()

    def _safe_float(self, val, default=0.0):
        try:
            if pd.isna(val):
                return default
            return float(val)
        except:
            return default

    # 历史数据部分保持不变（略）

    def get_realtime_quote(self, stock_code: str) -> Optional[RealtimeQuote]:
        if _is_hk_code(stock_code):
            return self._get_hk_realtime_quote(stock_code)
        elif _is_etf_code(stock_code):
            return self._get_etf_realtime_quote(stock_code)
        else:
            return self._get_stock_realtime_quote(stock_code)

    def _get_stock_realtime_quote(self, stock_code: str) -> Optional[RealtimeQuote]:
        import akshare as ak

        config = get_config()
        tushare_token = config.tushare_token or os.getenv("TUSHARE_TOKEN")

        df = None
        source = "AKShare"

        # 优先尝试 AKShare（东方财富）
        try:
            self._enforce_rate_limit()
            self._set_random_user_agent()
            logger.info(f"[API调用] ak.stock_zh_a_spot_em() 获取A股实时行情... (来源: {source})")
            df = ak.stock_zh_a_spot_em()
            logger.info(f"[API返回] ak.stock_zh_a_spot_em 成功: 返回 {len(df)} 只股票")
        except Exception as e:
            logger.warning(f"[AKShare 失败] {e}")

        # 如果失败，且有 Tushare Token，则 fallback 到 Tushare
        if (df is None or df.empty) and tushare_token:
            source = "Tushare (fallback)"
            try:
                logger.info(f"[Fallback] 切换到 Tushare 获取实时行情")
                pro = ts.pro_api(tushare_token)
                # Tushare daily_basic 提供最新交易日的估值和换手率
                # ts_code 需要加 .SH / .SZ 后缀
                suffix = '.SH' if stock_code.startswith('6') else '.SZ'
                df_ts = pro.daily_basic(
                    ts_code=stock_code + suffix,
                    fields='ts_code,trade_date,close,change,pct_chg,vol,amount,turnover_rate,pe,pb,total_mv,circ_mv'
                )
                if not df_ts.empty:
                    # 转换为与 AKShare 类似的结构
                    df = pd.DataFrame([{
                        '代码': stock_code,
                        '名称': stock_code,  # Tushare 无名称，可后续补充
                        '最新价': df_ts.iloc[0]['close'],
                        '涨跌幅': df_ts.iloc[0]['pct_chg'],
                        '涨跌额': df_ts.iloc[0]['change'],
                        '量比': 0.0,  # Tushare daily_basic 无量比，设默认
                        '换手率': df_ts.iloc[0]['turnover_rate'],
                        '振幅': 0.0,  # 无振幅
                        '市盈率-动态': df_ts.iloc[0]['pe'],
                        '市净率': df_ts.iloc[0]['pb'],
                        '总市值': df_ts.iloc[0]['total_mv'],
                        '流通市值': df_ts.iloc[0]['circ_mv'],
                    }])
                    logger.info(f"[Tushare] 实时行情成功: {stock_code}")
                else:
                    logger.warning(f"[Tushare] 返回空数据: {stock_code}")
            except Exception as e:
                logger.error(f"[Tushare fallback 失败] {e}")

        # 如果两种源都失败
        if df is None or df.empty:
            logger.warning(f"[实时行情] 最终为空，跳过 {stock_code} (来源尝试: AKShare + Tushare)")
            return None

        # 查找指定股票
        row = df[df['代码'] == stock_code]
        if row.empty:
            logger.warning(f"[API返回] 未找到股票 {stock_code} 的实时行情")
            return None

        row = row.iloc[0]

        quote = RealtimeQuote(
            code=stock_code,
            name=str(row.get('名称', '')),
            price=self._safe_float(row.get('最新价')),
            change_pct=self._safe_float(row.get('涨跌幅')),
            change_amount=self._safe_float(row.get('涨跌额')),
            volume_ratio=self._safe_float(row.get('量比')),
            turnover_rate=self._safe_float(row.get('换手率')),
            amplitude=self._safe_float(row.get('振幅')),
            pe_ratio=self._safe_float(row.get('市盈率-动态')),
            pb_ratio=self._safe_float(row.get('市净率')),
            total_mv=self._safe_float(row.get('总市值')),
            circ_mv=self._safe_float(row.get('流通市值')),
            change_60d=self._safe_float(row.get('60日涨跌幅', 0.0)),
            high_52w=self._safe_float(row.get('52周最高', 0.0)),
            low_52w=self._safe_float(row.get('52周最低', 0.0)),
        )

        logger.info(f"[{source} 实时行情] {stock_code} {quote.name}: 价格={quote.price}, "
                    f"涨跌={quote.change_pct}%, 量比={quote.volume_ratio}, 换手率={quote.turnover_rate}%, "
                    f"PE={quote.pe_ratio}, PB={quote.pb_ratio}")
        return quote

    # 其他方法（历史数据、ETF、港股、筹码分布）保持不变
    # ...（省略未修改部分）...

    def get_enhanced_data(self, stock_code: str, days: int = 60) -> Dict[str, Any]:
        result = {
            'code': stock_code,
            'daily_data': None,
            'realtime_quote': None,
            'chip_distribution': None,
        }
        try:
            result['daily_data'] = self.get_daily_data(stock_code, days=days)
        except Exception as e:
            logger.error(f"获取 {stock_code} 日线数据失败: {e}")
        result['realtime_quote'] = self.get_realtime_quote(stock_code)
        result['chip_distribution'] = self.get_chip_distribution(stock_code)
        return result
