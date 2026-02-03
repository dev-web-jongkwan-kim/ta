"""
트레이딩 강화학습 환경
Gymnasium 인터페이스 구현
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class TradingConfig:
    """트레이딩 환경 설정"""
    initial_balance: float = 10000.0
    max_position_size: float = 1.0  # 최대 포지션 (자본 대비)
    trading_fee: float = 0.0004  # 0.04%
    slippage: float = 0.0001  # 0.01%
    leverage: int = 3
    lookback_window: int = 60  # 과거 60분 참조


class TradingEnvironment(gym.Env):
    """
    트레이딩 강화학습 환경

    State: 모델 예측값 + 시장 상태 + 포지션 상태
    Action: 0=Hold, 1=Long, 2=Short, 3=Close
    Reward: 실현 손익 - 거래비용
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[TradingConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.config = config or TradingConfig()
        self.render_mode = render_mode

        # 데이터 준비
        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data)

        # State space: 모델 예측(4) + 시장지표(10) + 포지션상태(3) = 17차원
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )

        # Action space: Hold(0), Long(1), Short(2), Close(3)
        self.action_space = spaces.Discrete(4)

        # 상태 초기화
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # 시작 위치 (lookback 이후부터)
        self.current_step = self.config.lookback_window

        # 계좌 상태
        self.balance = self.config.initial_balance
        self.position = 0.0  # -1 (short) ~ +1 (long)
        self.position_price = 0.0
        self.position_time = 0

        # 기록
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.history = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """현재 상태 관측"""
        row = self.data.iloc[self.current_step]

        # 모델 예측값 (4개)
        model_features = [
            row.get("er_long", 0) or 0,
            row.get("q05_long", 0) or 0,
            row.get("e_mae_long", 0) or 0,
            row.get("e_hold_long", 0) or 0,
        ]

        # 시장 지표 (10개)
        market_features = [
            row.get("ret_1", 0) or 0,
            row.get("ret_5", 0) or 0,
            row.get("rsi", 50) / 100 - 0.5,  # -0.5 ~ 0.5로 정규화
            row.get("macd_z", 0) or 0,
            row.get("bb_z", 0) or 0,
            row.get("vol_z", 0) or 0,
            row.get("atr", 0) or 0,
            row.get("funding_z", 0) or 0,
            row.get("btc_regime", 0) or 0,
            row.get("adx", 25) / 50 - 0.5,  # 정규화
        ]

        # 포지션 상태 (3개)
        unrealized_pnl = self._calculate_unrealized_pnl()
        position_features = [
            self.position,  # -1 ~ 1
            min(self.position_time / 360, 1.0),  # 보유시간 (최대 360분 기준)
            unrealized_pnl / self.config.initial_balance,  # 미실현 손익률
        ]

        obs = np.array(model_features + market_features + position_features, dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _calculate_unrealized_pnl(self) -> float:
        """미실현 손익 계산"""
        if self.position == 0:
            return 0.0

        current_price = self.data.iloc[self.current_step]["close"]
        price_change = (current_price - self.position_price) / self.position_price

        return self.position * price_change * self.balance * self.config.leverage

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행"""
        current_price = self.data.iloc[self.current_step]["close"]
        reward = 0.0
        trade_info = None

        # 액션 실행
        if action == 1 and self.position <= 0:  # Long
            # 기존 숏 포지션 청산
            if self.position < 0:
                reward += self._close_position(current_price)
            # 롱 진입
            self._open_position(1.0, current_price)
            trade_info = {"action": "LONG", "price": current_price}

        elif action == 2 and self.position >= 0:  # Short
            # 기존 롱 포지션 청산
            if self.position > 0:
                reward += self._close_position(current_price)
            # 숏 진입
            self._open_position(-1.0, current_price)
            trade_info = {"action": "SHORT", "price": current_price}

        elif action == 3 and self.position != 0:  # Close
            reward += self._close_position(current_price)
            trade_info = {"action": "CLOSE", "price": current_price}

        # 포지션 보유 시간 증가
        if self.position != 0:
            self.position_time += 1
            # 펀딩비용 (8시간마다)
            if self.position_time % 480 == 0:
                funding_rate = self.data.iloc[self.current_step].get("funding_rate", 0) or 0
                reward -= abs(self.position) * funding_rate * self.balance

        # 기록 저장
        self.history.append({
            "step": self.current_step,
            "ts": self.data.iloc[self.current_step].get("ts"),
            "symbol": self.data.iloc[self.current_step].get("symbol"),
            "action": ["HOLD", "LONG", "SHORT", "CLOSE"][action],
            "position": self.position,
            "balance": self.balance,
            "price": current_price,
            "reward": reward,
            "unrealized_pnl": self._calculate_unrealized_pnl(),
            "trade_info": trade_info,
        })

        # 다음 스텝
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = self.balance <= 0  # 파산

        # 최종 청산 (에피소드 종료 시)
        if terminated and self.position != 0:
            reward += self._close_position(current_price)

        obs = self._get_observation() if not terminated else np.zeros(17, dtype=np.float32)

        info = {
            "balance": self.balance,
            "position": self.position,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": self.total_pnl,
        }

        return obs, reward, terminated, truncated, info

    def _open_position(self, direction: float, price: float) -> None:
        """포지션 오픈"""
        self.position = direction
        self.position_price = price
        self.position_time = 0

        # 거래 비용
        cost = abs(direction) * self.balance * (self.config.trading_fee + self.config.slippage)
        self.balance -= cost

    def _close_position(self, price: float) -> float:
        """포지션 청산"""
        if self.position == 0:
            return 0.0

        # 손익 계산
        price_change = (price - self.position_price) / self.position_price
        pnl = self.position * price_change * self.balance * self.config.leverage

        # 거래 비용
        cost = abs(self.position) * self.balance * (self.config.trading_fee + self.config.slippage)
        net_pnl = pnl - cost

        # 계좌 반영
        self.balance += net_pnl
        self.total_pnl += net_pnl
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1

        # 포지션 초기화
        self.position = 0.0
        self.position_price = 0.0
        self.position_time = 0

        return net_pnl

    def render(self) -> None:
        if self.render_mode == "human":
            print(f"Step {self.current_step}: Balance={self.balance:.2f}, Position={self.position:.2f}")

    def get_history_df(self) -> pd.DataFrame:
        """거래 기록을 DataFrame으로 반환"""
        return pd.DataFrame(self.history)
