"""
RL Agent wrapper for trading
Stable-baselines3 기반 PPO/A2C 에이전트
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    PPO = None
    A2C = None

from services.rl.environment import TradingEnvironment, TradingConfig


@dataclass
class RLAgentConfig:
    """RL Agent 설정"""
    algorithm: str = "PPO"  # PPO or A2C
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    verbose: int = 1
    device: str = "auto"


class RLDecision:
    """RL 에이전트의 결정 기록"""

    def __init__(
        self,
        ts: datetime,
        symbol: str,
        action: int,
        action_name: str,
        observation: np.ndarray,
        action_probs: Optional[np.ndarray] = None,
        value: Optional[float] = None,
        model_predictions: Optional[Dict[str, float]] = None,
        position_before: float = 0.0,
        position_after: float = 0.0,
        confidence: float = 0.0,
    ):
        self.ts = ts
        self.symbol = symbol
        self.action = action
        self.action_name = action_name
        self.observation = observation
        self.action_probs = action_probs
        self.value = value
        self.model_predictions = model_predictions or {}
        self.position_before = position_before
        self.position_after = position_after
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat() if isinstance(self.ts, datetime) else self.ts,
            "symbol": self.symbol,
            "action": self.action,
            "action_name": self.action_name,
            "observation": self.observation.tolist() if isinstance(self.observation, np.ndarray) else self.observation,
            "action_probs": self.action_probs.tolist() if self.action_probs is not None else None,
            "value": self.value,
            "model_predictions": self.model_predictions,
            "position_before": self.position_before,
            "position_after": self.position_after,
            "confidence": self.confidence,
        }


class TrainingCallback(BaseCallback):
    """학습 중 로깅 콜백"""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # 환경에서 정보 가져오기
            env = self.training_env.envs[0]
            if hasattr(env, 'total_pnl'):
                print(f"Step {self.n_calls}: PnL={env.total_pnl:.2f}, "
                      f"Trades={env.total_trades}, "
                      f"Win Rate={env.winning_trades/max(1, env.total_trades)*100:.1f}%")
        return True


class RLAgent:
    """
    RL 에이전트 래퍼 클래스

    기존 모델 예측값을 입력받아 거래 결정을 내림
    """

    ACTION_NAMES = ["HOLD", "LONG", "SHORT", "CLOSE"]

    def __init__(
        self,
        config: Optional[RLAgentConfig] = None,
        trading_config: Optional[TradingConfig] = None,
    ):
        if not HAS_SB3:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            )

        self.config = config or RLAgentConfig()
        self.trading_config = trading_config or TradingConfig()
        self.model = None
        self.env = None
        self.decisions: List[RLDecision] = []

    def create_env(self, data: pd.DataFrame) -> TradingEnvironment:
        """환경 생성"""
        self.env = TradingEnvironment(data, self.trading_config)
        return self.env

    def train(
        self,
        data: pd.DataFrame,
        total_timesteps: int = 100000,
        eval_data: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """에이전트 학습"""

        # 환경 생성
        env = DummyVecEnv([lambda: TradingEnvironment(data, self.trading_config)])

        # 알고리즘 선택
        AlgoClass = PPO if self.config.algorithm == "PPO" else A2C

        # 모델 생성
        self.model = AlgoClass(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size if self.config.algorithm == "PPO" else None,
            n_epochs=self.config.n_epochs if self.config.algorithm == "PPO" else None,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range if self.config.algorithm == "PPO" else None,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            verbose=self.config.verbose,
            device=self.config.device,
        )

        # 콜백 설정
        callbacks = [TrainingCallback(log_freq=5000)]

        if eval_data is not None:
            eval_env = DummyVecEnv([lambda: TradingEnvironment(eval_data, self.trading_config)])
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path if save_path else "./rl_models/",
                log_path="./rl_logs/",
                eval_freq=10000,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        # 학습
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)

        # 모델 저장
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            self.model.save(save_path)

        # 최종 평가
        final_env = TradingEnvironment(data, self.trading_config)
        metrics = self.evaluate(final_env)

        return {
            "total_timesteps": total_timesteps,
            "algorithm": self.config.algorithm,
            "metrics": metrics,
        }

    def load(self, path: str) -> None:
        """저장된 모델 로드"""
        AlgoClass = PPO if self.config.algorithm == "PPO" else A2C
        self.model = AlgoClass.load(path)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[np.ndarray], Optional[float]]:
        """
        단일 관측값에 대한 예측

        Returns:
            action: 선택된 액션 (0-3)
            action_probs: 각 액션의 확률 (softmax)
            value: 상태 가치 추정
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call train() or load() first.")

        # 배치 차원 추가
        obs = observation.reshape(1, -1)

        # 액션 예측
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action = int(action[0])

        # 액션 확률 및 가치 계산
        action_probs = None
        value = None

        try:
            # 정책에서 분포 가져오기
            obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
            distribution = self.model.policy.get_distribution(obs_tensor)
            action_probs = distribution.distribution.probs.detach().cpu().numpy()[0]

            # 가치 함수
            value = self.model.policy.predict_values(obs_tensor).detach().cpu().numpy()[0][0]
        except Exception:
            pass

        return action, action_probs, value

    def decide(
        self,
        ts: datetime,
        symbol: str,
        model_predictions: Dict[str, float],
        market_data: Dict[str, float],
        current_position: float = 0.0,
        position_time: int = 0,
        unrealized_pnl: float = 0.0,
        deterministic: bool = True,
    ) -> RLDecision:
        """
        실시간 거래 결정

        Args:
            ts: 현재 시간
            symbol: 심볼
            model_predictions: 모델 예측값 (er_long, q05_long, e_mae_long, e_hold_long)
            market_data: 시장 데이터 (ret_1, ret_5, rsi, macd_z, bb_z, vol_z, atr, funding_z, btc_regime, adx)
            current_position: 현재 포지션 (-1 ~ 1)
            position_time: 포지션 보유 시간 (분)
            unrealized_pnl: 미실현 손익
            deterministic: 결정론적 행동 여부

        Returns:
            RLDecision 객체
        """
        # 관측값 구성
        model_features = [
            model_predictions.get("er_long", 0) or 0,
            model_predictions.get("q05_long", 0) or 0,
            model_predictions.get("e_mae_long", 0) or 0,
            model_predictions.get("e_hold_long", 0) or 0,
        ]

        market_features = [
            market_data.get("ret_1", 0) or 0,
            market_data.get("ret_5", 0) or 0,
            (market_data.get("rsi", 50) or 50) / 100 - 0.5,
            market_data.get("macd_z", 0) or 0,
            market_data.get("bb_z", 0) or 0,
            market_data.get("vol_z", 0) or 0,
            market_data.get("atr", 0) or 0,
            market_data.get("funding_z", 0) or 0,
            market_data.get("btc_regime", 0) or 0,
            (market_data.get("adx", 25) or 25) / 50 - 0.5,
        ]

        position_features = [
            current_position,
            min(position_time / 360, 1.0),
            unrealized_pnl / self.trading_config.initial_balance,
        ]

        observation = np.array(
            model_features + market_features + position_features,
            dtype=np.float32
        )
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        # 예측
        action, action_probs, value = self.predict(observation, deterministic)

        # 포지션 변화 계산
        position_after = current_position
        if action == 1:  # LONG
            position_after = 1.0
        elif action == 2:  # SHORT
            position_after = -1.0
        elif action == 3:  # CLOSE
            position_after = 0.0

        # 신뢰도 계산
        confidence = 0.0
        if action_probs is not None:
            confidence = float(action_probs[action])

        # 결정 기록
        decision = RLDecision(
            ts=ts,
            symbol=symbol,
            action=action,
            action_name=self.ACTION_NAMES[action],
            observation=observation,
            action_probs=action_probs,
            value=value,
            model_predictions=model_predictions,
            position_before=current_position,
            position_after=position_after,
            confidence=confidence,
        )

        self.decisions.append(decision)
        return decision

    def evaluate(self, env: TradingEnvironment) -> Dict[str, Any]:
        """환경에서 에이전트 평가"""
        if self.model is None:
            raise ValueError("Model not loaded.")

        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _, _ = self.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # 거래 기록 분석
        history_df = env.get_history_df()
        trades = history_df[history_df["trade_info"].notna()]

        return {
            "total_reward": total_reward,
            "final_balance": env.balance,
            "total_pnl": env.total_pnl,
            "total_trades": env.total_trades,
            "winning_trades": env.winning_trades,
            "win_rate": env.winning_trades / max(1, env.total_trades),
            "profit_factor": self._calculate_profit_factor(history_df),
            "sharpe_ratio": self._calculate_sharpe(history_df),
            "max_drawdown": self._calculate_max_drawdown(history_df),
            "action_distribution": self._get_action_distribution(history_df),
        }

    def _calculate_profit_factor(self, history: pd.DataFrame) -> float:
        """Profit Factor 계산"""
        rewards = history["reward"]
        gains = rewards[rewards > 0].sum()
        losses = abs(rewards[rewards < 0].sum())
        return gains / max(losses, 1e-10)

    def _calculate_sharpe(self, history: pd.DataFrame, periods_per_year: int = 525600) -> float:
        """Sharpe Ratio 계산 (분봉 기준)"""
        rewards = history["reward"]
        if len(rewards) < 2:
            return 0.0
        mean_return = rewards.mean()
        std_return = rewards.std()
        if std_return == 0:
            return 0.0
        return mean_return / std_return * np.sqrt(periods_per_year)

    def _calculate_max_drawdown(self, history: pd.DataFrame) -> float:
        """최대 낙폭 계산"""
        balance = history["balance"]
        peak = balance.expanding().max()
        drawdown = (balance - peak) / peak
        return abs(drawdown.min())

    def _get_action_distribution(self, history: pd.DataFrame) -> Dict[str, int]:
        """액션 분포"""
        return history["action"].value_counts().to_dict()

    def get_decisions_df(self) -> pd.DataFrame:
        """결정 기록을 DataFrame으로 반환"""
        if not self.decisions:
            return pd.DataFrame()
        return pd.DataFrame([d.to_dict() for d in self.decisions])

    def clear_decisions(self) -> None:
        """결정 기록 초기화"""
        self.decisions = []
