"""
RL 에이전트 학습 스크립트
기존 모델 예측값 기반으로 거래 결정 학습
"""
import argparse
import gc
import os
import sys
import uuid
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.rl import RLAgent, RLAgentConfig, TradingConfig


def get_db_engine():
    """DB 연결"""
    return create_engine(os.getenv("DATABASE_URL"))


def load_training_data(
    engine,
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """학습 데이터 로드 (피처 + 모델 예측값)"""

    print(f"Loading data for {len(symbols)} symbols: {start_date} ~ {end_date}")

    all_data = []

    for symbol in symbols:
        print(f"  Loading {symbol}...")

        # 피처 데이터
        query = text("""
            SELECT
                f.ts,
                f.symbol,
                f.close,
                f.ret_1,
                f.ret_5,
                f.rsi,
                f.macd_z,
                f.bb_z,
                f.vol_z,
                f.atr,
                f.adx,
                s.er_long,
                s.q05_long,
                s.e_mae_long,
                s.e_hold_long,
                s.funding_z,
                s.btc_regime
            FROM features_1m f
            LEFT JOIN signals s ON f.symbol = s.symbol AND f.ts = s.ts
            WHERE f.symbol = :symbol
              AND f.ts >= :start_date
              AND f.ts < :end_date
            ORDER BY f.ts
        """)

        df = pd.read_sql(
            query,
            engine,
            params={"symbol": symbol, "start_date": start_date, "end_date": end_date}
        )

        if len(df) > 0:
            all_data.append(df)
            print(f"    Loaded {len(df):,} rows")

    if not all_data:
        raise ValueError("No data loaded")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Total: {len(combined):,} rows")

    # NaN 처리
    combined = combined.fillna({
        "er_long": 0,
        "q05_long": 0,
        "e_mae_long": 0,
        "e_hold_long": 0,
        "funding_z": 0,
        "btc_regime": 0,
        "ret_1": 0,
        "ret_5": 0,
        "rsi": 50,
        "macd_z": 0,
        "bb_z": 0,
        "vol_z": 0,
        "atr": 0,
        "adx": 25,
    })

    return combined


def train_rl_agent(
    symbols: list[str],
    train_days: int = 90,
    eval_days: int = 30,
    total_timesteps: int = 100000,
    algorithm: str = "PPO",
    save_dir: str = "./rl_models",
):
    """RL 에이전트 학습"""

    engine = get_db_engine()

    # 날짜 범위 설정
    end_date = datetime.now()
    eval_start = end_date - timedelta(days=eval_days)
    train_start = eval_start - timedelta(days=train_days)

    print(f"\n{'='*60}")
    print(f"RL Agent Training")
    print(f"{'='*60}")
    print(f"Algorithm: {algorithm}")
    print(f"Symbols: {symbols}")
    print(f"Train period: {train_start.date()} ~ {eval_start.date()} ({train_days} days)")
    print(f"Eval period: {eval_start.date()} ~ {end_date.date()} ({eval_days} days)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"{'='*60}\n")

    # 심볼별로 학습
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Training for {symbol}")
        print(f"{'='*60}")

        try:
            # 학습 데이터 로드
            train_data = load_training_data(
                engine,
                [symbol],
                train_start.strftime("%Y-%m-%d"),
                eval_start.strftime("%Y-%m-%d"),
            )

            if len(train_data) < 1000:
                print(f"  Insufficient data for {symbol} ({len(train_data)} rows). Skipping.")
                continue

            # 평가 데이터 로드
            eval_data = load_training_data(
                engine,
                [symbol],
                eval_start.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )

            # 설정
            rl_config = RLAgentConfig(
                algorithm=algorithm,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=1,
            )

            trading_config = TradingConfig(
                initial_balance=10000.0,
                trading_fee=0.0004,
                slippage=0.0001,
                leverage=3,
                lookback_window=60,
            )

            # 에이전트 생성 및 학습
            agent = RLAgent(config=rl_config, trading_config=trading_config)

            model_id = str(uuid.uuid4())[:8]
            save_path = os.path.join(save_dir, f"rl_{symbol}_{model_id}")
            os.makedirs(save_dir, exist_ok=True)

            result = agent.train(
                data=train_data,
                total_timesteps=total_timesteps,
                eval_data=eval_data if len(eval_data) > 100 else None,
                save_path=save_path,
            )

            # 결과 출력
            metrics = result["metrics"]
            print(f"\n{'='*40}")
            print(f"Training Results for {symbol}")
            print(f"{'='*40}")
            print(f"Final Balance: ${metrics['final_balance']:.2f}")
            print(f"Total PnL: ${metrics['total_pnl']:.2f}")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
            print(f"Action Distribution: {metrics['action_distribution']}")
            print(f"Model saved to: {save_path}")

            # DB에 모델 정보 저장
            save_rl_model_to_db(
                engine,
                model_id=model_id,
                symbol=symbol,
                algorithm=algorithm,
                train_start=train_start,
                train_end=eval_start,
                metrics=metrics,
                save_path=save_path,
            )

            # 메모리 정리
            del agent, train_data, eval_data
            gc.collect()

        except Exception as e:
            print(f"Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")


def save_rl_model_to_db(
    engine,
    model_id: str,
    symbol: str,
    algorithm: str,
    train_start: datetime,
    train_end: datetime,
    metrics: dict,
    save_path: str,
):
    """RL 모델 정보를 DB에 저장"""
    import json

    query = text("""
        INSERT INTO rl_models (
            model_id, symbol, algorithm, train_start, train_end,
            metrics, model_path, is_production, created_at
        ) VALUES (
            :model_id, :symbol, :algorithm, :train_start, :train_end,
            :metrics, :model_path, false, NOW()
        )
        ON CONFLICT (model_id) DO UPDATE SET
            metrics = :metrics,
            model_path = :model_path
    """)

    with engine.begin() as conn:
        conn.execute(query, {
            "model_id": model_id,
            "symbol": symbol,
            "algorithm": algorithm,
            "train_start": train_start,
            "train_end": train_end,
            "metrics": json.dumps(metrics, default=str),
            "model_path": save_path,
        })

    print(f"  Saved model {model_id} to database")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT",
        help="Comma-separated symbols to train on",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=90,
        help="Number of days for training",
    )
    parser.add_argument(
        "--eval-days",
        type=int,
        default=30,
        help="Number of days for evaluation",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "A2C"],
        help="RL algorithm to use",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./rl_models",
        help="Directory to save models",
    )

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    train_rl_agent(
        symbols=symbols,
        train_days=args.train_days,
        eval_days=args.eval_days,
        total_timesteps=args.timesteps,
        algorithm=args.algorithm,
        save_dir=args.save_dir,
    )
