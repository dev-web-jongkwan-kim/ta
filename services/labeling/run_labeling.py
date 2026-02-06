from __future__ import annotations

import argparse
import logging
import sys

from services.labeling.pipeline import LabelingConfig, run_labeling

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨링 파이프라인 실행")
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="전체 재라벨링 (기본값: 증분 라벨링)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="처리할 심볼 목록 (기본값: 전체 universe)",
    )
    parser.add_argument(
        "--k-tp",
        type=float,
        default=1.5,
        help="TP 배수 (기본값: 1.5)",
    )
    parser.add_argument(
        "--k-sl",
        type=float,
        default=1.0,
        help="SL 배수 (기본값: 1.0)",
    )
    parser.add_argument(
        "--h-bars",
        type=int,
        default=360,
        help="최대 보유 시간 (분) (기본값: 360)",
    )
    parser.add_argument(
        "--risk-mae-atr",
        type=float,
        default=3.0,
        help="리스크 MAE ATR 배수 (기본값: 3.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 로그 출력",
    )
    parser.add_argument(
        "--timeframe",
        "-t",
        type=str,
        default="1m",
        choices=["1m", "15m", "1h"],
        help="타임프레임 (기본값: 1m)",
    )
    parser.add_argument(
        "--atr-timeframe",
        type=str,
        default=None,
        choices=["1m", "15m", "1h"],
        help="ATR 계산용 타임프레임 (기본값: timeframe과 동일). 예: --timeframe 1m --atr-timeframe 15m",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # h_bars 조정: 15m이면 h_bars=24 (6시간), 1h이면 h_bars=6 (6시간)
    h_bars = args.h_bars
    if args.timeframe == "15m" and args.h_bars == 360:
        h_bars = 24  # 15분 × 24 = 6시간
        logger.info(f"h_bars adjusted for 15m: {args.h_bars} -> {h_bars}")
    elif args.timeframe == "1h" and args.h_bars == 360:
        h_bars = 6  # 1시간 × 6 = 6시간
        logger.info(f"h_bars adjusted for 1h: {args.h_bars} -> {h_bars}")

    config = LabelingConfig(
        k_tp=args.k_tp,
        k_sl=args.k_sl,
        h_bars=h_bars,
        risk_mae_atr=args.risk_mae_atr,
        timeframe=args.timeframe,
        atr_timeframe=args.atr_timeframe,
    )

    atr_tf_info = f", atr_timeframe={config.atr_timeframe}" if config.atr_timeframe else ""
    logger.info(f"Starting labeling (force_full={args.force_full}, timeframe={config.timeframe}{atr_tf_info})")
    logger.info(f"Config: k_tp={config.k_tp}, k_sl={config.k_sl}, h_bars={config.h_bars}, risk_mae_atr={config.risk_mae_atr}, timeframe={config.timeframe}, atr_timeframe={config.atr_timeframe}")

    stats = run_labeling(
        config=config,
        symbols=args.symbols,
        force_full=args.force_full,
    )

    # 결과 요약 출력
    logger.info("=" * 50)
    logger.info(f"Spec hash: {stats.spec_hash}")
    logger.info(f"Symbols processed: {stats.symbols_processed}")
    logger.info(f"New labels: {stats.total_new_labels:,}")
    logger.info(f"Existing labels: {stats.total_existing:,}")

    if args.verbose and stats.by_symbol:
        logger.info("-" * 50)
        logger.info("By symbol:")
        for symbol, result in stats.by_symbol.items():
            if "error" in result:
                logger.info(f"  {symbol}: ERROR - {result['error']}")
            else:
                new_total = result["new_long"] + result["new_short"]
                existing_total = result["existing_long"] + result["existing_short"]
                logger.info(f"  {symbol}: new={new_total:,}, existing={existing_total:,}")


if __name__ == "__main__":
    main()
