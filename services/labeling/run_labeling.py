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
    args = parser.parse_args()

    setup_logging(args.verbose)

    config = LabelingConfig(
        k_tp=args.k_tp,
        k_sl=args.k_sl,
        h_bars=args.h_bars,
        risk_mae_atr=args.risk_mae_atr,
    )

    logger.info(f"Starting labeling (force_full={args.force_full})")
    logger.info(f"Config: k_tp={config.k_tp}, k_sl={config.k_sl}, h_bars={config.h_bars}, risk_mae_atr={config.risk_mae_atr}")

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
