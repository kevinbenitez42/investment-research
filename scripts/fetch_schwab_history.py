"""Fetch as much Schwab transaction and order history as the API allows.

Raw responses are written to ``csv_files/schwab_api_raw/<timestamp>/``. That
directory is ignored by git because these files can contain account details.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from schwab.auth import easy_client


REQUIRED_ENV_VARS = [
    "SCHWAB_CLIENT_ID",
    "SCHWAB_APP_SECRET",
    "SCHWAB_CALLBACK_URL",
    "SCHWAB_TOKEN_PATH",
]


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _account_entries(account_numbers: Any) -> list[dict[str, Any]]:
    if isinstance(account_numbers, list):
        entries = account_numbers
    elif isinstance(account_numbers, dict):
        entries = account_numbers.get("accounts") or account_numbers.get("accountNumbers") or []
    else:
        entries = []
    return [entry for entry in entries if isinstance(entry, dict)]


def _stable_key(item: Any, fallback_prefix: str, fallback_idx: int) -> str:
    if isinstance(item, dict):
        for key in ("activityId", "transactionId", "orderId", "enteredTime", "time"):
            value = item.get(key)
            if value is not None:
                return str(value)
        return json.dumps(item, sort_keys=True, default=str)
    return f"{fallback_prefix}_{fallback_idx}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_client():
    missing = [key for key in REQUIRED_ENV_VARS if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    token_path = Path(os.environ["SCHWAB_TOKEN_PATH"]).expanduser()
    if not token_path.exists():
        raise RuntimeError("Configured SCHWAB_TOKEN_PATH does not exist.")

    return easy_client(
        api_key=os.environ["SCHWAB_CLIENT_ID"],
        app_secret=os.environ["SCHWAB_APP_SECRET"],
        callback_url=os.environ["SCHWAB_CALLBACK_URL"],
        token_path=str(token_path),
        interactive=False,
    )


def _call_with_retries(callable_obj, *, retries: int, label: str):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return callable_obj(), None
        except Exception as exc:  # noqa: BLE001 - preserve API failure details in manifest.
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    return None, f"{label}: {type(last_error).__name__}: {last_error}"


def fetch_history(
    *,
    project_root: Path,
    lookback_days: int,
    chunk_days: int,
    max_order_results: int,
    include_orders: bool,
    request_timeout: float,
    retries: int,
    stop_after_empty_windows: int,
    progress: bool,
) -> dict[str, Any]:
    _load_env_file(project_root / ".env")
    client = _build_client()
    if hasattr(client, "set_timeout"):
        client.set_timeout(request_timeout)

    numbers_response = client.get_account_numbers()
    if numbers_response.status_code >= 400:
        numbers_response.raise_for_status()

    account_numbers = numbers_response.json()
    accounts = _account_entries(account_numbers)

    now_utc = datetime.now(timezone.utc)
    target_start = now_utc - timedelta(days=lookback_days)
    run_stamp = now_utc.strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / "csv_files" / "schwab_api_raw" / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "account_numbers.json", account_numbers)

    manifest: dict[str, Any] = {
        "retrieved_at_utc": now_utc.isoformat(),
        "target_start_utc": target_start.isoformat(),
        "target_end_utc": now_utc.isoformat(),
        "lookback_days": lookback_days,
        "chunk_days": chunk_days,
        "request_timeout": request_timeout,
        "retries": retries,
        "stop_after_empty_windows": stop_after_empty_windows,
        "account_count": len(accounts),
        "windows": [],
        "combined": [],
        "output_dir": str(out_dir),
    }

    combined_by_account = {
        idx: {"transactions": {}, "orders": {}}
        for idx, _account in enumerate(accounts, start=1)
    }

    window_end = now_utc
    window_number = 0
    stop_older_transactions = False
    stop_older_orders = not include_orders
    empty_transaction_window_streak = 0

    while window_end > target_start and (
        not stop_older_transactions or not stop_older_orders
    ):
        window_number += 1
        window_start = max(target_start, window_end - timedelta(days=chunk_days))
        window_label = (
            f"{window_number:02d}_"
            f"{window_start.strftime('%Y%m%d')}_"
            f"{window_end.strftime('%Y%m%d')}"
        )
        window_entry: dict[str, Any] = {
            "label": window_label,
            "start_utc": window_start.isoformat(),
            "end_utc": window_end.isoformat(),
            "accounts": [],
        }

        tx_failures = 0
        order_failures = 0
        tx_attempts = 0
        order_attempts = 0
        window_transaction_count = 0
        window_order_count = 0

        for account_idx, account in enumerate(accounts, start=1):
            account_hash = account.get("hashValue") or account.get("hash")
            account_label = f"account_{account_idx}"
            account_entry: dict[str, Any] = {
                "label": account_label,
                "transactions_status": None,
                "transactions_count": None,
                "orders_status": None,
                "orders_count": None,
            }
            if not account_hash:
                account_entry["error"] = "missing account hash in account_numbers payload"
                window_entry["accounts"].append(account_entry)
                continue

            if not stop_older_transactions:
                tx_attempts += 1
                tx_resp, tx_error = _call_with_retries(
                    lambda: client.get_transactions(
                        account_hash,
                        start_date=window_start,
                        end_date=window_end,
                    ),
                    retries=retries,
                    label=f"{account_label} transactions {window_label}",
                )
                if tx_error is not None or tx_resp is None:
                    tx_failures += 1
                    account_entry["transactions_error"] = tx_error
                else:
                    account_entry["transactions_status"] = tx_resp.status_code
                    if tx_resp.status_code >= 400:
                        tx_failures += 1
                        account_entry["transactions_error"] = tx_resp.text[:500]
                    else:
                        tx_payload = tx_resp.json()
                        tx_count = len(tx_payload) if isinstance(tx_payload, list) else 0
                        account_entry["transactions_count"] = tx_count
                        window_transaction_count += tx_count
                        _write_json(
                            out_dir / f"{account_label}_{window_label}_transactions.json",
                            tx_payload,
                        )
                        if isinstance(tx_payload, list):
                            for item_idx, item in enumerate(tx_payload):
                                key = _stable_key(item, "tx", item_idx)
                                combined_by_account[account_idx]["transactions"][key] = item

            if not stop_older_orders:
                order_attempts += 1
                order_resp, order_error = _call_with_retries(
                    lambda: client.get_orders_for_account(
                        account_hash,
                        from_entered_datetime=window_start,
                        to_entered_datetime=window_end,
                        max_results=max_order_results,
                    ),
                    retries=retries,
                    label=f"{account_label} orders {window_label}",
                )
                if order_error is not None or order_resp is None:
                    order_failures += 1
                    account_entry["orders_error"] = order_error
                else:
                    account_entry["orders_status"] = order_resp.status_code
                    if order_resp.status_code >= 400:
                        order_failures += 1
                        account_entry["orders_error"] = order_resp.text[:500]
                    else:
                        order_payload = order_resp.json()
                        order_count = (
                            len(order_payload) if isinstance(order_payload, list) else 0
                        )
                        account_entry["orders_count"] = order_count
                        window_order_count += order_count
                        _write_json(
                            out_dir / f"{account_label}_{window_label}_orders.json",
                            order_payload,
                        )
                        if isinstance(order_payload, list):
                            for item_idx, item in enumerate(order_payload):
                                key = _stable_key(item, "order", item_idx)
                                combined_by_account[account_idx]["orders"][key] = item

            window_entry["accounts"].append(account_entry)

        if tx_attempts and tx_failures == tx_attempts:
            stop_older_transactions = True
            window_entry["transactions_stop_reason"] = (
                "all transaction requests failed for this window"
            )
        if order_attempts and order_failures == order_attempts:
            stop_older_orders = True
            window_entry["orders_stop_reason"] = "all order requests failed for this window"

        if tx_attempts and tx_failures == 0:
            if window_transaction_count == 0:
                empty_transaction_window_streak += 1
            else:
                empty_transaction_window_streak = 0
            window_entry["empty_transaction_window_streak"] = (
                empty_transaction_window_streak
            )
            if (
                stop_after_empty_windows > 0
                and empty_transaction_window_streak >= stop_after_empty_windows
            ):
                stop_older_transactions = True
                window_entry["transactions_stop_reason"] = (
                    f"{stop_after_empty_windows} consecutive empty transaction windows"
                )

        manifest["windows"].append(window_entry)
        if progress:
            cumulative_transactions = sum(
                len(account_payload["transactions"])
                for account_payload in combined_by_account.values()
            )
            cumulative_orders = sum(
                len(account_payload["orders"])
                for account_payload in combined_by_account.values()
            )
            print(
                f"{window_label}: "
                f"transactions={window_transaction_count} "
                f"orders={window_order_count} "
                f"cumulative_transactions={cumulative_transactions} "
                f"cumulative_orders={cumulative_orders}",
                flush=True,
            )
        window_end = window_start - timedelta(seconds=1)

    for account_idx, payloads in combined_by_account.items():
        account_label = f"account_{account_idx}"
        transactions = list(payloads["transactions"].values())
        orders = list(payloads["orders"].values())
        _write_json(out_dir / f"{account_label}_combined_transactions.json", transactions)
        if include_orders:
            _write_json(out_dir / f"{account_label}_combined_orders.json", orders)
        manifest["combined"].append(
            {
                "label": account_label,
                "transactions_count": len(transactions),
                "orders_count": len(orders),
            }
        )

    _write_json(out_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=730,
        help="How far back to attempt before stopping, default 730.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=59,
        help="Window size per request, default 59 to stay inside Schwab's 60-day limit.",
    )
    parser.add_argument(
        "--max-order-results",
        type=int,
        default=3000,
        help="Maximum order records requested per account/window.",
    )
    parser.add_argument(
        "--transactions-only",
        action="store_true",
        help="Skip order history and fetch transactions only.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count for transient request failures.",
    )
    parser.add_argument(
        "--stop-after-empty-windows",
        type=int,
        default=0,
        help="Stop after N consecutive empty transaction windows. 0 disables this.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print one progress line per completed date window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = fetch_history(
        project_root=Path.cwd(),
        lookback_days=args.lookback_days,
        chunk_days=args.chunk_days,
        max_order_results=args.max_order_results,
        include_orders=not args.transactions_only,
        request_timeout=args.timeout,
        retries=args.retries,
        stop_after_empty_windows=args.stop_after_empty_windows,
        progress=args.progress,
    )

    print(f"OUT_DIR={manifest['output_dir']}")
    print(f"TARGET_UTC={manifest['target_start_utc']} to {manifest['target_end_utc']}")
    print(f"ACCOUNT_COUNT={manifest['account_count']}")
    print(f"WINDOWS_ATTEMPTED={len(manifest['windows'])}")
    for row in manifest["combined"]:
        print(
            f"{row['label']}: "
            f"combined_transactions={row['transactions_count']} "
            f"combined_orders={row['orders_count']}"
        )
    if manifest["windows"]:
        last = manifest["windows"][-1]
        if last.get("transactions_stop_reason"):
            print(f"TRANSACTIONS_STOP={last['transactions_stop_reason']}")
        if last.get("orders_stop_reason"):
            print(f"ORDERS_STOP={last['orders_stop_reason']}")


if __name__ == "__main__":
    main()
