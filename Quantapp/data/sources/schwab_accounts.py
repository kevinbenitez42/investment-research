"""Schwab account data fetch helpers."""

from __future__ import annotations

from typing import Any


def _json_payload(response_or_payload: Any) -> Any:
    """Return decoded JSON when passed an HTTP response-like object."""
    if hasattr(response_or_payload, "json"):
        return response_or_payload.json()
    return response_or_payload


def fetch_schwab_accounts(client: Any) -> Any:
    """Fetch Schwab account information using an authenticated schwab-py client."""
    return _json_payload(client.get_accounts())


def fetch_schwab_account_numbers(client: Any) -> Any:
    """Fetch Schwab account number/hash mappings."""
    return _json_payload(client.get_account_numbers())


def extract_schwab_account_entries(account_numbers_payload: Any) -> list[dict]:
    """Normalize Schwab account-number payloads to a list of account records."""
    if isinstance(account_numbers_payload, dict):
        entries = (
            account_numbers_payload.get("accounts")
            or account_numbers_payload.get("accountNumbers")
            or []
        )
    elif isinstance(account_numbers_payload, list):
        entries = account_numbers_payload
    else:
        entries = []

    return [entry for entry in entries if isinstance(entry, dict)]


def select_schwab_account_hash(
    account_numbers_payload: Any,
    *,
    account_hash: str | None = None,
    account_index: int = 0,
) -> str:
    """Select an account hash from Schwab account-number payloads."""
    if account_hash:
        return str(account_hash)

    accounts = extract_schwab_account_entries(account_numbers_payload)
    if not accounts:
        raise ValueError("No Schwab accounts returned from get_account_numbers().")

    try:
        selected_account = accounts[account_index]
    except IndexError as exc:
        raise ValueError(f"Schwab account_index {account_index} is out of range.") from exc

    selected_hash = selected_account.get("hashValue") or selected_account.get("hash")
    if not selected_hash:
        raise ValueError("Missing hashValue in Schwab account numbers response.")
    return str(selected_hash)


def _resolve_positions_field() -> Any:
    """Resolve schwab-py's positions field lazily."""
    try:
        from schwab.client import Client
    except ImportError:
        return None
    return Client.Account.Fields.POSITIONS


def fetch_schwab_account(
    client: Any,
    account_hash: str,
    *,
    fields: Any | None = None,
) -> dict:
    """Fetch one Schwab account, including positions when supported."""
    resolved_fields = _resolve_positions_field() if fields is None else fields
    if resolved_fields is None:
        return _json_payload(client.get_account(account_hash))
    return _json_payload(client.get_account(account_hash, fields=resolved_fields))


def extract_schwab_positions(account_payload: dict) -> list[dict]:
    """Extract positions from a Schwab account payload."""
    if not isinstance(account_payload, dict):
        return []
    securities_account = account_payload.get("securitiesAccount", {})
    if not isinstance(securities_account, dict):
        return []
    positions = securities_account.get("positions", [])
    return positions if isinstance(positions, list) else []


def fetch_schwab_account_snapshot(
    client: Any,
    *,
    account_hash: str | None = None,
    account_index: int = 0,
    fields: Any | None = None,
) -> dict[str, Any]:
    """Fetch account list, selected account, and raw positions from Schwab."""
    account_information = fetch_schwab_accounts(client)
    account_numbers = fetch_schwab_account_numbers(client)
    selected_account_hash = select_schwab_account_hash(
        account_numbers,
        account_hash=account_hash,
        account_index=account_index,
    )
    account = fetch_schwab_account(client, selected_account_hash, fields=fields)
    positions = extract_schwab_positions(account)
    return {
        "account_information": account_information,
        "account_numbers": account_numbers,
        "account_hash": selected_account_hash,
        "account": account,
        "positions": positions,
    }
