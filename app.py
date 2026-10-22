"""Personal Expense Tracker.

Stack: Reflex (UI + state), Firebase Firestore (storage, via the server-side
Admin SDK), Firebase Authentication (email/password via the Identity Toolkit
REST API, verified server-side) and the Google Gen AI SDK (Gemini insights).

Run:  see README.md  (this file becomes ``expense_tracker/expense_tracker.py``
after ``reflex init``).

Security model (important):
* The browser never talks to Firestore. All reads/writes happen on the Reflex
  server through the Admin SDK, so no privileged credential reaches the client.
* The Admin SDK bypasses Firestore Security Rules, therefore user isolation is
  enforced HERE: every path is ``users/{uid}/expenses/{id}`` and ``uid`` comes
  only from a server-verified Firebase ID token (stored in a backend-only
  state variable that is never sent to the browser).
"""

import base64
import json
import logging
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import lru_cache
from typing import Any, Iterator
from uuid import uuid4

import firebase_admin
import reflex as rx
import requests
from dotenv import load_dotenv
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials, firestore
from google import genai
from google.api_core import exceptions as gexc
from google.genai import errors as genai_errors
from google.genai import types as genai_types

load_dotenv()
logger = logging.getLogger("expense_tracker")

# =============================================================================
# SECTION 1: Constants and configuration loading / validation
# =============================================================================

CATEGORIES: tuple[str, ...] = (
    "Food", "Transport", "Education", "Shopping", "Bills", "Healthcare", "Other",
)
PAYMENT_METHODS: tuple[str, ...] = ("Cash", "Card", "Bank Transfer")
DEFAULT_CURRENCIES: tuple[str, ...] = ("PKR", "USD", "EUR", "GBP", "AED", "SAR", "INR")
SORT_OPTIONS: tuple[str, ...] = (
    "Date (newest first)",
    "Date (oldest first)",
    "Amount (high to low)",
    "Amount (low to high)",
)
PERIOD_OPTIONS: tuple[str, ...] = ("This month", "Last month", "Last 30 days", "All time")

MAX_TITLE_LEN = 100
MAX_NOTES_LEN = 500
MAX_AMOUNT = Decimal("1000000000000")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings read from environment variables."""

    firebase_credentials_path: str
    firebase_credentials_json: str
    firebase_web_api_key: str
    gemini_api_key: str
    gemini_model: str
    ai_enabled: bool
    default_currency: str
    currencies: tuple[str, ...]
    request_timeout: int
    ai_cooldown: int

    @property
    def ai_available(self) -> bool:
        """AI is usable only when enabled AND a key is present."""
        return self.ai_enabled and bool(self.gemini_api_key)


def load_settings() -> Settings:
    """Read settings from the environment (never raises on missing values)."""
    raw_currencies = os.getenv("SUPPORTED_CURRENCIES", ",".join(DEFAULT_CURRENCIES))
    currencies = tuple(c.strip().upper() for c in raw_currencies.split(",") if c.strip())
    default_currency = os.getenv("DEFAULT_CURRENCY", "PKR").strip().upper() or "PKR"
    if default_currency not in currencies:
        currencies = (default_currency, *currencies)
    return Settings(
        firebase_credentials_path=os.getenv("FIREBASE_CREDENTIALS_PATH", "").strip(),
        firebase_credentials_json=os.getenv("FIREBASE_CREDENTIALS_JSON", "").strip(),
        firebase_web_api_key=os.getenv("FIREBASE_WEB_API_KEY", "").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        ai_enabled=_env_bool("AI_ENABLED", True),
        default_currency=default_currency,
        currencies=currencies,
        request_timeout=_env_int("REQUEST_TIMEOUT_SECONDS", 20),
        ai_cooldown=_env_int("AI_COOLDOWN_SECONDS", 10, minimum=0),
    )


def validate_settings(settings: Settings) -> list[str]:
    """Return human-readable configuration problems (empty list = OK)."""
    problems: list[str] = []
    if not (settings.firebase_credentials_path or settings.firebase_credentials_json):
        problems.append(
            "Set FIREBASE_CREDENTIALS_PATH (or FIREBASE_CREDENTIALS_JSON) in your .env file."
        )
    if not settings.firebase_web_api_key:
        problems.append("Set FIREBASE_WEB_API_KEY (Firebase Web API key) in your .env file.")
    return problems


SETTINGS = load_settings()
CONFIG_PROBLEMS = validate_settings(SETTINGS)


# =============================================================================
# SECTION 2: Errors and Firebase initialisation
# =============================================================================

class ServiceError(Exception):
    """Error whose message is safe to show to the end user."""


class AuthError(ServiceError):
    """Authentication failed."""


class ValidationError(ServiceError):
    """Input validation failed. ``errors`` maps field name -> message."""

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        super().__init__(" ".join(errors.values()))


def _load_service_account(settings: Settings) -> credentials.Certificate:
    """Build a credential from a JSON/base64 env var or a key-file path."""
    try:
        if settings.firebase_credentials_json:
            raw = settings.firebase_credentials_json
            try:
                info = json.loads(raw)
            except json.JSONDecodeError:
                info = json.loads(base64.b64decode(raw).decode("utf-8"))
            return credentials.Certificate(info)
        if not os.path.isfile(settings.firebase_credentials_path):
            raise ServiceError(
                "Firebase credentials file was not found. Check FIREBASE_CREDENTIALS_PATH."
            )
        return credentials.Certificate(settings.firebase_credentials_path)
    except ServiceError:
        raise
    except Exception as exc:  # malformed JSON, wrong key type, bad base64...
        logger.error("Could not load Firebase credentials: %s", type(exc).__name__)
        raise ServiceError("Firebase credentials are invalid. See the README.") from exc


@lru_cache(maxsize=1)
def get_firestore_client() -> Any:
    """Initialise the Admin SDK once and return a Firestore client.

    Exceptions are not cached by ``lru_cache``, so a fixed config is retried.
    """
    if not (SETTINGS.firebase_credentials_path or SETTINGS.firebase_credentials_json):
        raise ServiceError("Firebase is not configured. See the README.")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(_load_service_account(SETTINGS))
    return firestore.client()


# =============================================================================
# SECTION 3: Expense validation (server-side)
# =============================================================================

@dataclass(frozen=True)
class ExpenseInput:
    """Validated, normalised expense data ready to be written."""

    title: str
    amount: Decimal
    currency: str
    category: str
    expense_date: date
    payment_method: str
    notes: str


def validate_expense(raw: dict[str, Any], currencies: tuple[str, ...]) -> ExpenseInput:
    """Validate raw form data. Raises ValidationError with per-field messages."""
    errors: dict[str, str] = {}

    title = " ".join(str(raw.get("title", "")).split())
    if not title:
        errors["title"] = "Title is required."
    elif len(title) > MAX_TITLE_LEN:
        errors["title"] = f"Title must be at most {MAX_TITLE_LEN} characters."

    amount = Decimal("0")
    amount_text = str(raw.get("amount", "")).strip().replace(",", "")
    try:
        amount = Decimal(amount_text)
        if not amount.is_finite():
            raise InvalidOperation
    except InvalidOperation:
        errors["amount"] = "Amount must be a valid number."
    else:
        if amount <= 0:
            errors["amount"] = "Amount must be greater than zero."
        elif amount > MAX_AMOUNT:
            errors["amount"] = "Amount is too large."
        elif amount != amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP):
            errors["amount"] = "Amount can have at most 2 decimal places."
        else:
            amount = amount.quantize(Decimal("0.01"))

    currency = str(raw.get("currency", "")).strip().upper()
    if currency not in currencies:
        errors["currency"] = "Please choose a supported currency."

    category = str(raw.get("category", "")).strip()
    if category not in CATEGORIES:
        errors["category"] = "Please choose a valid category."

    payment_method = str(raw.get("payment_method", "")).strip()
    if payment_method not in PAYMENT_METHODS:
        errors["payment_method"] = "Please choose a valid payment method."

    expense_date = date.today()
    try:
        expense_date = date.fromisoformat(str(raw.get("date", "")).strip())
        if expense_date.year < 2000 or expense_date > date.today() + timedelta(days=366):
            errors["date"] = "Date is out of the allowed range."
    except ValueError:
        errors["date"] = "Please enter a valid date."

    notes = str(raw.get("notes", "")).strip()
    if len(notes) > MAX_NOTES_LEN:
        errors["notes"] = f"Notes must be at most {MAX_NOTES_LEN} characters."

    if errors:
        raise ValidationError(errors)
    return ExpenseInput(title, amount, currency, category, expense_date, payment_method, notes)


# =============================================================================
# SECTION 4: Firestore CRUD (kept out of the UI event handlers)
# =============================================================================
# Structure:  users/{uid}/expenses/{expenseId}
# Fields: title, amount (string, exact decimal), currency, category,
#         date ("YYYY-MM-DD"), payment_method, notes, created_at, updated_at

@contextmanager
def firestore_errors() -> Iterator[None]:
    """Translate low-level Firestore errors into user-safe ServiceErrors."""
    try:
        yield
    except ServiceError:
        raise
    except gexc.PermissionDenied as exc:
        raise ServiceError(
            "Firestore denied the request. Check the service account's permissions."
        ) from exc
    except gexc.NotFound as exc:
        raise ServiceError(
            "Not found. The expense may have been deleted, or the Firestore "
            "database has not been created yet."
        ) from exc
    except (gexc.ServiceUnavailable, gexc.DeadlineExceeded, gexc.RetryError) as exc:
        raise ServiceError("Firestore is unreachable right now. Please try again.") from exc
    except gexc.GoogleAPICallError as exc:
        logger.error("Firestore API error: %s", type(exc).__name__)
        raise ServiceError("The database request failed. Please try again.") from exc
    except Exception as exc:
        logger.error("Unexpected database error: %s", type(exc).__name__)
        raise ServiceError("Something went wrong while accessing the database.") from exc


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat(timespec="seconds")
    return str(value or "")


def _snapshot_to_record(snapshot: Any) -> dict[str, str]:
    data = snapshot.to_dict() or {}
    return {
        "id": snapshot.id,
        "title": str(data.get("title", "")),
        "amount": str(data.get("amount", "0")),
        "currency": str(data.get("currency", SETTINGS.default_currency)),
        "category": str(data.get("category", "Other")),
        "date": str(data.get("date", "")),
        "payment_method": str(data.get("payment_method", "")),
        "notes": str(data.get("notes", "")),
        "created_at": _iso(data.get("created_at")),
        "updated_at": _iso(data.get("updated_at")),
    }


def _input_to_fields(data: ExpenseInput) -> dict[str, Any]:
    return {
        "title": data.title,
        "amount": str(data.amount),
        "currency": data.currency,
        "category": data.category,
        "date": data.expense_date.isoformat(),
        "payment_method": data.payment_method,
        "notes": data.notes,
    }


class ExpenseRepository:
    """CRUD for ONE user's expenses. All paths are scoped to ``uid``."""

    def __init__(self, uid: str) -> None:
        if not uid:
            raise AuthError("Please sign in first.")
        self._uid = uid
        self._timeout = float(SETTINGS.request_timeout)

    def _collection(self) -> Any:
        return (
            get_firestore_client()
            .collection("users").document(self._uid).collection("expenses")
        )

    def list_expenses(self) -> list[dict[str, str]]:
        """Return all of the user's expenses (single read of the collection)."""
        with firestore_errors():
            snapshots = self._collection().stream(timeout=self._timeout)
            return [_snapshot_to_record(s) for s in snapshots]

    def create(self, data: ExpenseInput) -> dict[str, str]:
        """Create a new expense with a unique ID. Returns the saved record."""
        expense_id = uuid4().hex
        now = datetime.now(timezone.utc)
        fields = {**_input_to_fields(data), "created_at": now, "updated_at": now}
        with firestore_errors():
            self._collection().document(expense_id).create(fields, timeout=self._timeout)
        return {"id": expense_id, **{k: _iso(v) if k.endswith("_at") else str(v)
                                     for k, v in fields.items()}}

    def update(self, expense_id: str, data: ExpenseInput) -> dict[str, str]:
        """Update an existing expense. Fails (no duplicate) if it is missing."""
        now = datetime.now(timezone.utc)
        fields = {**_input_to_fields(data), "updated_at": now}
        with firestore_errors():
            self._collection().document(expense_id).update(fields, timeout=self._timeout)
        return {"id": expense_id, **{k: _iso(v) if k.endswith("_at") else str(v)
                                     for k, v in fields.items()}}

    def delete(self, expense_id: str) -> None:
        """Delete an expense; raises if it does not exist."""
        with firestore_errors():
            ref = self._collection().document(expense_id)
            if not ref.get(timeout=self._timeout).exists:
                raise ServiceError("That expense no longer exists.")
            ref.delete(timeout=self._timeout)


# =============================================================================
# SECTION 5: Firebase Authentication (server-side)
# =============================================================================

_AUTH_MESSAGES = {
    "EMAIL_EXISTS": "An account with this email already exists.",
    "WEAK_PASSWORD": "Password is too weak (use at least 8 characters).",
    "INVALID_EMAIL": "Please enter a valid email address.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please wait and try again.",
    "USER_DISABLED": "This account has been disabled.",
    "OPERATION_NOT_ALLOWED": "Email/password sign-in is not enabled in Firebase.",
}
_LOGIN_FAILED = "Incorrect email or password."


def firebase_authenticate(email: str, password: str, *, sign_up: bool) -> str:
    """Sign in / sign up through Firebase Auth and return the verified uid."""
    email = email.strip()
    if not EMAIL_RE.match(email):
        raise AuthError("Please enter a valid email address.")
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")
    if not SETTINGS.firebase_web_api_key:
        raise ServiceError("Firebase is not configured. See the README.")

    get_firestore_client()  # makes sure the Admin app exists (needed to verify tokens)
    action = "signUp" if sign_up else "signInWithPassword"
    try:
        response = requests.post(
            f"https://identitytoolkit.googleapis.com/v1/accounts:{action}",
            params={"key": SETTINGS.firebase_web_api_key},
            json={"email": email, "password": password, "returnSecureToken": True},
            timeout=SETTINGS.request_timeout,
        )
    except requests.RequestException as exc:
        raise AuthError("Could not reach Firebase Authentication. Check your connection.") from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code != 200:
        code = str(payload.get("error", {}).get("message", "")).split(" ")[0].strip(":")
        if not sign_up and code in {"INVALID_LOGIN_CREDENTIALS", "INVALID_PASSWORD", "EMAIL_NOT_FOUND"}:
            raise AuthError(_LOGIN_FAILED)
        raise AuthError(_AUTH_MESSAGES.get(code, "Authentication failed. Please try again."))

    try:
        decoded = firebase_auth.verify_id_token(payload["idToken"], clock_skew_seconds=10)
        return str(decoded["uid"])
    except Exception as exc:
        logger.error("ID token verification failed: %s", type(exc).__name__)
        raise AuthError("Could not verify your sign-in. Please try again.") from exc


# =============================================================================
# SECTION 6: Dashboard calculations (pure functions, easy to test)
# =============================================================================

def parse_amount(text: str) -> Decimal:
    """Parse a stored amount string; corrupt values count as zero."""
    try:
        value = Decimal(text)
        return value if value.is_finite() else Decimal("0")
    except (InvalidOperation, TypeError):
        return Decimal("0")


def format_money(amount: Decimal, currency: str) -> str:
    """Format like ``PKR 1,250.00``."""
    return f"{currency} {amount:,.2f}"


def filter_expenses(
    expenses: list[dict[str, str]], search: str, category: str, date_from: str, date_to: str,
) -> list[dict[str, str]]:
    """Apply search (title/notes), category and ISO date-range filters."""
    needle = search.strip().lower()
    result = []
    for item in expenses:
        if needle and needle not in item["title"].lower() and needle not in item["notes"].lower():
            continue
        if category != "All" and item["category"] != category:
            continue
        if date_from and item["date"] < date_from:
            continue
        if date_to and item["date"] > date_to:
            continue
        result.append(item)
    return result


def sort_expenses(expenses: list[dict[str, str]], option: str) -> list[dict[str, str]]:
    """Sort by date or amount according to one of SORT_OPTIONS."""
    if option.startswith("Amount"):
        return sorted(expenses, key=lambda e: parse_amount(e["amount"]),
                      reverse="high to low" in option)
    return sorted(expenses, key=lambda e: (e["date"], e["created_at"]),
                  reverse="newest" in option)


def totals_by_category(
    expenses: list[dict[str, str]], currency: str,
    start: date | None = None, end: date | None = None,
) -> dict[str, Decimal]:
    """Sum amounts per category for one currency (never mixes currencies)."""
    totals: dict[str, Decimal] = {}
    for item in expenses:
        if item["currency"] != currency:
            continue
        if start and item["date"] < start.isoformat():
            continue
        if end and item["date"] > end.isoformat():
            continue
        totals[item["category"]] = totals.get(item["category"], D
