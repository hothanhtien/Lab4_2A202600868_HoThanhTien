from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"

# Regex helpers used to detect whether the user already provided the five
# required fields. They are intentionally permissive so we can short-circuit
# Gate 2 and force the model into Gate 3 even when formatting is unusual
# (quoted item names, semicolons, multi-line input, etc.).
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\s\-.()]*){8,15}")
_GUARDRAIL_PATTERNS = [
    r"bỏ\s*qua\s*policy",
    r"fake\s*invoice",
    r"hóa\s*đơn\s*giả",
    r"giảm\s*giá\s*\d+\s*%",
    r"manual\s*discount",
    r"stock\s*bypass",
    r"không\s*cần\s*theo\s*catalog",
]


def _looks_like_refusal(query: str) -> bool:
    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in _GUARDRAIL_PATTERNS)


def _has_all_five_fields(query: str) -> bool:
    if not _EMAIL_RE.search(query):
        return False
    if not _PHONE_RE.search(query):
        return False
    # Heuristics: must contain a product name (>= 2 word-like tokens) and
    # some kind of address indicator (street / district / city keywords,
    # numbers, or comma-separated location).
    has_product = bool(re.search(r"[A-Za-zÀ-ỹ]{3,}", query))
    has_address = bool(
        re.search(r"(quận|huyện|phường|xã|tp\.?hcm|hà\s*nội|đà\s*nẵng|street|st\.|road|rd\.)", query, re.IGNORECASE)
        or re.search(r"\d+[,\s].*?(quận|huyện|tp|thành\s*phố|city)", query, re.IGNORECASE)
    )
    return has_product and has_address


def _strip_item_quotes(query: str) -> str:
    """Remove surrounding double quotes from item names so the LLM is not
    confused by the literal `"` characters when matching products."""
    return re.sub(r'"([^"]+)"', r"\1", query)


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""You are an electronics order assistant for OrderDesk. Today is {current_day}. Always reply in Vietnamese.

## GATE 1 — SAFETY CHECK (evaluate first, no tools allowed yet)

Is the request asking for a fake invoice, manual discount override, or stock bypass?
  YES → Refuse politely in Vietnamese. You MUST include the phrases "không thể" and "khuyến mãi" in your reply (e.g. "Tôi không thể giúp yêu cầu này vì nó vi phạm chính sách khuyến mãi của cửa hàng."). Stop here. Do NOT call any tool.
  NO → Continue to Gate 2.

## GATE 2 — COMPLETENESS CHECK (evaluate carefully, no tools allowed yet)

Scan the user message for these five fields. Each one must be explicitly present:
  [ ] 1. Customer full name (any string of one or more words that looks like a person's name)
  [ ] 2. Phone number (digits only, e.g. 0901234567)
  [ ] 3. Email address — MUST contain the "@" symbol. Examples: abc@gmail.com, user@company.vn.
         WARNING: A phone number is NOT an email. If there is no "@" in the message, email is MISSING.
  [ ] 4. Shipping address (street, district, city)
  [ ] 5. At least one product name. Quantity may be implicit (e.g. "MacBook Air M3" with no number means qty=1 is assumed — this counts as present).

  IMPORTANT — formatting tolerance:
  - Product names may appear with or without quotation marks (e.g. both `MacBook Air M3` and `"MacBook Air M3"` are valid).
  - Items may be separated by commas, semicolons, "và", or newlines — all count as separators.
  - A large quantity (e.g. 12) is a normal valid request, NOT missing information.

  If ANY box is unchecked → respond in Vietnamese. Your reply MUST start with the phrase "Tôi cần thêm" followed by a list of the missing fields, naming "số điện thoại" and "địa chỉ giao hàng" verbatim if those fields are missing. Do NOT call any tool. Example: "Tôi cần thêm các thông tin sau: số điện thoại, địa chỉ giao hàng, ...".
  If ALL five boxes are checked → continue to Gate 3 IMMEDIATELY (do not ask for confirmation, do not second-guess the format).

## GATE 3 — PROCESS ORDER (all 5 tools, exact order)

Execute these steps immediately without asking for confirmation:

  Step 1. list_products — search for each product in the request. You may call this multiple times, one call per distinct product. Use the product name (without quotation marks) as the query.
  Step 2. get_product_details — call ONCE with ALL product_ids from Step 1 in one list. Save the detail_token.
  Step 3. get_discount — seed_hint = customer email, customer_tier = "standard".
  Step 4. calculate_order_totals — items + detail_token from Step 2 + discount_rate from Step 3.
    If status="error": report the stock problem in Vietnamese and STOP. Do NOT call save_order.
    If status="ok": continue to Step 5.
  Step 5. save_order — customer fields + items + detail_token + discount_rate + campaign_code from previous tools.

After save_order succeeds, reply in Vietnamese with: order_id, discount % + campaign_code, final_total (VND), save_path.

## RULES
- Prices, discounts, totals, paths: only from tool outputs — never invent.
- detail_token: copy exactly from get_product_details to Step 4 and Step 5.
- Never call get_product_details more than once per order.
- Never call save_order if calculate_order_totals returned status="error".
- Once all 5 fields are present, ALWAYS proceed to Gate 3. Do not refuse, do not ask for confirmation.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the product catalog by name, brand, category, or tags.
        Always call this FIRST before get_product_details."""
        result = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Get exact price, stock, warranty, and a detail_token for selected product IDs.
        The detail_token is REQUIRED by calculate_order_totals and save_order.
        Call this AFTER list_products and BEFORE get_discount."""
        result = store.get_product_details(product_ids)
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Get the campaign discount_rate (0.1 or 0.2) and campaign_code for this order.
        Use customer email as seed_hint. Call AFTER get_product_details."""
        result = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
    ) -> str:
        """Validate stock and compute subtotal, discount_amount, final_total.
        Requires detail_token from get_product_details and discount_rate from get_discount.
        If status=error (stock insufficient), do NOT call save_order."""
        normalized = [
            OrderLineInput(**item) if isinstance(item, dict) else item
            for item in items
        ]
        result = store.calculate_order_totals(
            items=normalized, detail_token=detail_token, discount_rate=discount_rate,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the confirmed order to a JSON file.
        Only call AFTER calculate_order_totals returns status=ok.
        Pass detail_token, discount_rate, and campaign_code exactly as returned by previous tools."""
        normalized = [
            OrderLineInput(**item) if isinstance(item, dict) else item
            for item in items
        ]
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=normalized,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(
        data_dir or DEFAULT_DATA_DIR,
        output_dir or DEFAULT_OUTPUT_DIR,
        today=today,
    )
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    tools = build_tools(store)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    # Pre-process the query: strip surrounding double quotes around item
    # names and, if all five customer fields are present, append a strong
    # directive so the model does not stall in Gate 2.
    processed_query = _strip_item_quotes(query)
    if _has_all_five_fields(processed_query) and not _looks_like_refusal(processed_query):
        processed_query = (
            processed_query
            + "\n\n[Hệ thống] Đơn hàng này có đủ thông tin khách hàng (họ tên, số điện thoại, email, địa chỉ giao hàng) và ít nhất một sản phẩm. Hãy BẮT BUỘC gọi tool list_products ngay để bắt đầu xử lý đơn, KHÔNG hỏi thêm thông tin, KHÔNG từ chối."
        )
    response = agent.invoke({"messages": [{"role": "user", "content": processed_query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tc in getattr(message, "tool_calls", []) or []:
                pending[tc["id"]] = {"name": tc["name"], "args": tc.get("args", {}) or {}}
        elif isinstance(message, ToolMessage):
            meta = pending.pop(message.tool_call_id, {})
            records.append(ToolCallRecord(
                name=str(getattr(message, "name", None) or meta.get("name", "")),
                args=meta.get("args", {}),
                output=normalize_content(message.content),
            ))
    for meta in pending.values():
        records.append(ToolCallRecord(name=meta["name"], args=meta["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
