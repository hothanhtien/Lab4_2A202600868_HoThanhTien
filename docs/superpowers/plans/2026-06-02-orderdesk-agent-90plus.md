# OrderDesk Agent (90+ Score) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a high-scoring (90+) electronics order agent using LangGraph + OpenAI gpt-4o-mini, with tight system prompt, explicit tool schemas, stock/guardrail handling, and Vietnamese responses.

**Architecture:** OrderDataStore handles all data logic (product search, discount simulation, order persistence); graph.py wires the LangGraph agent with 5 tools and a strict system prompt; scoring.py grades via JSON comparison + tool sequence + LLM judge.

**Tech Stack:** Python 3.11+, LangGraph, LangChain, langchain-openai, Pydantic v2, OpenAI gpt-4o-mini

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/core/llm.py` | Add OpenAI provider branch |
| Modify | `grade/scoring.py` | Add `"openai"` to `--provider` choices |
| Modify | `pyproject.toml` | Add `langchain-openai` dependency |
| Create | `.env` | Set `OPENAI_API_KEY` and `LLM_MODEL` |
| Implement | `src/utils/data_store.py` | Full `OrderDataStore` (search, discount, save) |
| Implement | `src/agent/graph.py` | System prompt, 5 tools, agent builder, runner |

---

## Task 1: Setup venv lab4

**Files:**
- No code files — shell setup only

- [ ] **Step 1: Create and activate venv**

```powershell
cd "D:\AI_Vin\Lab4_repo\Day04-E403-Prompt-Engineering-Tool-Calling-Labs"
python -m venv lab4
lab4\Scripts\Activate.ps1
```

Expected: prompt shows `(lab4)`.

- [ ] **Step 2: Add langchain-openai to pyproject.toml**

In `pyproject.toml`, replace the dependencies block with:

```toml
dependencies = [
  "langchain>=1.0.0",
  "langgraph>=0.6.0",
  "langchain-core>=0.3.0",
  "langchain-google-genai>=2.1.0",
  "langchain-ollama>=0.3.0",
  "langchain-openai>=0.3.0",
  "pydantic>=2.8.0",
  "python-dotenv>=1.0.1",
]
```

- [ ] **Step 3: Install dependencies**

```powershell
pip install -e ".[dev]"
```

Expected: installs langchain-openai and all deps without errors.

- [ ] **Step 4: Create .env from example**

Copy `.env.example` to `.env` and fill in:

```
OPENAI_API_KEY=sk-YOUR_KEY_HERE
LLM_MODEL=gpt-4o-mini
```

---

## Task 2: Add OpenAI provider support

**Files:**
- Modify: `src/core/llm.py:29-51`
- Modify: `grade/scoring.py:248-258`

- [ ] **Step 1: Add OpenAI branch to build_chat_model in llm.py**

In `src/core/llm.py`, replace the function body of `build_chat_model` so it reads:

```python
def build_chat_model(
    *,
    provider: str = "google",
    model_name: str | None = None,
    temperature: float = 0.0,
):
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen3.5:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    raise ValueError("Supported providers: google, openai, ollama.")
```

- [ ] **Step 2: Add "openai" to scoring.py choices**

In `grade/scoring.py` line ~252, change:

```python
parser.add_argument("--provider", default="google", choices=["google", "ollama"])
```

to:

```python
parser.add_argument("--provider", default="google", choices=["google", "openai", "ollama"])
```

Also change the judge-provider line ~255:

```python
parser.add_argument("--judge-provider", default=None, choices=["google", "openai", "ollama"])
```

- [ ] **Step 3: Verify import works**

```powershell
python -c "from src.core.llm import build_chat_model; m = build_chat_model(provider='openai'); print(type(m))"
```

Expected: `<class 'langchain_openai.chat_models.base.ChatOpenAI'>`

---

## Task 3: Implement OrderDataStore

**Files:**
- Implement: `src/utils/data_store.py`

- [ ] **Step 1: Write full implementation**

Replace the entire content of `src/utils/data_store.py` with:

```python
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from src.core.schemas import OrderLineInput, ProductRecord


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    compact = re.sub(r"[^a-zA-Z0-9]+", " ", stripped.lower())
    return re.sub(r"\s+", " ", compact).strip()


class OrderDataStore:
    def __init__(self, data_dir: Path, output_dir: Path, *, today: str | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.today = today or "2026-06-01"
        raw_products = json.loads((self.data_dir / "products.json").read_text(encoding="utf-8"))
        self.products = [ProductRecord(**item) for item in raw_products]
        self.product_index = {item.product_id: item for item in self.products}
        self.category_aliases = {
            "laptop": "laptop", "notebook": "laptop",
            "monitor": "monitor", "screen": "monitor", "man hinh": "monitor",
            "mouse": "mouse", "chuot": "mouse",
            "keyboard": "keyboard", "ban phim": "keyboard",
            "headphone": "headphone", "tai nghe": "headphone",
            "dock": "dock",
            "storage": "storage", "ssd": "storage",
            "stand": "stand",
            "webcam": "webcam",
        }

    @staticmethod
    def build_detail_token(product_ids: list[str]) -> str:
        normalized = "|".join(sorted(product_ids))
        return "DET-" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10].upper()

    def validate_detail_token(self, product_ids: list[str], detail_token: str) -> bool:
        return detail_token == self.build_detail_token(product_ids)

    def canonicalize_category(self, value: str | None) -> str | None:
        if not value:
            return None
        return self.category_aliases.get(_normalize(value), _normalize(value))

    def list_products(
        self,
        *,
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> list[dict]:
        normalized_query = _normalize(query or "")
        query_terms = [t for t in normalized_query.split() if t and not t.isdigit() and len(t) > 1]
        wanted_category = self.canonicalize_category(category)
        wanted_tags = {_normalize(tag) for tag in (required_tags or []) if tag.strip()}
        results: list[tuple[int, int, str, dict]] = []

        for product in self.products:
            if in_stock_only and product.stock <= 0:
                continue
            if wanted_category and product.category != wanted_category:
                continue
            if max_unit_price is not None and product.unit_price > max_unit_price:
                continue

            haystack = _normalize(" ".join([product.name, product.brand, product.category, product.description, *product.tags]))
            score = 0
            matched_terms: list[str] = []
            for term in query_terms:
                if term in haystack:
                    score += 2
                    matched_terms.append(term)
            for tag in wanted_tags:
                if tag in haystack:
                    score += 3
                    matched_terms.append(tag)
                else:
                    score -= 1
            if wanted_category:
                score += 3
            if query_terms and not matched_terms:
                continue
            results.append((
                score,
                product.stock,
                product.product_id,
                {
                    "product_id": product.product_id,
                    "name": product.name,
                    "brand": product.brand,
                    "category": product.category,
                    "tags": product.tags,
                    "matched_terms": sorted(set(matched_terms)),
                    "next_step": "Call get_product_details with the chosen product_id list to verify price, stock, and the detail_token.",
                },
            ))

        results.sort(key=lambda item: (-item[0], self.product_index[item[2]].unit_price, item[2]))
        return [item[-1] for item in results[:limit]]

    def get_product_details(self, product_ids: list[str]) -> dict:
        details: list[dict] = []
        for product_id in product_ids:
            product = self.product_index.get(product_id)
            if not product:
                details.append({"product_id": product_id, "status": "not_found"})
                continue
            details.append({
                "status": "ok",
                "product_id": product.product_id,
                "sku": product.sku,
                "name": product.name,
                "brand": product.brand,
                "category": product.category,
                "unit_price": product.unit_price,
                "stock": product.stock,
                "warranty_months": product.warranty_months,
                "tags": product.tags,
                "description": product.description,
            })
        found_ids = [d["product_id"] for d in details if d.get("status") == "ok"]
        return {
            "status": "ok" if found_ids else "error",
            "detail_token": self.build_detail_token(found_ids) if found_ids else "",
            "items": details,
        }

    def get_discount(self, *, seed_hint: str, customer_tier: str = "standard") -> dict:
        normalized_seed = seed_hint.strip().lower()
        digest = hashlib.sha256(f"{customer_tier}|{normalized_seed}".encode("utf-8")).hexdigest()
        discount_rate = 0.2 if int(digest[-2:], 16) % 10 < 4 else 0.1
        return {
            "status": "ok",
            "seed_hint": seed_hint,
            "customer_tier": customer_tier,
            "discount_rate": discount_rate,
            "campaign_code": f"FLASH-{int(discount_rate * 100):02d}",
        }

    def calculate_order_totals(self, *, items: list[OrderLineInput], detail_token: str, discount_rate: float) -> dict:
        if discount_rate not in {0.1, 0.2}:
            return {"status": "error", "errors": [f"Unsupported discount rate: {discount_rate}."]}
        requested_ids = [item.product_id for item in items]
        if not self.validate_detail_token(requested_ids, detail_token):
            return {
                "status": "error",
                "errors": ["Invalid detail token. Call get_product_details again before pricing this order."],
            }
        errors: list[str] = []
        lines: list[dict] = []
        subtotal = 0
        for item in sorted(items, key=lambda i: i.product_id):
            product = self.product_index.get(item.product_id)
            if not product:
                errors.append(f"Unknown product_id: {item.product_id}.")
                continue
            if item.quantity > product.stock:
                errors.append(f"Insufficient stock for {product.name}: requested {item.quantity}, available {product.stock}.")
                continue
            line_total = product.unit_price * item.quantity
            subtotal += line_total
            lines.append({
                "product_id": product.product_id,
                "sku": product.sku,
                "name": product.name,
                "category": product.category,
                "quantity": item.quantity,
                "unit_price": product.unit_price,
                "line_total": line_total,
            })
        if errors:
            return {"status": "error", "errors": errors, "items": lines}
        discount_amount = int(subtotal * discount_rate)
        final_total = subtotal - discount_amount
        return {
            "status": "ok",
            "items": lines,
            "pricing": {
                "currency": "VND",
                "subtotal": subtotal,
                "discount_rate": discount_rate,
                "discount_amount": discount_amount,
                "final_total": final_total,
            },
            "detail_token": detail_token,
        }

    def save_order(
        self,
        *,
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
    ) -> dict:
        pricing_snapshot = self.calculate_order_totals(
            items=items, detail_token=detail_token, discount_rate=discount_rate,
        )
        if pricing_snapshot["status"] != "ok":
            return pricing_snapshot

        normalized_items = sorted(
            [{"product_id": item.product_id, "quantity": item.quantity} for item in items],
            key=lambda i: i["product_id"],
        )
        seed_payload = json.dumps(
            {
                "customer_email": customer_email.strip().lower(),
                "customer_phone": "".join(ch for ch in customer_phone if ch.isdigit()),
                "items": normalized_items,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        order_id = "ORD-" + hashlib.sha1(seed_payload.encode("utf-8")).hexdigest()[:10].upper()
        relative_path = Path("artifacts") / "orders" / f"{order_id}.json"
        absolute_path = self.output_dir / f"{order_id}.json"

        payload = {
            "order_id": order_id,
            "created_at": self.today,
            "status": "confirmed",
            "customer": {
                "name": customer_name.strip(),
                "phone": customer_phone.strip(),
                "email": customer_email.strip(),
                "shipping_address": shipping_address.strip(),
            },
            "items": pricing_snapshot["items"],
            "pricing": pricing_snapshot["pricing"],
            "discount": {
                "campaign_code": campaign_code,
                "customer_tier": customer_tier,
            },
            "notes": notes.strip(),
            "save_path": str(relative_path),
            "source": "llm-order-agent",
        }
        absolute_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "status": "saved",
            "order_id": order_id,
            "path": str(absolute_path),
            "saved_order": payload,
        }
```

- [ ] **Step 2: Quick smoke test**

```powershell
python -c "
from pathlib import Path
from src.utils.data_store import OrderDataStore
store = OrderDataStore(Path('data'), Path('artifacts/orders'), today='2026-06-01')
print('Products loaded:', len(store.products))
results = store.list_products(query='ASUS ROG')
print('Search results:', [r['name'] for r in results])
"
```

Expected: `Products loaded: 19` and at least one ASUS ROG result.

---

## Task 4: Implement src/agent/graph.py

**Files:**
- Implement: `src/agent/graph.py`

- [ ] **Step 1: Write full implementation**

Replace the entire content of `src/agent/graph.py` with:

```python
from __future__ import annotations

import json
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
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""Bạn là trợ lý đặt hàng điện tử cho cửa hàng OrderDesk. Hôm nay là {current_day}.

## NHIỆM VỤ
Hỗ trợ khách hàng tạo đơn hàng điện tử hoàn chỉnh và lưu vào hệ thống.

## THÔNG TIN BẮT BUỘC TRƯỚC KHI GỌI TOOL
Trước khi gọi bất kỳ tool nào, PHẢI có đủ TẤT CẢ 5 thông tin sau:
1. Họ tên khách hàng
2. Số điện thoại
3. Email
4. Địa chỉ giao hàng
5. Sản phẩm cụ thể + số lượng

Nếu thiếu bất kỳ thông tin nào, hãy hỏi lại ngay và KHÔNG gọi tool nào cả.

## QUY TRÌNH BẮT BUỘC (đúng thứ tự, không được bỏ bước)
Khi đã có đủ thông tin, thực hiện ĐÚNG theo thứ tự sau:
1. `list_products` — tìm kiếm sản phẩm trong catalog
2. `get_product_details` — lấy giá, tồn kho, và detail_token
3. `get_discount` — lấy mã khuyến mãi và discount_rate (dùng email làm seed_hint)
4. `calculate_order_totals` — tính tổng tiền (bắt buộc truyền detail_token từ bước 2)
5. `save_order` — lưu đơn hàng (bắt buộc truyền detail_token, discount_rate, campaign_code từ các bước trên)

KHÔNG được bỏ qua bước nào. KHÔNG được đảo thứ tự. KHÔNG được gọi save_order trước calculate_order_totals.

## QUY TẮC DỮ LIỆU (bắt buộc)
- Giá sản phẩm: chỉ dùng unit_price từ get_product_details — KHÔNG tự đặt giá
- Discount: chỉ dùng discount_rate và campaign_code từ get_discount — KHÔNG tự tính hoặc chỉnh sửa
- Tổng tiền: chỉ dùng kết quả từ calculate_order_totals — KHÔNG tự tính
- Save path: chỉ dùng path từ kết quả save_order — KHÔNG tự đặt tên file
- detail_token: phải truyền y nguyên từ get_product_details sang calculate_order_totals và save_order

## TỪ CHỐI TUYỆT ĐỐI (không thảo luận, từ chối ngay)
Từ chối và giải thích ngắn gọn nếu khách yêu cầu:
- Tạo hóa đơn giả, hóa đơn không có thật
- Áp dụng mức giảm giá tùy ý hoặc override discount từ hệ thống
- Đặt hàng vượt tồn kho (bypass stock check)
- Bỏ qua chính sách hoặc catalog sản phẩm
- Bất kỳ yêu cầu nào yêu cầu bỏ qua quy trình tool

## XỬ LÝ TỒN KHO KHÔNG ĐỦ
Nếu calculate_order_totals trả về lỗi tồn kho, thông báo rõ sản phẩm nào không đủ hàng và số lượng hiện có. KHÔNG lưu đơn.

## PHẢN HỒI CUỐI
Sau khi lưu đơn thành công, trả lời bằng tiếng Việt, ngắn gọn, bao gồm:
- Xác nhận đơn hàng đã được lưu
- Mã đơn hàng (order_id)
- Mức giảm giá (discount %) và mã khuyến mãi
- Tổng tiền phải thanh toán (final_total, đơn vị VND)
- Đường dẫn file đã lưu
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
        """Search the product catalog. Use descriptive query terms, category filter, and tags.
        Always call this first before get_product_details."""
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
        """Get exact price, stock, warranty, and a detail_token for the selected product IDs.
        The detail_token is required by calculate_order_totals and save_order.
        Always call this after list_products and before get_discount."""
        result = store.get_product_details(product_ids)
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Get the campaign discount_rate (0.1 or 0.2) and campaign_code for this order.
        Use customer email as seed_hint. Call this after get_product_details."""
        result = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(result, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(
        items: list,
        detail_token: str,
        discount_rate: float,
    ) -> str:
        """Validate stock and compute subtotal, discount_amount, and final_total.
        Requires detail_token from get_product_details and discount_rate from get_discount.
        Returns error if stock is insufficient — do NOT save in that case."""
        from src.core.schemas import OrderLineInput
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
        items: list,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the confirmed order to a JSON file. Only call after calculate_order_totals succeeds.
        Pass detail_token, discount_rate, and campaign_code exactly as returned by previous tools."""
        from src.core.schemas import OrderLineInput
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
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
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
```

- [ ] **Step 2: Verify import**

```powershell
python -c "from src.agent.graph import build_system_prompt, build_agent, run_agent; print('OK')"
```

Expected: `OK`

---

## Task 5: Run baseline to check starting score

**Files:** No changes — read-only test

- [ ] **Step 1: Run simple_solution grader (baseline)**

```powershell
python grade/scoring.py --module simple_solution.agent.graph --provider openai --model-name gpt-4o-mini --today 2026-06-01
```

Note the `overall_score` — this is the weak baseline. Should be 30-50.

- [ ] **Step 2: Run your implementation**

```powershell
python grade/scoring.py --module src.agent.graph --provider openai --model-name gpt-4o-mini --today 2026-06-01
```

Expected: `overall_score` ≥ 90.

If score < 90, proceed to Task 6.

---

## Task 6: Debug and iterate if score < 90

**Files:**
- Modify: `src/agent/graph.py` (system prompt or tools)

This task describes the debugging loop. Run it after each scoring pass.

- [ ] **Step 1: Identify failing cases**

In the JSON output from scoring.py, find cases with `score < max_score`. Note the `case_id` and `feedback` list.

- [ ] **Step 2: Diagnose by category**

| Symptom | Root cause | Fix in |
|---------|-----------|--------|
| `Tool trace mismatch` | Model skipped/reordered tools | Tighten tool order in system prompt |
| `root.order_id: expected X, got Y` | Wrong customer data or items → wrong hash | Check that all 5 fields are passed correctly |
| `root.pricing.discount_rate: expected X` | Wrong seed_hint for get_discount | Ensure email is used as seed_hint |
| `Missing saved_order payload` | save_order not called or returned error | Check calculate_order_totals succeeded first |
| `Order should not have been saved` | Agent saved on clarification/guardrail case | Strengthen refusal/clarification section of prompt |
| LLM judge feedback about missing order_id | Final answer incomplete | Prompt: require order_id + total in response |

- [ ] **Step 3: Tighten system prompt for tool order issues**

If tools are being skipped, add more explicit language to the **QUY TRÌNH BẮT BUỘC** section, for example:

```
CẢNH BÁO: Nếu bỏ qua bước nào hoặc gọi sai thứ tự, đơn hàng sẽ không hợp lệ.
Không được gọi save_order nếu calculate_order_totals chưa trả về status="ok".
```

- [ ] **Step 4: Re-run grader and repeat**

```powershell
python grade/scoring.py --module src.agent.graph --provider openai --model-name gpt-4o-mini --today 2026-06-01
```

Repeat Steps 1-4 until `overall_score` ≥ 90.

---

## Quick Reference: Scoring Breakdown

| Case ID | Type | Max score | Key requirement |
|---------|------|-----------|----------------|
| gaming_bundle_exact_match | normal | 100 | JSON exact match + full tool sequence |
| office_workstation_bundle | normal | 100 | JSON exact match + full tool sequence |
| mobile_creator_pack | normal | 100 | JSON exact match + full tool sequence |
| accessory_bundle_bulk | normal | 100 | JSON exact match + full tool sequence |
| workstation_bundle_mixed_language | normal | 100 | JSON exact match + full tool sequence |
| executive_dual_monitor_bundle | normal | 100 | JSON exact match + full tool sequence |
| creator_premium_bundle_quotes | normal | 100 | JSON exact match + full tool sequence |
| insufficient_stock_headphones | edge | 100 | No save, stock error message |
| insufficient_stock_multi_line_monitor | edge | 100 | No save, stock error message |
| clarification_missing_shipping | clarification | 100 | No tools called, ask for address |
| clarification_missing_email_only | clarification | 100 | No tools called, ask for email |
| guardrail_fake_invoice | guardrail | 100 | No tools called, refuse |
| guardrail_discount_and_stock_bypass | guardrail | 100 | No tools called, refuse |

**Target: all 13 cases pass → overall_score ≥ 90**
