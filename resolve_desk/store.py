"""SQLite retail sandbox. The model can propose; only explicit confirmation commits."""

import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .protocol import parse_action


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS orders(id TEXT PRIMARY KEY,
                body TEXT NOT NULL, revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,
                body TEXT NOT NULL, revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS proposals(id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                body TEXT NOT NULL, status TEXT NOT NULL, receipt TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
                kind TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL);
            """)

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.path, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except BaseException:
            c.rollback()
            raise
        finally:
            c.close()

    def create(self, case):
        sid = secrets.token_hex(12)
        state = {
            "id": sid,
            "case_id": case["id"],
            "customer": case["customer"],
            "products": case["products"],
            "user_id": None,
            "read_orders": {},
            "messages": [{"role": "customer", "content": case["initial_message"]}],
            "observations": [],
            "status": "running",
            "proposal": None,
            "revision": 0,
        }
        # Every session gets a separate sandbox order key, even for the same demo case.
        state["order_keys"] = {o["id"]: sid + ":" + o["id"] for o in case["orders"]}
        with self.connect() as c:
            for order in case["orders"]:
                c.execute(
                    "INSERT INTO orders VALUES(?,?,0)", (state["order_keys"][order["id"]], canonical(order))
                )
            c.execute("INSERT INTO sessions VALUES(?,?,0)", (sid, canonical(state)))
            self.event(c, sid, "session_created", {"case_id": case["id"]})
        return self.get(sid)

    @staticmethod
    def event(c, sid, kind, value):
        c.execute(
            "INSERT INTO events(session_id,kind,body,created) VALUES(?,?,?,?)",
            (sid, kind, canonical(value), time.time()),
        )

    def get(self, sid):
        with self.connect() as c:
            row = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
            if row is None:
                raise KeyError("Session not found")
            state = json.loads(row["body"])
            state["revision"] = row["revision"]
            return state

    @staticmethod
    def save(c, state):
        old = state["revision"]
        state["revision"] += 1
        changed = c.execute(
            "UPDATE sessions SET body=?,revision=? WHERE id=? AND revision=?",
            (canonical(state), state["revision"], state["id"], old),
        ).rowcount
        if changed != 1:
            raise Conflict("Session changed; reload before continuing")

    @staticmethod
    def check_revision(state, revision):
        if state["revision"] != revision:
            raise Conflict("Stale session revision")

    def customer_message(self, sid, text, revision):
        if not text.strip() or len(text) > 2000:
            raise ValueError("Message must contain 1–2000 characters")
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            state = self.get(sid)
            self.check_revision(state, revision)
            if state["status"] in {"completed", "handoff"}:
                raise Conflict("This session is closed; start a new case")
            if state["proposal"]:
                c.execute(
                    "UPDATE proposals SET status='revoked' WHERE id=? AND status='pending'",
                    (state["proposal"]["id"],),
                )
                state["proposal"] = None
            state["messages"].append({"role": "customer", "content": text})
            state["status"] = "running"
            self.event(c, sid, "customer_message", {"content": text})
            self.save(c, state)
        return self.get(sid)

    def apply(self, sid, raw, timing, revision):
        """One validated model decision; policy failures stay visible to the model and evaluator."""
        try:
            action = parse_action(raw)
        except ValueError as e:
            action, parse_error = None, str(e)[:250]
        else:
            parse_error = None
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            state = self.get(sid)
            self.check_revision(state, revision)
            if state["status"] != "running":
                raise Conflict("Session is not awaiting a model decision")
            if action is None:
                result = {"error": "invalid_action_json", "detail": parse_error}
                state["status"] = "failed"
            else:
                try:
                    result = self.tool(c, state, action["tool"], action["arguments"])
                except (ValueError, KeyError) as e:
                    result = {"error": "policy_or_argument_rejection", "detail": str(e)[:250]}
            record = {"action": action, "result": result, "raw": raw, "timing": timing}
            state["observations"].append(record)
            self.event(c, sid, "model_tool", record)
            self.save(c, state)
        return self.get(sid)

    def tool(self, c, state, name, args):
        if name == "ask_customer":
            state["status"] = "awaiting_customer"
            return {"question": args["message"], "field": args["field"]}
        if name == "handoff":
            state["status"] = "handoff"
            return {"handoff": True, "reason": args["reason"]}
        if name == "identify_customer":
            if args["email"].casefold() != state["customer"]["email"].casefold():
                raise ValueError("No matching demo customer")
            state["user_id"] = state["customer"]["id"]
            return {
                "user_id": state["user_id"],
                "gift_card": state["customer"]["gift_card"],
                "note": "Demo identity lookup only; not production authentication",
            }
        if not state["user_id"]:
            raise ValueError("Identify the customer by email first")
        if name == "get_product":
            product = state["products"].get(args["product_id"])
            if not product:
                raise ValueError("Unknown product_id")
            return product
        oid = args["order_id"]
        key = state["order_keys"].get(oid)
        row = c.execute("SELECT * FROM orders WHERE id=?", (key,)).fetchone()
        if not row:
            raise ValueError("Unknown order_id")
        order = json.loads(row["body"])
        if order["user_id"] != state["user_id"]:
            raise ValueError("Other customers' orders are outside this session")
        if name == "get_order":
            state["read_orders"][oid] = row["revision"]
            return {**order, "revision": row["revision"]}
        if state["read_orders"].get(oid) != row["revision"]:
            raise ValueError("Read the current order before proposing an operation")
        body = {
            "kind": name.removeprefix("propose_"),
            "order_id": oid,
            "order_revision": row["revision"],
            "arguments": args,
        }
        if name == "propose_cancel":
            if order["status"] != "pending" or args["reason"] not in {
                "no longer needed",
                "ordered by mistake",
            }:
                raise ValueError("Cancellation requires pending status and an accepted reason")
            body.update(
                amount_cents=sum(i["price_cents"] for i in order["items"]),
                target_status="cancelled",
                payment_method="original",
            )
        else:
            if order["status"] != "delivered":
                raise ValueError("Returns and exchanges require delivered status")
            item = next((i for i in order["items"] if i["item_id"] == args["item_id"]), None)
            if not item:
                raise ValueError("item_id is not an item in this order")
            payment = args["payment_method"]
            if payment not in {"original", state["customer"]["gift_card"]["id"]}:
                raise ValueError("Payment must be original or this customer's existing gift card")
            body.update(item_id=item["item_id"], payment_method=payment)
            if name == "propose_return":
                body.update(amount_cents=item["price_cents"], target_status="return requested")
            elif name == "propose_exchange":
                product = state["products"][item["product_id"]]
                read_product = any(
                    o["action"]
                    and o["action"]["tool"] == "get_product"
                    and o["action"]["arguments"]["product_id"] == item["product_id"]
                    and "error" not in o["result"]
                    for o in state["observations"]
                )
                new = next((v for v in product["variants"] if v["item_id"] == args["new_item_id"]), None)
                if not read_product or not new or not new["available"] or new["item_id"] == item["item_id"]:
                    raise ValueError(
                        "Inspect the product; select an available different same-product variant"
                    )
                delta = new["price_cents"] - item["price_cents"]
                if payment != "original" and delta > state["customer"]["gift_card"]["balance_cents"]:
                    raise ValueError("Gift card cannot cover the price difference")
                body.update(
                    amount_cents=delta, new_item_id=new["item_id"], target_status="exchange requested"
                )
            else:
                raise ValueError("Unsupported operation")
        # Client receives an opaque proposal ID; it cannot alter the stored action body.
        pid = secrets.token_hex(16)
        body["fingerprint"] = hashlib.sha256(canonical(body).encode()).hexdigest()
        c.execute("INSERT INTO proposals VALUES(?,?,?,'pending',NULL)", (pid, state["id"], canonical(body)))
        state["proposal"] = {"id": pid, **body}
        state["status"] = "awaiting_confirmation"
        return {"proposal": state["proposal"], "requires_customer_confirmation": True}

    def confirm(self, sid, pid, revision, accepted, fault=None):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM proposals WHERE id=? AND session_id=?", (pid, sid)).fetchone()
            if row is None:
                raise Conflict("Proposal does not belong to this session")
            if row["status"] == "committed":
                return json.loads(row["receipt"])  # Same proposal is exactly-once, even after retry.
            if row["status"] != "pending":
                raise Conflict("Proposal is no longer pending")
            state = self.get(sid)
            self.check_revision(state, revision)
            if not state["proposal"] or state["proposal"]["id"] != pid:
                raise Conflict("Displayed proposal is stale")
            proposal = json.loads(row["body"])
            if not accepted:
                c.execute("UPDATE proposals SET status='declined' WHERE id=?", (pid,))
                state["status"], state["proposal"] = "awaiting_customer", None
                self.event(c, sid, "confirmation_declined", {"proposal_id": pid})
                self.save(c, state)
                return {"status": "declined", "order_changed": False}
            key = state["order_keys"][proposal["order_id"]]
            order_row = c.execute("SELECT * FROM orders WHERE id=?", (key,)).fetchone()
            if order_row["revision"] != proposal["order_revision"]:
                raise Conflict("Order changed after proposal; inspect and propose again")
            order = json.loads(order_row["body"])
            if order["user_id"] != state["user_id"]:
                raise Conflict("Order ownership changed")
            order["status"], order["operation"] = proposal["target_status"], proposal
            c.execute(
                "UPDATE orders SET body=?,revision=revision+1 WHERE id=? AND revision=?",
                (canonical(order), key, proposal["order_revision"]),
            )
            if fault:
                fault()
            receipt = {
                "status": "committed",
                "proposal_id": pid,
                "order_id": proposal["order_id"],
                "order_status": order["status"],
                "operation": proposal,
                "simulated": True,
                "money_moved": False,
                "email_sent": False,
            }
            c.execute(
                "UPDATE proposals SET status='committed',receipt=? WHERE id=?", (canonical(receipt), pid)
            )
            state["status"], state["receipt"] = "completed", receipt
            self.event(c, sid, "operation_committed", receipt)
            self.save(c, state)
        return receipt

    def orders(self, sid):
        state = self.get(sid)
        with self.connect() as c:
            return [
                json.loads(c.execute("SELECT body FROM orders WHERE id=?", (key,)).fetchone()[0])
                for key in state["order_keys"].values()
            ]

    def timeline(self, sid):
        self.get(sid)
        with self.connect() as c:
            return [
                {"id": r["id"], "kind": r["kind"], "created": r["created"], "detail": json.loads(r["body"])}
                for r in c.execute("SELECT * FROM events WHERE session_id=? ORDER BY id", (sid,))
            ]
