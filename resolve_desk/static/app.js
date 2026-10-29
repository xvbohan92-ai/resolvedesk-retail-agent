const $ = (id) => document.getElementById(id);
const el = (tag, text, cls) => {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
};
let selected = null,
  active = null,
  busy = false,
  evaluationVersion = 0;
async function api(path, body) {
  const r = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const d = await r.json();
  if (!r.ok)
    throw new Error(
      typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail),
    );
  return d;
}
function title(s) {
  return s.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
function money(c) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  }).format(c / 100);
}
function pair(parent, name, value) {
  const row = el("div", undefined, "key-value");
  row.append(el("span", name), el("strong", value));
  parent.append(row);
}
function setBusy(value) {
  busy = value;
  if (value) $("status").textContent = "Working…";
  document
    .querySelectorAll("button,select")
    .forEach((n) => (n.disabled = value));
  if (!value) $("start").disabled = !selected;
}
async function action(work) {
  if (busy) return;
  setBusy(true);
  $("error").textContent = "";
  try {
    await work();
  } catch (e) {
    $("error").textContent = e.message;
    $("status").textContent = "Paused";
  } finally {
    setBusy(false);
  }
}
function renderIntent(s, projected) {
  const box = $("intent-summary");
  box.replaceChildren();
  const record = [...s.observations]
    .reverse()
    .find(
      (o) =>
        o.result.resolution_request &&
        (o.result.source_message_count === s.messages.length || projected),
    );
  if (!record) return;
  const request = projected || record.result.resolution_request,
    detail = el("details", undefined, "proposal");
  detail.append(
    el(
      "summary",
      `${s.workflow_version === "v7" ? "Current interpretation" : "Interpreted request"} · ${title(request.operation)}`,
    ),
  );
  for (const [label, key] of [
    ["Replacement option", "requested_option"],
    ["Payment preference", "payment_method"],
    ["Cancellation reason", "reason"],
  ])
    if (request[key] !== "unknown") pair(detail, label, title(request[key]));
  pair(
    detail,
    s.workflow_version === "v7"
      ? "Current fallback prediction"
      : ["v6", "v6_1"].includes(s.workflow_version)
        ? "Initial fallback prediction"
        : "Return fallback interpretation",
    { yes: "Read as allowed", no: "Read as refused", unknown: "Not specified" }[
      request.fallback_return
    ],
  );
  detail.append(
    el(
      "p",
      "Check this model interpretation against your request before confirming an operation.",
      "quiet small",
    ),
    el("blockquote", record.result.evidence),
  );
  if (s.workflow_version === "v7")
    detail.append(
      el(
        "p",
        "This is the current accepted prediction, including any corrections. Raw initial outputs remain in the execution trace.",
        "quiet small",
      ),
    );
  box.append(detail);
}

function render(data) {
  active = data;
  $("request-edit").hidden = true;
  const s = data.session;
  renderIntent(s, data.interpretation);
  $("consent-choice").replaceChildren();
  const pendingConsent =
    s.status === "awaiting_customer" &&
    s.observations.at(-1)?.result.field === "fallback_return" &&
    s.observations.at(-1)?.result.consent_scope;
  $("reply").placeholder = pendingConsent
    ? "Reply YES or NO. Other text requests human assistance."
    : "Provide the requested information, or change your request…";
  if (pendingConsent) {
    const choices = el("div", undefined, "buttons");
    for (const [text, reply] of [
      ["YES · authorize fallback", "YES"],
      ["NO · request human help", "NO"],
    ]) {
      const button = el(
        "button",
        text,
        reply === "YES" ? "primary" : "secondary",
      );
      button.onclick = () =>
        action(async () => {
          const state = active.session;
          render(
            await api(`/api/sessions/${state.id}/messages`, {
              revision: state.revision,
              message: reply,
              provider: $("provider").value,
            }),
          );
        });
      choices.append(button);
    }
    $("consent-choice").append(
      el(
        "p",
        "Choose only for the exact unavailable replacement above. This does not confirm a return operation.",
        "quiet small",
      ),
      choices,
    );
  }
  localStorage.setItem(
    "resolvedesk-last-session",
    JSON.stringify({ id: s.id, provider: $("provider").value }),
  );
  $("status").textContent =
    s.status === "running" ? "Needs review" : title(s.status);
  $("conversation").replaceChildren();
  const dialogue = [{ role: "customer", content: s.messages[0].content }];
  for (const e of data.timeline) {
    if (e.kind === "customer_message")
      dialogue.push({ role: "customer", content: e.detail.content });
    if (
      e.kind === "model_tool" &&
      e.detail.action?.tool === "ask_customer" &&
      !e.detail.result.error &&
      e.detail.result.question
    )
      dialogue.push({ role: "agent", content: e.detail.result.question });
  }
  for (const m of dialogue) {
    const n = el(
      "div",
      undefined,
      m.role === "agent" ? "message agent" : "message",
    );
    n.append(
      el("span", m.role === "agent" ? "RESOLVEDESK" : "CUSTOMER", "who"),
      el("span", m.content),
    );
    $("conversation").append(n);
  }
  if (s.status === "handoff") {
    const o = s.observations.at(-1);
    $("conversation").append(
      el(
        "div",
        `Human review requested. ${o.result.reason || ""}`,
        "message agent",
      ),
    );
  }
  $("start").hidden = true;
  $("reply-form").hidden = ![
    "awaiting_customer",
    "awaiting_confirmation",
    "running",
  ].includes(s.status);
  $("proposal").replaceChildren();
  if (s.status === "awaiting_confirmation" && s.proposal) {
    const p = s.proposal,
      n = el("div", undefined, "proposal");
    n.append(
      el("p", "CUSTOMER CONFIRMATION", "eyebrow"),
      el("h3", `Review proposed ${p.kind}`),
    );
    pair(n, "Order", p.order_id);
    if (p.arguments.reason) pair(n, "Cancellation reason", p.arguments.reason);
    const variants = Object.values(s.products).flatMap((x) => x.variants);
    const option = (id) => variants.find((v) => v.item_id === id)?.option || id;
    if (p.item_id) pair(n, "Current option", option(p.item_id));
    if (p.new_item_id) pair(n, "Requested option", option(p.new_item_id));
    pair(
      n,
      p.kind === "exchange" ? "Price difference" : "Refund amount",
      money(p.amount_cents),
    );
    pair(n, "Payment method", p.payment_method);
    pair(n, "Resulting status", p.target_status);
    n.append(
      el(
        "p",
        "This will update the simulated order only. Confirm these exact details or decline.",
        "quiet",
      ),
    );
    const buttons = el("div", undefined, "buttons");
    const yes = el("button", "Confirm operation ↗", "primary"),
      no = el("button", "Decline", "secondary");
    yes.onclick = () => confirm(true);
    no.onclick = () => confirm(false);
    buttons.append(yes, no);
    n.append(buttons);
    $("proposal").append(n);
  }
  if (s.status === "completed") {
    $("proposal").append(
      el(
        "div",
        `Confirmed. Simulated order is now “${s.receipt.order_status}”. No money moved; no email sent.`,
        "message agent",
      ),
    );
  }
  if (data.confirmation_result?.status === "declined")
    $("conversation").append(
      el(
        "div",
        "Operation declined. Your order is unchanged. You can send a different request.",
        "message agent",
      ),
    );
  if (s.status === "failed")
    $("error").textContent =
      "The model returned an invalid action. The order was not changed. Start a new case to retry.";
  if (s.status === "running")
    $("error").textContent =
      "The bounded tool-call budget was reached. The order remains unconfirmed.";
  $("orders").replaceChildren();
  for (const o of data.orders) {
    const n = el("div");
    n.append(el("div", o.id, "order-id"), el("div", o.status, "order-state"));
    for (const i of o.items)
      n.append(
        el(
          "div",
          `${i.option} · ${money(i.price_cents)} · ${i.item_id}`,
          "order-item",
        ),
      );
    if (o.operation)
      n.append(
        el(
          "div",
          `Operation: ${o.operation.kind} · recorded once`,
          "order-item",
        ),
      );
    $("orders").append(n);
  }
  $("trace-count").textContent = String(s.observations.length);
  $("trace").replaceChildren();
  for (const [i, o] of s.observations.entries()) {
    const n = el("div", undefined, "trace-entry"),
      h = el("div", undefined, "trace-tool");
    h.append(
      el(
        "span",
        `${String(i + 1).padStart(2, "0")} ${o.action?.tool || "Invalid JSON"}`,
      ),
      el("small", `${((o.timing.latency_ms || 0) / 1000).toFixed(2)}s`),
    );
    const details = el("details"),
      summary = el(
        "summary",
        o.result.error
          ? "Rejected · inspect result"
          : "Inspect arguments & result",
      );
    details.append(
      summary,
      el(
        "pre",
        JSON.stringify({ action: o.action, result: o.result }, null, 2),
      ),
    );
    n.append(h, details);
    $("trace").append(n);
  }
  if (s.receipt) {
    const n = el("div", undefined, "trace-entry");
    n.append(
      el("strong", "✓ Explicit confirmation"),
      el(
        "p",
        "Transaction committed once. Retrying the same proposal returns the same receipt.",
        "quiet small",
      ),
    );
    $("trace").append(n);
  }
}
async function confirm(accepted) {
  await action(async () => {
    const s = active.session;
    render(
      await api(`/api/sessions/${s.id}/confirm`, {
        revision: s.revision,
        proposal_id: s.proposal.id,
        accepted,
      }),
    );
  });
}
$("start").onclick = () =>
  action(async () => {
    render(
      await api("/api/sessions", {
        case_id: selected.id,
        message: $("initial-request").value.trim(),
        provider: $("provider").value,
      }),
    );
  });
$("reply-form").onsubmit = (e) => {
  e.preventDefault();
  action(async () => {
    const s = active.session;
    const text = $("reply").value.trim();
    if (!text) return;
    render(
      await api(`/api/sessions/${s.id}/messages`, {
        revision: s.revision,
        message: text,
        provider: $("provider").value,
      }),
    );
    $("reply").value = "";
  });
};
function selectCase(c) {
  $("intent-summary").replaceChildren();
  $("consent-choice").replaceChildren();
  selected = c;
  active = null;
  $("case-title").textContent = title(c.family);
  $("status").textContent = "Ready";
  $("conversation").replaceChildren();
  $("request-edit").hidden = false;
  $("initial-request").value = c.message;
  $("proposal").replaceChildren();
  $("error").textContent = "";
  $("orders").replaceChildren(
    el("p", "Start a case to inspect its isolated order.", "quiet"),
  );
  $("trace").replaceChildren();
  $("trace-count").textContent = "0";
  $("reply-form").hidden = true;
  $("start").hidden = false;
  $("start").disabled = false;
  document
    .querySelectorAll(".case-btn")
    .forEach((b) => b.classList.toggle("active", b.dataset.id === c.id));
}
async function evaluation() {
  const version = ++evaluationVersion,
    experiment = $("experiment").value,
    box = $("evaluation");
  box.replaceChildren(
    el("p", "Loading matching experiment evidence…", "quiet"),
  );
  let d;
  try {
    d = await api(`/api/evidence?experiment=${experiment}`);
  } catch (error) {
    if (version === evaluationVersion)
      box.replaceChildren(el("div", error.message, "panel scope-note"));
    return;
  }
  if (version !== evaluationVersion) return;
  box.replaceChildren();
  if (!d.available) {
    box.append(el("div", d.reason, "panel scope-note"));
    return;
  }
  if (["cuda_eval", "cuda_eval2"].includes(d.experiment)) {
    renderCUDA(d, box);
    return;
  }
  if (d.experiment === "v7") {
    renderV7(d, box);
    return;
  }
  if (d.experiment === "v6_1") {
    renderV61(d, box);
    return;
  }
  if (d.experiment === "v6") {
    renderV6(d, box);
    return;
  }
  if (d.experiment === "v5") {
    renderV5(d, box);
    return;
  }
  if (d.experiment === "v4") {
    renderV4(d, box);
    return;
  }
  if (d.experiment === "v3") {
    renderV3(d, box);
    return;
  }
  if (d.experiment === "v2") {
    renderV2(d, box);
    return;
  }
  const a = d.summary.base,
    b = d.summary.finetuned;
  const grid = el("div", undefined, "eval-grid");
  for (const [label, value, note] of [
    [
      "Verified task outcomes",
      `${a.task_success}/${a.tasks} → ${b.task_success}/${b.tasks}`,
      "Same MLX base, tokenizer, tools and greedy decoding.",
    ],
    [
      "QLoRA parameters",
      `${(d.training.parameters / 1e6).toFixed(2)}M`,
      "Actual weight updates. Checkpoint selected on validation loss.",
    ],
    [
      "Frozen test scope",
      `${b.tasks} cases`,
      "Self-authored English retail scenarios · one trial per case.",
    ],
  ]) {
    const card = el("div", undefined, "panel metric");
    card.append(
      el("span", label, "eyebrow"),
      el("strong", value),
      el("p", note),
    );
    grid.append(card);
  }
  box.append(grid);
  const panel = el("section", undefined, "panel");
  panel.append(
    el(
      "p",
      d.release_gate.promoted
        ? "Candidate promotion check passed."
        : "Candidate not promoted: task success declined, one action was invalid, and rejected write attempts increased.",
      "scope-note",
    ),
  );
  panel.append(el("h2", "Before / after · final business state"));
  const table = el("table", undefined, "eval-table");
  const rows = [
    ["Metric", "Base MLX", "MLX + QLoRA"],
    [
      "Task success",
      `${a.task_success}/${a.tasks}`,
      `${b.task_success}/${b.tasks}`,
    ],
    [
      "Schema-valid actions",
      `${a.schema_valid}/${a.model_calls}`,
      `${b.schema_valid}/${b.model_calls}`,
    ],
    [
      "Required clarifications",
      `${a.clarifications_obtained}/${a.required_clarifications}`,
      `${b.clarifications_obtained}/${b.required_clarifications}`,
    ],
    [
      "Rejected write attempts",
      String(a.rejected_write_attempts),
      String(b.rejected_write_attempts),
    ],
    [
      "Median task time",
      `${(a.latency_p50_ms / 1000).toFixed(2)}s`,
      `${(b.latency_p50_ms / 1000).toFixed(2)}s`,
    ],
  ];
  for (const [i, values] of rows.entries()) {
    const tr = el("tr");
    for (const v of values) tr.append(el(i ? "td" : "th", v));
    table.append(tr);
  }
  panel.append(
    table,
    el(
      "p",
      "Task success requires the correct final state and required customer clarifications. The same deterministic policy and confirmation guards protect both models. Their invariants are engineering behavior, not learned model compliance. Ollama uses JSON Schema decoding and is reported separately as an integration smoke.",
      "scope-note",
    ),
  );
  panel.append(
    el(
      "p",
      "Grading correction: missing-information guesses do not count as customer clarification. Both variants were regraded from saved traces; prompts, checkpoints and cases were unchanged.",
      "scope-note",
    ),
  );
  for (const name of ["evaluation", "training"]) {
    const img = el("img", undefined, "figure");
    img.src = `/api/figure/${name}.png`;
    img.alt =
      name === "evaluation"
        ? "Paired test outcomes and per-family task success"
        : "Validation loss and training checkpoint selection";
    panel.append(img);
  }
  panel.append(el("p", d.scope, "scope-note"), el("h2", "Retained failures"));
  if (!d.failures.length)
    panel.append(
      el(
        "p",
        "No failures in this small fixed test set. This does not establish real-world performance.",
        "scope-note",
      ),
    );
  for (const f of d.failures)
    panel.append(
      el(
        "div",
        `${f.case_id} · base ${f.base ? "pass" : "fail"} / QLoRA ${f.finetuned ? "pass" : "fail"}`,
        "failure",
      ),
    );
  box.append(panel);
}
function renderV2(d, box) {
  const variants = ["legacy_base", "workflow_base", "workflow_qlora_v2"];
  const a = d.summary.workflow_base,
    b = d.summary.workflow_qlora_v2;
  const grid = el("div", undefined, "eval-grid");
  for (const [label, value, note] of [
    [
      "Training comparison",
      `${a.task_success}/${a.tasks} → ${b.task_success}/${b.tasks}`,
      "Same v2 workflow; base weights versus selected QLoRA.",
    ],
    [
      "Exchange scenarios",
      `${a.exchange_related_success}/${a.exchange_related_tasks} → ${b.exchange_related_success}/${b.exchange_related_tasks}`,
      "Includes requested exchanges, authorized fallback and safe handoff.",
    ],
    [
      "Repeated reads executed",
      `${a.duplicate_read_executions} → ${b.duplicate_read_executions}`,
      "Execution guard shared by both v2 models; not learned compliance.",
    ],
  ]) {
    const card = el("div", undefined, "panel metric");
    card.append(
      el("span", label, "eyebrow"),
      el("strong", value),
      el("p", note),
    );
    grid.append(card);
  }
  box.append(grid);
  const panel = el("section", undefined, "panel");
  panel.append(
    el(
      "p",
      d.release_gate.promoted
        ? "Candidate meets the predeclared local promotion rule. This is still a synthetic prototype evaluation."
        : "Candidate did not meet every predeclared promotion condition. Retain the default Ollama integration and inspect remaining failures.",
      "scope-note",
    ),
  );
  panel.append(el("h2", "Three arms · one new frozen test set"));
  const regressions = d.failures.filter((f) => f.base && !f.finetuned);
  if (regressions.length) {
    const before = d.family_success.exchange.workflow_base;
    const after = d.family_success.exchange.workflow_qlora_v2;
    panel.append(
      el(
        "p",
        `Paired regressions remain in ${regressions.length} cases. Ordinary exchanges: ${before.passed}/${before.tasks} → ${after.passed}/${after.tasks}. An aggregate rule pass does not remove these failures. Inspect the per-family results before replacing any model.`,
        "scope-note",
      ),
    );
  }
  const rows = [
    ["Metric", "Legacy + base", "V2 + base", "V2 + QLoRA"],
    [
      "Task success",
      ...variants.map(
        (v) => `${d.summary[v].task_success}/${d.summary[v].tasks}`,
      ),
    ],
    [
      "Ordinary exchange requests",
      ...variants.map(
        (v) =>
          `${d.family_success.exchange[v].passed}/${d.family_success.exchange[v].tasks}`,
      ),
    ],
    [
      "Exchange-related success",
      ...variants.map(
        (v) =>
          `${d.summary[v].exchange_related_success}/${d.summary[v].exchange_related_tasks}`,
      ),
    ],
    [
      "Fulfilled exchange requests",
      ...variants.map(
        (v) =>
          `${d.summary[v].fulfillable_exchange_success}/${d.summary[v].fulfillable_exchange_tasks}`,
      ),
    ],
    [
      "Repeated read attempts",
      ...variants.map((v) => String(d.summary[v].duplicate_read_attempts)),
    ],
    [
      "All identical action repeats",
      ...variants.map((v) => String(d.identical_action_repeats[v])),
    ],
    [
      "Repeated reads executed",
      ...variants.map((v) => String(d.summary[v].duplicate_read_executions)),
    ],
    [
      "Rejected write attempts",
      ...variants.map((v) => String(d.summary[v].rejected_write_attempts)),
    ],
    [
      "Required clarifications",
      ...variants.map(
        (v) =>
          `${d.summary[v].clarifications_obtained}/${d.summary[v].required_clarifications}`,
      ),
    ],
    [
      "Schema-valid actions",
      ...variants.map(
        (v) => `${d.summary[v].schema_valid}/${d.summary[v].model_calls}`,
      ),
    ],
    [
      "Median task time",
      ...variants.map(
        (v) => `${(d.summary[v].latency_p50_ms / 1000).toFixed(2)}s`,
      ),
    ],
  ];
  const wrap = el("div", undefined, "table-wrap"),
    table = el("table", undefined, "eval-table");
  for (const [i, values] of rows.entries()) {
    const row = el("tr");
    for (const value of values) row.append(el(i ? "td" : "th", value));
    table.append(row);
  }
  wrap.append(table);
  panel.append(
    wrap,
    el(
      "p",
      "Legacy → V2 base measures the combined workflow, prompt and catalog changes. V2 base → V2 QLoRA measures training with that workflow held fixed. Round 1 used a different test and is shown only in the archive. Single deterministic trials; latency is descriptive. Ollama JSON-Schema decoding is a separate live integration.",
      "scope-note",
    ),
  );
  for (const [name, alt] of [
    [
      "evaluation",
      "Three-arm outcomes, duplicate executions and paired training results",
    ],
    ["families", "All fourteen scenario families, three cases per family"],
    ["training", "Round two validation loss and selected checkpoint"],
  ]) {
    const img = el("img", undefined, "figure");
    img.src = `/api/figure/${name}.png?experiment=v2`;
    img.alt = alt;
    panel.append(img);
  }
  panel.append(
    el("p", d.scope, "scope-note"),
    el("h2", "Retained training comparison failures"),
  );
  for (const f of d.failures)
    panel.append(
      el(
        "div",
        `${f.case_id} · V2 base ${f.base ? "pass" : "fail"} / QLoRA ${f.finetuned ? "pass" : "fail"}`,
        "failure",
      ),
    );
  if (!d.failures.length)
    panel.append(
      el(
        "p",
        "No failed cases in this small fixed suite. This does not establish production reliability.",
        "scope-note",
      ),
    );
  box.append(panel);
}
function renderCUDA(d, box) {
  const variants = ["predecessor_fp16", "cloud1344_fp16"],
    before = d.summary.predecessor_fp16,
    after = d.summary.cloud1344_fp16,
    fraction = (row, key, denominator) => `${row[key]}/${row[denominator]}`;
  const cards = el("div", undefined, "eval-grid");
  for (const [label, key, denominator, note] of [
    ["Complete tasks", "task_success", "tasks", "Correct outcomes include both approved and declined transactions."],
    ["Explicit grants recognized", "true_permission_yes", "authorized_tasks", "First model output, before any correction."],
    ["Unsupported YES predictions ↓", "unsupported_permission_yes", "non_authorized_tasks", "The denominator contains refused or unspecified fallback permission."],
  ]) {
    const card = el("div", undefined, "panel metric");
    card.append(el("span", label, "eyebrow"), el("strong", `${fraction(before, key, denominator)} → ${fraction(after, key, denominator)}`), el("p", note));
    cards.append(card);
  }
  const panel = el("section", undefined, "panel");
  panel.append(
    el("h2", "Fresh tasks before and after Colab training"),
    el("p", `${d.metadata.test_cases} new authored synthetic requests in separate order worlds. The V6.1 predecessor and fixed 1,344-update adapter use the same FP16 base, workflow, greedy decoding and ${d.metadata.max_total_calls} total model calls per task, including rejected actions. These results are separate from earlier tests.`, "scope-note"),
    el("p", `${d.paired.improved} tasks improved; ${d.paired.regressed} regressed; ${d.paired.both_pass} passed in both arms and ${d.paired.both_fail} failed in both.`, "scope-note"),
  );
  const rows = [["Metric", "Before · V6.1 adapter", "After · Colab 1,344 updates"]];
  for (const [label, key, denominator] of [
    ["Complete tasks", "task_success", "tasks"],
    ["Available exchanges handled correctly", "fulfillable_exchange_success", "fulfillable_exchange_tasks"],
    ["Authorized stockout returns handled correctly", "fallback_success", "fallback_tasks"],
    ["Required human handoffs", "handoff_success", "handoff_tasks"],
    ["Missing-email resolution", "missing_email_success", "missing_email_tasks"],
    ["Explicit grants recognized", "true_permission_yes", "authorized_tasks"],
    ["Unsupported YES predictions ↓", "unsupported_permission_yes", "non_authorized_tasks"],
    ["All raw initial fields correct", "intent_joint_correct", "intent_tasks"],
    ["Initial parse failures ↓", "initial_raw_parse_failures", "intent_tasks"],
    ["Initial invalid-enum tasks ↓", "initial_enum_error_tasks", "intent_tasks"],
    ["Initial reason-enum errors ↓", "initial_reason_enum_error_tasks", "intent_tasks"],
    ["Confirmation preserved", "confirmation_preserved", "tasks"],
  ]) rows.push([label, ...variants.map((variant) => fraction(d.summary[variant], key, denominator))]);
  for (const [label, key] of [
    ["Unnecessary handoffs executed ↓", "unnecessary_handoff_executed"],
    ["Repeated field-question attempts ↓", "repeated_question_attempts"],
    ["Repeated questions delivered ↓", "repeated_questions_delivered"],
    ["All identical-action repeats ↓", "identical_action_repeats"],
    ["Total model calls", "model_calls"],
  ]) rows.push([label, ...variants.map((variant) => String(d.summary[variant][key]))]);
  const wrap = el("div", undefined, "table-wrap"), table = el("table", undefined, "eval-table");
  rows.forEach((values, index) => {
    const row = el("tr");
    values.forEach((value) => row.append(el(index ? "td" : "th", value)));
    table.append(row);
  });
  wrap.append(table);
  panel.append(wrap, el("p", "First-output errors remain counted after model repair. Zero repeated field questions does not mean zero repeated actions. Serial timing includes different generated trajectories and does not establish a hardware speedup.", "scope-note"));
  const protocol = d.protocol_audit, scoped = protocol.summary.cloud1344_fp16;
  panel.append(
    el("h2", "Observed scoped-consent checks"),
    el("p", `On ${scoped.gold_exchange_tasks} exchange-origin tasks: ${scoped.return_proposals_without_scoped_yes} created return proposals and ${scoped.committed_returns_without_scoped_yes} committed returns lacked verified scoped YES; ${scoped.final_returns_without_verified_commit} final return orders lacked a verified scoped commit. These units must not be added together.`, "scope-note"),
    el("p", protocol.candidate_safety_passed ? "No such violations were observed in this small cohort. The known operation-rewrite bypass remains; these observations do not establish production safety." : "An execution-consent violation was observed. Keep this adapter experimental and inspect the retained trajectory.", protocol.candidate_safety_passed ? "scope-note" : "failure"),
  );
  for (const [name, alt] of [
    ["evaluation", "Matched before-and-after business outcomes on fresh synthetic tasks"],
    ["intent", "Raw initial grants, false positives and enum errors"],
    ["recovery", "Repeated questions, repeated actions, calls and latency"],
    ["families", "Paired outcomes across 24 authored scenario families"],
    ["consent", "Scoped consent on exchange-origin return proposals and commits"],
    ["training", "Actual Colab training updates and measured resource use"],
  ]) {
    const image = el("img", undefined, "figure");
    image.src = `/api/figure/${name}.png?experiment=${encodeURIComponent(d.experiment)}`;
    image.alt = alt;
    panel.append(image);
  }
  panel.append(el("h2", "Retained failures"));
  for (const failure of d.failures) panel.append(el("div", `${failure.case_id} · before ${failure.predecessor_fp16 ? "pass" : "fail"} / after ${failure.cloud1344_fp16 ? "pass" : "fail"}`, "failure"));
  if (!d.failures.length) panel.append(el("p", "No failures in this authored suite. This does not measure real customer reliability.", "scope-note"));
  panel.append(el("p", d.scope, "scope-note"), el("p", protocol.limitations, "scope-note"));
  box.append(cards, panel);
}

function renderV7(d, box) {
  if (!d.protocol_audit) {
    box.append(
      el(
        "div",
        "Round 7 supplemental scoped-consent audit is pending. Verified evidence will appear when both business and protocol reports are ready.",
        "panel scope-note",
      ),
    );
    return;
  }
  const variants = [
      "previous_old_flow",
      "previous_new_flow",
      "base_new_flow",
      "qlora_v7",
    ],
    summary = d.primary_summary,
    previous = summary.previous_new_flow,
    current = summary.qlora_v7,
    previousRecovery = d.recovery_diagnostics.previous_new_flow,
    currentRecovery = d.recovery_diagnostics.qlora_v7,
    count = d.primary_test_cases,
    protocol = d.protocol_audit,
    scoped = protocol.summary.qlora_v7;
  const cards = el("div", undefined, "eval-grid");
  for (const [label, value, note] of [
    [
      "Explicit conditional grants recognized",
      `${previous.true_permission_yes}/${previous.authorized_tasks} → ${current.true_permission_yes}/${current.authorized_tasks}`,
      "Raw first model outputs. Later corrections do not replace initial permission predictions.",
    ],
    [
      "Initial reason enum errors ↓",
      `${previousRecovery.initial_reason_enum_error_tasks}/${previous.intent_tasks} → ${currentRecovery.initial_reason_enum_error_tasks}/${current.intent_tasks}`,
      "Requests with an invalid cancellation-reason value in the first output, before any repair.",
    ],
    [
      "Required human handoffs",
      `${previous.handoff_success}/${previous.handoff_tasks} → ${current.handoff_success}/${current.handoff_tasks}`,
      "Correct model-issued escalation after required identity, order and catalog checks.",
    ],
  ]) {
    const card = el("div", undefined, "panel metric");
    card.append(
      el("span", label, "eyebrow"),
      el("strong", value),
      el("p", note),
    );
    cards.append(card);
  }
  box.append(cards);
  const panel = el("section", undefined, "panel");
  panel.append(
    el(
      "p",
      `${count} fresh synthetic tasks, each with its own order world. All four arms use the same requests and twelve total model calls, including rejected and corrective actions. Split-specific authored language and scripted replies limit generalization.`,
      "scope-note",
    ),
    el("h2", "First-output interpretation and eventual task recovery"),
    el(
      "p",
      `Frozen business-gate verdict: ${d.release_gate.promoted ? "pass" : "fail"}. ` +
        (d.release_gate.promoted
          ? "This uses the original scorer and is unchanged. The supplemental protocol audit is assessed separately."
          : "Some original targets remain unmet. This adapter remains an experiment; inspect the failures before adopting it."),
      "scope-note",
    ),
    el(
      "p",
      `First output: ${current.true_permission_yes}/${current.authorized_tasks} explicit grants recognized; ${current.unsupported_permission_yes}/${current.non_authorized_tasks} unsupported affirmative permissions; ${current.initial_raw_parse_failures}/${current.intent_tasks} parse or structural failures. A missing parse is not evidence of correct refusal.`,
      "scope-note",
    ),
  );
  panel.append(
    el("h2", "Supplemental scoped-consent audit"),
    el(
      "p",
      `On ${scoped.gold_exchange_tasks} initially requested exchange tasks: ${scoped.return_proposals_without_scoped_yes} return proposals lacked a verified scoped YES; ${scoped.committed_returns_without_scoped_yes} return commits lacked it; ${scoped.final_returns_without_verified_commit} final return orders lacked a verified scoped commit. These counts can refer to the same transaction and must not be added together.`,
      "scope-note",
    ),
    el(
      "p",
      protocol.candidate_safety_passed
        ? "No created, committed or final-return consent violations were found in this audit cohort. This is a separate small-suite audit, not a certification of production safety."
        : "Supplemental safety limitation: a return proposal, commit or final order lacked verified scoped consent. Do not adopt this candidate despite any passing frozen business verdict. The original business scores and release gate remain unchanged.",
      protocol.candidate_safety_passed ? "scope-note" : "failure",
    ),
  );
  const checks = Object.entries(d.release_gate.checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => title(name));
  if (checks.length)
    panel.append(el("p", "Unmet checks: " + checks.join(" · "), "scope-note"));
  if (current.true_permission_yes < current.authorized_tasks)
    panel.append(
      el(
        "p",
        `Permission recall gap: ${current.authorized_tasks - current.true_permission_yes} explicit grants were missed. Zero unsupported YES predictions alone does not establish correct authorization understanding.`,
        "scope-note",
      ),
    );
  const pairs = d.paired;
  panel.append(
    el(
      "p",
      `Compared with the previous system: ${pairs.end_to_end.improved} tasks improved and ${pairs.end_to_end.regressed} regressed. With the new workflow fixed: ${pairs.training.improved} improved and ${pairs.training.regressed} regressed after training.`,
      "scope-note",
    ),
  );
  const rows = [
    [
      "Metric",
      "V6.1 · original flow",
      "V6.1 · new flow",
      "Base · new flow",
      "V7 · new flow",
    ],
  ];
  for (const [label, key, den] of [
    ["Task success", "task_success", "tasks"],
    [
      "Available-exchange tasks correct",
      "fulfillable_exchange_success",
      "fulfillable_exchange_tasks",
    ],
    ["Authorized stockout fallback", "fallback_success", "fallback_tasks"],
    ["Required model handoff", "handoff_success", "handoff_tasks"],
    [
      "Missing-email tasks correct",
      "missing_email_success",
      "missing_email_tasks",
    ],
    [
      "Explicit grants recognized in first output",
      "true_permission_yes",
      "authorized_tasks",
    ],
    [
      "Unsupported first-output YES ↓",
      "unsupported_permission_yes",
      "non_authorized_tasks",
    ],
    [
      "All seven raw initial fields correct",
      "intent_joint_correct",
      "intent_tasks",
    ],
    [
      "Accepted initial request records",
      "initial_records_valid",
      "intent_tasks",
    ],
    ["Confirmation preserved", "confirmation_preserved", "tasks"],
    [
      "Required clarification obtained",
      "clarifications_obtained",
      "required_clarifications",
    ],
    ["Schema-valid model actions", "schema_valid", "model_calls"],
  ])
    rows.push([
      label,
      ...variants.map((v) => `${summary[v][key]}/${summary[v][den]}`),
    ]);
  rows.push([
    "Initial reason enum-invalid tasks ↓",
    ...variants.map(
      (v) =>
        `${d.recovery_diagnostics[v].initial_reason_enum_error_tasks}/${summary[v].intent_tasks}`,
    ),
  ]);
  for (const [label, key] of [
    ["Initial parse / structural failures ↓", "initial_raw_parse_failures"],
    ["Initial enum-invalid tasks ↓", "initial_enum_error_tasks"],
    ["All reason enum error attempts ↓", "reason_enum_errors"],
    ["All enum error attempts ↓", "enum_error_attempts"],
    ["Rejected request-record attempts", "record_rejections"],
    ["Unnecessary handoffs executed ↓", "unnecessary_handoff_executed"],
    ["Unnecessary model handoff attempts ↓", "unnecessary_handoff_attempts"],
    ["Repeated field attempts ↓", "repeated_question_attempts"],
    ["Repeated questions actually delivered ↓", "repeated_questions_delivered"],
    ["Available-exchange tasks attempting return ↓", "exchange_as_return"],
    [
      "Business: return attempted with another expected outcome ↓",
      "unauthorized_return",
    ],
    ["Backend slot normalizations", "slot_normalizations"],
    ["All identical action repeats", "identical_action_repeats"],
    ["Total model calls including correction", "model_calls"],
  ])
    rows.push([label, ...variants.map((v) => String(summary[v][key]))]);
  rows.push([
    "Tasks with a model-issued enum repair",
    ...variants.map((v) =>
      summary[v].enum_error_tasks
        ? `${summary[v].enum_recovered_tasks}/${summary[v].enum_error_tasks}`
        : "No enum errors",
    ),
  ]);
  rows.push([
    "Supplemental audit: initial exchange cohort",
    ...variants.map(
      (v) =>
        `${protocol.summary[v].gold_exchange_tasks}/${protocol.summary[v].tasks}`,
    ),
  ]);
  for (const [label, key] of [
    [
      "Audit: all raw return attempts without scoped YES",
      "return_attempts_without_scoped_yes",
    ],
    [
      "Audit: rejected raw returns without scoped YES",
      "rejected_return_attempts_without_scoped_yes",
    ],
    [
      "Audit: created return proposals without scoped YES",
      "return_proposals_without_scoped_yes",
    ],
    [
      "Audit: committed returns without scoped YES",
      "committed_returns_without_scoped_yes",
    ],
    [
      "Audit: final returns without verified scoped commit",
      "final_returns_without_verified_commit",
    ],
  ])
    rows.push([
      label,
      ...variants.map((v) => String(protocol.summary[v][key])),
    ]);
  const wrap = el("div", undefined, "table-wrap"),
    table = el("table", undefined, "eval-table");
  rows.forEach((values, index) => {
    const row = el("tr");
    values.forEach((value) => row.append(el(index ? "td" : "th", value)));
    table.append(row);
  });
  wrap.append(table);
  panel.append(
    wrap,
    el(
      "p",
      "Raw intent is scored from the first model emission even if it is rejected or later repaired. Invalid enums receive bounded structured feedback; the model must issue the correction itself. A repaired record can improve eventual completion while its first-output error remains counted.",
      "scope-note",
    ),
    el(
      "p",
      "V6.1 original → new flow holds weights fixed and measures controller/prompt changes. V6.1 → V7 under the same new flow measures the combined new-data, sampling and SFT-objective changes. Deterministic contact/catalog reply validation and actual scoped YES consent are controller behavior, not learned model accuracy.",
      "scope-note",
    ),
  );
  for (const [name, alt] of [
    [
      "evaluation",
      "Four-arm business recovery outcomes and retained protections",
    ],
    ["families", "All 24 scenario families in the new test"],
    ["intent", "Raw first-output intent fields before correction"],
    [
      "consent",
      "Explicit permission recognition and unsupported affirmative predictions",
    ],
    ["recovery", "Initial enum failures versus model-issued corrections"],
    [
      "confusion",
      "Expected versus proposed operations including unfinished tasks",
    ],
    [
      "training",
      "Actual optimizer updates and validation-only checkpoint selection",
    ],
  ]) {
    const image = el("img", undefined, "figure");
    image.src = `/api/figure/${name}.png?experiment=v7`;
    image.alt = alt;
    panel.append(image);
  }
  panel.append(
    el("p", protocol.scope, "scope-note"),
    el("p", protocol.limitations, "scope-note"),
    el("p", d.scope, "scope-note"),
    el("h2", "Retained per-case failures"),
  );
  for (const failure of d.failures)
    panel.append(
      el(
        "div",
        `${failure.case_id} · old ${failure.previous_old_flow ? "pass" : "fail"} / flow ${failure.previous_new_flow ? "pass" : "fail"} / base ${failure.base_new_flow ? "pass" : "fail"} / trained ${failure.qlora_v7 ? "pass" : "fail"}`,
        "failure",
      ),
    );
  if (!d.failures.length)
    panel.append(
      el(
        "p",
        "No failures in this synthetic suite; this is not evidence of production reliability.",
        "scope-note",
      ),
    );
  box.append(panel);
}

function renderV61(d, box) {
  const rawSummary = d.summary;
  d = { ...d, summary: d.primary_summary };
  const variants = [
      "previous_old_flow",
      "previous_new_flow",
      "base_new_flow",
      "qlora_v6_1",
    ],
    previous = d.summary.previous_new_flow,
    current = d.summary.qlora_v6_1;
  const cards = el("div", undefined, "eval-grid");
  for (const [label, value, note] of [
    [
      "Unsupported fallback permission",
      `${previous.unsupported_permission_yes}/${previous.non_authorized_tasks} → ${current.unsupported_permission_yes}/${current.non_authorized_tasks}`,
      "Initial model predictions without authorization. Lower is better; scoped customer consent is separate.",
    ],
    [
      "Required human handoffs",
      `${previous.handoff_success}/${previous.handoff_tasks} → ${current.handoff_success}/${current.handoff_tasks}`,
      "Correct model-issued escalation with identity, order and catalog prerequisites.",
    ],
    [
      "Missing-email tasks correct",
      `${previous.missing_email_success}/${previous.missing_email_tasks} → ${current.missing_email_success}/${current.missing_email_tasks}`,
      "Includes bare addresses, free-text email replies and a subsequent payment question.",
    ],
  ]) {
    const card = el("div", undefined, "panel metric");
    card.append(
      el("span", label, "eyebrow"),
      el("strong", value),
      el("p", note),
    );
    cards.append(card);
  }
  box.append(cards);
  const panel = el("section", undefined, "panel");
  panel.append(
    el(
      "p",
      "Final test: 48 unique interactions. The earlier V6 test plan was retired before inference because it contained aliases and validation still showed clarification failures. This final controller and fresh test were frozen separately, using the same real SFT candidates.",
      "scope-note",
    ),
  );
  panel.append(
    el(
      "p",
      `Across all 48 raw candidate rows: ${rawSummary.qlora_v6_1.unauthorized_return} tasks with unauthorized return attempts and ${rawSummary.qlora_v6_1.repeated_question_attempts} repeated-field attempts.`,
      "scope-note",
    ),
  );
  panel.append(
    el("h2", "Intent extraction, workflow and training measured separately"),
  );
  panel.append(
    el(
      "p",
      d.release_gate.promoted
        ? "The fixed absolute local targets pass. These small synthetic tests do not establish production reliability."
        : "Some fixed absolute targets remain unmet. Inspect the failures before adopting this adapter.",
      "scope-note",
    ),
  );
  const permission = d.permission_diagnostics.qlora_v6_1;
  if (permission.unsupported_affirmative)
    panel.append(
      el(
        "p",
        `Interpretation gap: ${permission.unsupported_affirmative}/${permission.non_authorized_tasks} initial requests were recorded as allowing a return fallback without customer authorization. These remain parser errors even if current stock prevents a wrong return.`,
        "scope-note",
      ),
    );
  if (permission.missed_affirmative)
    panel.append(
      el(
        "p",
        `Permission recall gap: only ${permission.true_affirmative}/${permission.authorized_tasks} explicit grants were recognized. Missing a real grant remains an interpretation error and can cause unnecessary human handoff.`,
        "scope-note",
      ),
    );
  const failed = Object.entries(d.release_gate.checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => title(name));
  if (failed.length)
    panel.append(el("p", "Unmet checks: " + failed.join(" · "), "scope-note"));
  const totalPairs = d.paired.end_to_end,
    trainingPairs = d.paired.training;
  panel.append(
    el(
      "p",
      `Against the prior system: ${totalPairs.improved} tasks improved and ${totalPairs.regressed} regressed. With the typed flow held fixed: ${trainingPairs.improved} improved and ${trainingPairs.regressed} regressed after training. Per-case failures remain below.`,
      "scope-note",
    ),
  );
  const rows = [
    [
      "Metric",
      "V5 · original flow",
      "V5 · scoped flow",
      "Base · scoped flow",
      "V6 · scoped flow",
    ],
  ];
  for (const [label, key, den] of [
    ["Task success", "task_success", "tasks"],
    ["Missing-email tasks", "missing_email_success", "missing_email_tasks"],
    [
      "Available-exchange tasks correct",
      "fulfillable_exchange_success",
      "fulfillable_exchange_tasks",
    ],
    ["Authorized stockout fallback", "fallback_success", "fallback_tasks"],
    ["Required model handoff", "handoff_success", "handoff_tasks"],
    [
      "Exchange tasks with a return attempt ↓",
      "exchange_as_return",
      "fulfillable_exchange_tasks",
    ],
    [
      "Required clarification obtained",
      "clarifications_obtained",
      "required_clarifications",
    ],
    ["Direct-return controls", "direct_return_success", "direct_return_tasks"],
    ["Schema-valid actions", "schema_valid", "model_calls"],
  ])
    rows.push([
      label,
      ...variants.map((v) => `${d.summary[v][key]}/${d.summary[v][den]}`),
    ]);
  for (const [label, key] of [
    ["Unnecessary handoffs executed ↓", "unnecessary_handoff_executed"],
    ["Unnecessary handoff attempts ↓", "unnecessary_handoff_attempts"],
    ["Tasks with unauthorized return attempts ↓", "unauthorized_return"],
    ["Repeated field attempts ↓", "repeated_question_attempts"],
    ["Repeated questions actually delivered ↓", "repeated_questions_delivered"],
    ["Unrequested substitute attempts ↓", "unrequested_substitute_attempts"],
    ["Policy-gate rejections", "policy_gate_rejections"],
    ["Backend slot normalizations", "slot_normalizations"],
    ["All identical action repeats", "identical_action_repeats"],
    ["Total generation calls", "model_calls"],
  ])
    rows.push([label, ...variants.map((v) => String(d.summary[v][key]))]);
  rows.push([
    "Initial intent: all seven fields correct",
    ...variants.map((v) =>
      d.summary[v].intent_tasks
        ? `${d.summary[v].intent_joint_correct}/${d.summary[v].intent_tasks}`
        : "Not recorded",
    ),
  ]);
  for (const [label, key, den] of [
    [
      "Authorized fallback correctly recognized",
      "true_affirmative",
      "authorized_tasks",
    ],
    [
      "Unsupported affirmative fallback permission ↓",
      "unsupported_affirmative",
      "non_authorized_tasks",
    ],
  ])
    rows.push([
      label,
      ...variants.map((v) =>
        d.permission_diagnostics[v]
          ? `${d.permission_diagnostics[v][key]}/${d.permission_diagnostics[v][den]}`
          : "Not recorded",
      ),
    ]);
  const wrap = el("div", undefined, "table-wrap"),
    table = el("table", undefined, "eval-table");
  rows.forEach((values, i) => {
    const row = el("tr");
    values.forEach((value) => row.append(el(i ? "td" : "th", value)));
    table.append(row);
  });
  wrap.append(table);
  panel.append(wrap);
  panel.append(
    el(
      "p",
      "V5 original → scoped flow measures controller and prompt changes with weights fixed. The new flow requires a separate customer YES for a stockout return and validates contact boilerplate and canonical catalog/payment replies. Vague option predictions become unknown with a trace entry. These corrections are not credited as raw model intent accuracy. V5 → V6 under the same new flow measures the combined training-data and permission-loss changes. Controller behavior is not learned model reasoning.",
      "scope-note",
    ),
  );
  panel.append(
    el(
      "p",
      "All arms use twelve total calls on the same 48 unique final test cases. Every rejected action counts; all 192 records are retained. Shared worlds, authored language and scripted replies limit generalization. V6 weights were trained once; validation-driven controller calibration did not update weights.",
      "scope-note",
    ),
  );
  for (const [name, alt] of [
    ["evaluation", "Four-arm recovery outcomes and raw attempts"],
    ["families", "All 24 final scenario families"],
    ["intent", "Seven initial intent fields across all four arms"],
    [
      "consent",
      "Model permission predictions and actual unauthorized return attempts",
    ],
    ["confusion", "Expected versus proposed operations"],
    [
      "training",
      "Actual optimizer loss and validation-only checkpoint selection",
    ],
  ]) {
    const img = el("img", undefined, "figure");
    img.src = `/api/figure/${name}.png?experiment=v6_1`;
    img.alt = alt;
    panel.append(img);
  }
  panel.append(
    el("p", d.scope, "scope-note"),
    el("h2", "Retained per-case failures"),
  );
  d.failures.forEach((f) =>
    panel.append(
      el(
        "div",
        `${f.case_id} · old ${f.previous_old_flow ? "pass" : "fail"} / flow ${f.previous_new_flow ? "pass" : "fail"} / base ${f.base_new_flow ? "pass" : "fail"} / trained ${f.qlora_v6_1 ? "pass" : "fail"}`,
        "failure",
      ),
    ),
  );
  if (!d.failures.length)
    panel.append(
      el(
        "p",
        "No failures in these synthetic tasks; this is not evidence of production reliability.",
        "scope-note",
      ),
    );
  box.append(panel);
}

function renderV6(d, box) {
  const rawSummary = d.summary;
  d = { ...d, summary: d.primary_summary };
  const variants = [
      "previous_old_flow",
      "previous_new_flow",
      "base_new_flow",
      "qlora_v6",
    ],
    previous = d.summary.previous_new_flow,
    current = d.summary.qlora_v6;
  const cards = el("div", undefined, "eval-grid");
  for (const [label, value, note] of [
    [
      "Unsupported fallback permission",
      `${previous.unsupported_permission_yes}/${previous.non_authorized_tasks} → ${current.unsupported_permission_yes}/${current.non_authorized_tasks}`,
      "Initial model predictions without authorization. Lower is better; scoped customer consent is separate.",
    ],
    [
      "Required human handoffs",
      `${previous.handoff_success}/${previous.handoff_tasks} → ${current.handoff_success}/${current.handoff_tasks}`,
      "Correct model-issued escalation with identity, order and catalog prerequisites.",
    ],
    [
      "Missing-email tasks correct",
      `${previous.missing_email_success}/${previous.missing_email_tasks} → ${current.missing_email_success}/${current.missing_email_tasks}`,
      "Includes bare addresses, free-text email replies and a subsequent payment question.",
    ],
  ]) {
    const card = el("div", undefined, "panel metric");
    card.append(
      el("span", label, "eyebrow"),
      el("strong", value),
      el("p", note),
    );
    cards.append(card);
  }
  box.append(cards);
  const panel = el("section", undefined, "panel");
  panel.append(
    el(
      "p",
      "Primary results: 44 unique requests. The frozen experiment contains 48 rows, including four exact aliases; source-first deduplication was fixed before held-out inference. The acceptance gate still uses all 48 raw rows.",
      "scope-note",
    ),
  );
  panel.append(
    el(
      "p",
      `Across all 48 raw candidate rows: ${rawSummary.qlora_v6.unauthorized_return} unauthorized return attempts and ${rawSummary.qlora_v6.repeated_question_attempts} repeated-field attempts.`,
      "scope-note",
    ),
  );
  panel.append(
    el("h2", "Intent extraction, workflow and training measured separately"),
  );
  panel.append(
    el(
      "p",
      d.release_gate.promoted
        ? "The fixed absolute local targets pass. These small synthetic tests do not establish production reliability."
        : "Some fixed absolute targets remain unmet. Inspect the failures before adopting this adapter.",
      "scope-note",
    ),
  );
  const permission = d.permission_diagnostics.qlora_v6;
  if (permission.unsupported_affirmative)
    panel.append(
      el(
        "p",
        `Interpretation gap: ${permission.unsupported_affirmative}/${permission.non_authorized_tasks} initial requests were recorded as allowing a return fallback without customer authorization. These remain parser errors even if current stock prevents a wrong return.`,
        "scope-note",
      ),
    );
  const failed = Object.entries(d.release_gate.checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => title(name));
  if (failed.length)
    panel.append(el("p", "Unmet checks: " + failed.join(" · "), "scope-note"));
  const totalPairs = d.paired.end_to_end,
    trainingPairs = d.paired.training;
  panel.append(
    el(
      "p",
      `Against the prior system: ${totalPairs.improved} tasks improved and ${totalPairs.regressed} regressed. With the typed flow held fixed: ${trainingPairs.improved} improved and ${trainingPairs.regressed} regressed after training. Per-case failures remain below.`,
      "scope-note",
    ),
  );
  const rows = [
    [
      "Metric",
      "V5 · original flow",
      "V5 · scoped flow",
      "Base · scoped flow",
      "V6 · scoped flow",
    ],
  ];
  for (const [label, key, den] of [
    ["Task success", "task_success", "tasks"],
    ["Missing-email tasks", "missing_email_success", "missing_email_tasks"],
    [
      "Available-exchange tasks correct",
      "fulfillable_exchange_success",
      "fulfillable_exchange_tasks",
    ],
    ["Authorized stockout fallback", "fallback_success", "fallback_tasks"],
    ["Required model handoff", "handoff_success", "handoff_tasks"],
    [
      "Exchange incorrectly changed to return ↓",
      "exchange_as_return",
      "fulfillable_exchange_tasks",
    ],
    [
      "Required clarification obtained",
      "clarifications_obtained",
      "required_clarifications",
    ],
    ["Direct-return controls", "direct_return_success", "direct_return_tasks"],
    ["Schema-valid actions", "schema_valid", "model_calls"],
  ])
    rows.push([
      label,
      ...variants.map((v) => `${d.summary[v][key]}/${d.summary[v][den]}`),
    ]);
  for (const [label, key] of [
    ["Unnecessary handoffs executed ↓", "unnecessary_handoff_executed"],
    ["Unnecessary handoff attempts ↓", "unnecessary_handoff_attempts"],
    ["Unauthorized return attempts ↓", "unauthorized_return"],
    ["Repeated field attempts ↓", "repeated_question_attempts"],
    ["Repeated questions actually delivered ↓", "repeated_questions_delivered"],
    ["Unrequested substitute attempts ↓", "unrequested_substitute_attempts"],
    ["Policy-gate rejections", "policy_gate_rejections"],
    ["All identical action repeats", "identical_action_repeats"],
    ["Total generation calls", "model_calls"],
  ])
    rows.push([label, ...variants.map((v) => String(d.summary[v][key]))]);
  rows.push([
    "Initial intent: all seven fields correct",
    ...variants.map((v) =>
      d.summary[v].intent_tasks
        ? `${d.summary[v].intent_joint_correct}/${d.summary[v].intent_tasks}`
        : "Not recorded",
    ),
  ]);
  for (const [label, key, den] of [
    [
      "Authorized fallback correctly recognized",
      "true_affirmative",
      "authorized_tasks",
    ],
    [
      "Unsupported affirmative fallback permission ↓",
      "unsupported_affirmative",
      "non_authorized_tasks",
    ],
  ])
    rows.push([
      label,
      ...variants.map((v) =>
        d.permission_diagnostics[v]
          ? `${d.permission_diagnostics[v][key]}/${d.permission_diagnostics[v][den]}`
          : "Not recorded",
      ),
    ]);
  const wrap = el("div", undefined, "table-wrap"),
    table = el("table", undefined, "eval-table");
  rows.forEach((values, i) => {
    const row = el("tr");
    values.forEach((value) => row.append(el(i ? "td" : "th", value)));
    table.append(row);
  });
  wrap.append(table);
  panel.append(wrap);
  panel.append(
    el(
      "p",
      "V5 original → scoped flow measures controller and prompt changes with weights fixed. The new flow requires a separate customer YES for a stockout return and projects only literal single-address email replies. Free-text email replies still require a model update with latest evidence. V5 → V6 under the same new flow measures the combined training-data and permission-loss changes. Controller behavior is not learned model reasoning.",
      "scope-note",
    ),
  );
  panel.append(
    el(
      "p",
      "All arms use twelve total calls per row. Primary comparisons use 44 unique requests; all 192 raw four-arm records remain retained, including the four duplicate requests per arm. Historical V5 tests are development evidence, never pooled. Shared worlds, authored language and scripted replies limit generalization.",
      "scope-note",
    ),
  );
  for (const [name, alt] of [
    ["evaluation", "Four-arm recovery outcomes and raw attempts"],
    ["families", "All 24 raw scenario labels, including duplicate aliases"],
    ["intent", "Seven initial intent fields across all four arms"],
    [
      "consent",
      "Model permission predictions and actual unauthorized return attempts",
    ],
    ["confusion", "Expected versus proposed operations"],
    [
      "training",
      "Actual optimizer loss and validation-only checkpoint selection",
    ],
  ]) {
    const img = el("img", undefined, "figure");
    img.src = `/api/figure/${name}.png?experiment=v6`;
    img.alt = alt;
    panel.append(img);
  }
  panel.append(
    el("p", d.scope, "scope-note"),
    el("h2", "Retained per-case failures"),
  );
  d.failures.forEach((f) =>
    panel.append(
      el(
        "div",
        `${f.case_id} · old ${f.previous_old_flow ? "pass" : "fail"} / flow ${f.previous_new_flow ? "pass" : "fail"} / base ${f.base_new_flow ? "pass" : "fail"} / trained ${f.qlora_v6 ? "pass" : "fail"}`,
        "failure",
      ),
    ),
  );
  if (!d.failures.length)
    panel.append(
      el(
        "p",
        "No failures in these synthetic tasks; this is not evidence of production reliability.",
        "scope-note",
      ),
    );
  box.append(panel);
}

function renderV5(d, box) {
  const variants = [
      "previous_old_flow",
      "previous_new_flow",
      "base_new_flow",
      "qlora_v5",
    ],
    previous = d.summary.previous_new_flow,
    current = d.summary.qlora_v5;
  const cards = el("div", undefined, "eval-grid");
  for (const [label, value, note] of [
    [
      "Authorized stockout fallback",
      `${previous.fallback_success}/${previous.fallback_tasks} → ${current.fallback_success}/${current.fallback_tasks}`,
      "Same typed-intent flow; previous versus newly trained weights.",
    ],
    [
      "Unnecessary human handoffs",
      `${previous.unnecessary_handoff_executed} → ${current.unnecessary_handoff_executed}`,
      "Executed handoffs on 24 tasks that should be resolved without a person. Attempts are also reported.",
    ],
    [
      "Available exchanges correct",
      `${previous.fulfillable_exchange_success}/${previous.fulfillable_exchange_tasks} → ${current.fulfillable_exchange_success}/${current.fulfillable_exchange_tasks}`,
      "Exact requested variant, payment, amount, prerequisites and confirmation.",
    ],
  ]) {
    const card = el("div", undefined, "panel metric");
    card.append(
      el("span", label, "eyebrow"),
      el("strong", value),
      el("p", note),
    );
    cards.append(card);
  }
  box.append(cards);
  const panel = el("section", undefined, "panel");
  panel.append(
    el("h2", "Intent extraction, workflow and training measured separately"),
  );
  panel.append(
    el(
      "p",
      d.release_gate.promoted
        ? "The fixed absolute local targets pass. These small synthetic tests do not establish production reliability."
        : "Some fixed absolute targets remain unmet. Inspect the failures before adopting this adapter.",
      "scope-note",
    ),
  );
  const permission = d.permission_diagnostics.qlora_v5;
  if (permission.unsupported_affirmative)
    panel.append(
      el(
        "p",
        `Interpretation gap: ${permission.unsupported_affirmative}/${permission.non_authorized_tasks} initial requests were recorded as allowing a return fallback without customer authorization. These remain parser errors even if current stock prevents a wrong return.`,
        "scope-note",
      ),
    );
  const failed = Object.entries(d.release_gate.checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => title(name));
  if (failed.length)
    panel.append(el("p", "Unmet checks: " + failed.join(" · "), "scope-note"));
  const totalPairs = d.paired.end_to_end,
    trainingPairs = d.paired.training;
  panel.append(
    el(
      "p",
      `Against the prior system: ${totalPairs.improved} tasks improved and ${totalPairs.regressed} regressed. With the typed flow held fixed: ${trainingPairs.improved} improved and ${trainingPairs.regressed} regressed after training. Per-case failures remain below.`,
      "scope-note",
    ),
  );
  const rows = [
    [
      "Metric",
      "V4 · old flow",
      "V4 · typed flow",
      "Base · typed flow",
      "V5 · typed flow",
    ],
  ];
  for (const [label, key, den] of [
    ["Task success", "task_success", "tasks"],
    [
      "Available-exchange tasks correct",
      "fulfillable_exchange_success",
      "fulfillable_exchange_tasks",
    ],
    ["Authorized stockout fallback", "fallback_success", "fallback_tasks"],
    ["Required model handoff", "handoff_success", "handoff_tasks"],
    [
      "Exchange incorrectly changed to return ↓",
      "exchange_as_return",
      "fulfillable_exchange_tasks",
    ],
    [
      "Required clarification obtained",
      "clarifications_obtained",
      "required_clarifications",
    ],
    ["Direct-return controls", "direct_return_success", "direct_return_tasks"],
    ["Schema-valid actions", "schema_valid", "model_calls"],
  ])
    rows.push([
      label,
      ...variants.map((v) => `${d.summary[v][key]}/${d.summary[v][den]}`),
    ]);
  for (const [label, key] of [
    ["Unnecessary handoffs executed ↓", "unnecessary_handoff_executed"],
    ["Unnecessary handoff attempts ↓", "unnecessary_handoff_attempts"],
    ["Unauthorized return attempts ↓", "unauthorized_return"],
    ["Repeated field attempts ↓", "repeated_question_attempts"],
    ["Repeated questions actually delivered ↓", "repeated_questions_delivered"],
    ["Unrequested substitute attempts ↓", "unrequested_substitute_attempts"],
    ["Policy-gate rejections", "policy_gate_rejections"],
    ["All identical action repeats", "identical_action_repeats"],
    ["Total generation calls", "model_calls"],
  ])
    rows.push([label, ...variants.map((v) => String(d.summary[v][key]))]);
  rows.push([
    "Initial intent: all seven fields correct",
    ...variants.map((v) =>
      d.summary[v].intent_tasks
        ? `${d.summary[v].intent_joint_correct}/${d.summary[v].intent_tasks}`
        : "Not recorded",
    ),
  ]);
  for (const [label, key, den] of [
    [
      "Authorized fallback correctly recognized",
      "true_affirmative",
      "authorized_tasks",
    ],
    [
      "Unsupported affirmative fallback permission ↓",
      "unsupported_affirmative",
      "non_authorized_tasks",
    ],
  ])
    rows.push([
      label,
      ...variants.map((v) =>
        d.permission_diagnostics[v]
          ? `${d.permission_diagnostics[v][key]}/${d.permission_diagnostics[v][den]}`
          : "Not recorded",
      ),
    ]);
  const wrap = el("div", undefined, "table-wrap"),
    table = el("table", undefined, "eval-table");
  rows.forEach((values, i) => {
    const row = el("tr");
    values.forEach((value) => row.append(el(i ? "td" : "th", value)));
    table.append(row);
  });
  wrap.append(table);
  panel.append(wrap);
  panel.append(
    el(
      "p",
      "Old → typed flow includes a new model-extracted request record and a deterministic policy gate over observed tools. The gate constrains legal actions; it does not read benchmark labels or certify the meaning of a quotation. The V4 → V5 comparison on the identical typed flow measures bundled SFT of intent extraction and tool following. Do not attribute controller policy to learned model reasoning.",
      "scope-note",
    ),
  );
  panel.append(
    el(
      "p",
      "All four arms use twelve total model calls, including intent recording and every rejected attempt. Historical round-four scores used eight calls and a different test. One greedy trial per task, scripted customer replies and correlated synthetic worlds. Timing is descriptive only; background models are left untouched.",
      "scope-note",
    ),
  );
  for (const [name, alt] of [
    ["evaluation", "Four-arm recovery outcomes and raw attempts"],
    ["families", "All sixteen scenario families"],
    ["intent", "Seven initial intent fields across three weight variants"],
    ["confusion", "Expected versus proposed operations"],
    [
      "training",
      "Actual optimizer loss and validation-only checkpoint selection",
    ],
  ]) {
    const img = el("img", undefined, "figure");
    img.src = `/api/figure/${name}.png?experiment=v5`;
    img.alt = alt;
    panel.append(img);
  }
  panel.append(
    el("p", d.scope, "scope-note"),
    el("h2", "Retained per-case failures"),
  );
  d.failures.forEach((f) =>
    panel.append(
      el(
        "div",
        `${f.case_id} · old ${f.previous_old_flow ? "pass" : "fail"} / flow ${f.previous_new_flow ? "pass" : "fail"} / base ${f.base_new_flow ? "pass" : "fail"} / trained ${f.qlora_v5 ? "pass" : "fail"}`,
        "failure",
      ),
    ),
  );
  if (!d.failures.length)
    panel.append(
      el(
        "p",
        "No failures in these 32 synthetic tasks; this is not evidence of production reliability.",
        "scope-note",
      ),
    );
  box.append(panel);
}

function renderV4(d, box) {
  const variants = [
    "previous_old_flow",
    "previous_new_flow",
    "base_new_flow",
    "qlora_v4",
  ];
  const old = d.summary.previous_new_flow,
    current = d.summary.qlora_v4;
  const grid = el("div", undefined, "eval-grid");
  for (const [label, value, note] of [
    [
      "Repeated field attempts",
      `${old.repeated_question_attempts} → ${current.repeated_question_attempts}`,
      "Raw model attempts, including questions blocked from being sent again.",
    ],
    [
      "Fallback / handoff tasks correct",
      `${current.fallback_success}/${current.fallback_tasks} · ${current.handoff_success}/${current.handoff_tasks}`,
      "Correct fallback outcome, including expected declines; handoff requires a model handoff action.",
    ],
    [
      "Exchange → return error",
      `${old.exchange_as_return}/${old.fulfillable_exchange_tasks} → ${current.exchange_as_return}/${current.fulfillable_exchange_tasks}`,
      "Any return attempt on an available-exchange task. Lower is better.",
    ],
  ]) {
    const card = el("div", undefined, "panel metric");
    card.append(
      el("span", label, "eyebrow"),
      el("strong", value),
      el("p", note),
    );
    grid.append(card);
  }
  box.append(grid);
  const panel = el("section", undefined, "panel");
  panel.append(
    el(
      "p",
      d.release_gate.promoted
        ? "Candidate meets the fixed local prototype checks. This synthetic suite does not establish production reliability."
        : "Candidate does not meet every fixed local check. Retain the default integration and inspect regressions.",
      "scope-note",
    ),
  );
  panel.append(
    el("h2", "Workflow effects and training effects, measured separately"),
  );
  if (current.fallback_success < current.fallback_tasks)
    panel.append(
      el(
        "p",
        `Authorized stockout fallback remains incomplete: ${current.fallback_success}/${current.fallback_tasks}. Meeting relative checks does not mean this target is fixed.`,
        "scope-note",
      ),
    );
  const rows = [
    [
      "Metric",
      "V3 · old flow",
      "V3 · new flow",
      "Base · new flow",
      "V4 · new flow",
    ],
  ];
  for (const [label, key, den] of [
    ["Task success", "task_success", "tasks"],
    [
      "Available-exchange tasks correct",
      "fulfillable_exchange_success",
      "fulfillable_exchange_tasks",
    ],
    ["Authorized out-of-stock fallback", "fallback_success", "fallback_tasks"],
    ["Expected handoff completed by model", "handoff_success", "handoff_tasks"],
    [
      "Exchange incorrectly changed to return ↓",
      "exchange_as_return",
      "fulfillable_exchange_tasks",
    ],
    [
      "Direct-return tasks correct",
      "direct_return_success",
      "direct_return_tasks",
    ],
    [
      "Required clarification obtained",
      "clarifications_obtained",
      "required_clarifications",
    ],
    ["Schema-valid actions", "schema_valid", "model_calls"],
  ])
    rows.push([
      label,
      ...variants.map((v) => `${d.summary[v][key]}/${d.summary[v][den]}`),
    ]);
  for (const [label, key] of [
    ["Repeated field attempts", "repeated_question_attempts"],
    [
      "Repeated field questions actually delivered",
      "repeated_questions_delivered",
    ],
    ["All identical action repeats", "identical_action_repeats"],
    ["Model calls", "model_calls"],
    ["Unauthorized return attempts", "unauthorized_return"],
    ["Rejected write attempts", "rejected_write_attempts"],
  ])
    rows.push([label, ...variants.map((v) => String(d.summary[v][key]))]);
  rows.push([
    "Verified handoffs / model handoffs",
    ...variants.map(
      (v) =>
        `${d.handoff_diagnostics[v].verified_handoffs}/${d.handoff_diagnostics[v].model_handoffs}`,
    ),
  ]);
  rows.push([
    "Unnecessary model handoffs ↓",
    ...variants.map(
      (v) =>
        `${d.handoff_diagnostics[v].unnecessary_handoffs}/${d.handoff_diagnostics[v].non_handoff_tasks}`,
    ),
  ]);
  const wrap = el("div", undefined, "table-wrap"),
    table = el("table", undefined, "eval-table");
  for (const [i, values] of rows.entries()) {
    const row = el("tr");
    for (const value of values) row.append(el(i ? "td" : "th", value));
    table.append(row);
  }
  wrap.append(table);
  panel.append(wrap);
  panel.append(
    el(
      "p",
      "V3 old → new flow measures memory, duplicate-question guards, policy guidance and observed affordability facts. V3 → V4 on the new flow measures the revised training recipe: 480 SFT updates, 136 action-preference/NLL updates and 80 recovery-state updates, followed by a 48-update calibration trial chosen only if validation regression guards hold. Validation exposed over-escalation from unregularized sequence preference; all revisions preceded new test inference. Final weight choice uses validation regression guards. Test results did not select weights.",
      "scope-note",
    ),
  );
  panel.append(
    el(
      "p",
      "Exact duplicate questions with an existing reply are blocked by code; model attempts remain counted. Different precise followups remain possible, and new unsolicited instructions reset old clarification context. All test arms use the same eight-total-call budget and greedy unconstrained generation. These are fresh synthetic correlated tasks with scripted replies, not human customer judgments.",
      "scope-note",
    ),
  );
  for (const [name, alt] of [
    ["evaluation", "Four-arm outcomes and question attempts"],
    ["families", "All sixteen task families"],
    ["confusion", "Expected versus proposed business operations"],
    ["training", "SFT loss, calibration loss and pre-test validation"],
  ]) {
    const img = el("img", undefined, "figure");
    img.src = `/api/figure/${name}.png?experiment=v4`;
    img.alt = alt;
    panel.append(img);
  }
  panel.append(
    el("p", d.scope, "scope-note"),
    el("h2", "Retained per-case failures"),
  );
  for (const f of d.failures)
    panel.append(
      el(
        "div",
        `${f.case_id} · old ${f.previous_old_flow ? "pass" : "fail"} / flow ${f.previous_new_flow ? "pass" : "fail"} / base ${f.base_new_flow ? "pass" : "fail"} / trained ${f.qlora_v4 ? "pass" : "fail"}`,
        "failure",
      ),
    );
  box.append(panel);
}

function renderV3(d, box) {
  const variants = ["base", "qlora_v2", "qlora_v3"];
  const old = d.summary.qlora_v2,
    current = d.summary.qlora_v3;
  const grid = el("div", undefined, "eval-grid");
  for (const [label, value, note] of [
    [
      "Exchange → return error",
      `${old.exchange_as_return}/${old.fulfillable_exchange_tasks} → ${current.exchange_as_return}/${current.fulfillable_exchange_tasks}`,
      "Any return proposal on an available-exchange request. Lower is better.",
    ],
    [
      "Available exchanges completed",
      `${old.fulfillable_exchange_success}/${old.fulfillable_exchange_tasks} → ${current.fulfillable_exchange_success}/${current.fulfillable_exchange_tasks}`,
      "Correct operation, variant, payment, amount and prerequisites.",
    ],
    [
      "All test tasks",
      `${old.task_success}/${old.tasks} → ${current.task_success}/${current.tasks}`,
      "Same controller and prompts for all three weight variants.",
    ],
  ]) {
    const card = el("div", undefined, "panel metric");
    card.append(
      el("span", label, "eyebrow"),
      el("strong", value),
      el("p", note),
    );
    grid.append(card);
  }
  box.append(grid);
  const panel = el("section", undefined, "panel");
  panel.append(
    el(
      "p",
      d.release_gate.promoted
        ? "Candidate meets the fixed local release criterion on this synthetic suite. This does not establish production reliability."
        : "Candidate does not meet every fixed release condition. Inspect failed cases before replacing a model.",
      "scope-note",
    ),
  );
  panel.append(el("h2", "Three weights · identical retail workflow"));
  const rows = [["Metric", "Base MLX", "Previous QLoRA", "New QLoRA"]];
  for (const [label, key, den] of [
    ["Task success", "task_success", "tasks"],
    [
      "Exchange incorrectly changed to return ↓",
      "exchange_as_return",
      "fulfillable_exchange_tasks",
    ],
    [
      "Available exchanges completed",
      "fulfillable_exchange_success",
      "fulfillable_exchange_tasks",
    ],
    [
      "Ordinary exchanges",
      "ordinary_exchange_success",
      "ordinary_exchange_tasks",
    ],
    [
      "Direct returns completed",
      "direct_return_success",
      "direct_return_tasks",
    ],
    [
      "Required clarifications",
      "clarifications_obtained",
      "required_clarifications",
    ],
    ["Schema-valid actions", "schema_valid", "model_calls"],
  ])
    rows.push([
      label,
      ...variants.map((v) => `${d.summary[v][key]}/${d.summary[v][den]}`),
    ]);
  for (const [label, key] of [
    ["Unauthorized return proposals", "unauthorized_return"],
    ["Repeated business reads executed", "duplicate_read_executions"],
    ["Rejected write attempts", "rejected_write_attempts"],
    ["Model calls", "model_calls"],
  ])
    rows.push([label, ...variants.map((v) => String(d.summary[v][key]))]);
  rows.push([
    "All identical action repeats",
    ...variants.map((v) => String(d.identical_action_repeats[v])),
  ]);
  rows.push([
    "Median local task time",
    ...variants.map(
      (v) => `${(d.summary[v].latency_p50_ms / 1000).toFixed(2)}s`,
    ),
  ]);
  const wrap = el("div", undefined, "table-wrap"),
    table = el("table", undefined, "eval-table");
  for (const [i, values] of rows.entries()) {
    const row = el("tr");
    for (const value of values) row.append(el(i ? "td" : "th", value));
    table.append(row);
  }
  wrap.append(table);
  panel.append(
    wrap,
    el(
      "p",
      `Checkpoint ${d.training.selected.iteration} was selected using validation business outcomes. Test data was not used for selection. The new recipe combines counterfactual cases, tool-name token weighting and business-based selection; it does not isolate those changes individually.`,
      "scope-note",
    ),
  );
  panel.append(
    el(
      "p",
      "All 48 tasks use fresh split-specific worlds/phrasing. Counterfactual pairs share worlds within the test, so observations are correlated. Scripted field-based replies and single greedy trials do not measure human customer satisfaction. Ollama remains a separate live integration; archived scores use different tests.",
      "scope-note",
    ),
  );
  for (const [name, alt] of [
    ["evaluation", "New test task completion and exchange-to-return errors"],
    [
      "confusion",
      "Expected versus final proposed operation for all three models",
    ],
    ["families", "All sixteen scenario families"],
    [
      "training",
      "Validation task-based checkpoint selection and weighted loss",
    ],
  ]) {
    const img = el("img", undefined, "figure");
    img.src = `/api/figure/${name}.png?experiment=v3`;
    img.alt = alt;
    panel.append(img);
  }
  panel.append(
    el("p", d.scope, "scope-note"),
    el("h2", "Retained failures and paired outcomes"),
  );
  for (const f of d.failures)
    panel.append(
      el(
        "div",
        `${f.case_id} · base ${f.base ? "pass" : "fail"} / previous ${f.previous ? "pass" : "fail"} / new ${f.new ? "pass" : "fail"}`,
        "failure",
      ),
    );
  if (!d.failures.length)
    panel.append(
      el(
        "p",
        "No failed cases in this small fixed suite. This does not establish general reliability.",
        "scope-note",
      ),
    );
  box.append(panel);
}
$("experiment").onchange = () =>
  evaluation().catch((e) =>
    $("evaluation").replaceChildren(el("p", e.message)),
  );
$("nav-work").onclick = () => {
  $("work-page").hidden = false;
  $("eval-page").hidden = true;
  $("nav-work").classList.add("active");
  $("nav-eval").classList.remove("active");
  window.scrollTo(0, 0);
};
$("nav-eval").onclick = () => {
  $("work-page").hidden = true;
  $("eval-page").hidden = false;
  $("nav-eval").classList.add("active");
  $("nav-work").classList.remove("active");
  window.scrollTo(0, 0);
  evaluation().catch((e) =>
    $("evaluation").replaceChildren(el("p", e.message, "scope-note")),
  );
};
$("provider").onchange = () => {
  $("model-note").hidden = $("provider").value !== "finetuned";
};
async function init() {
  try {
    const cases = await api("/api/cases");
    const prefix = cases.some((c) => c.id.startsWith("v7-"))
      ? "v7-"
      : cases.some((c) => c.id.startsWith("v6_1-"))
        ? "v6_1-"
        : cases.some((c) => c.id.startsWith("v6-"))
          ? "v6-"
          : cases.some((c) => c.id.startsWith("v5-"))
            ? "v5-"
            : cases.some((c) => c.id.startsWith("v4-"))
              ? "v4-"
              : cases.some((c) => c.id.startsWith("v3-"))
                ? "v3-"
                : cases.some((c) => c.id.startsWith("v2-"))
                  ? "v2-"
                  : "";
    const shown = cases.filter(
      (c) => c.id.endsWith("-00") && c.id.startsWith(prefix),
    );
    $("case-count").textContent = String(shown.length);
    for (const c of shown) {
      const b = el("button", undefined, "case-btn");
      b.dataset.id = c.id;
      b.append(
        el("strong", title(c.family)),
        el("small", "ENGLISH · VALIDATION"),
      );
      b.onclick = () => selectCase(c);
      $("cases").append(b);
    }
    selectCase(shown.find((c) => c.family === "exchange") || shown[0]);
    const h = await api("/api/health");
    const ollamaReady = h.models.ollama.available,
      trainedReady = h.models.finetuned.available;
    $("health").textContent = trainedReady
      ? ollamaReady
        ? "Ollama · QLoRA V7 connected"
        : "QLoRA V7 connected"
      : ollamaReady
        ? "Qwen3 · Ollama connected"
        : "Local models unavailable";
    $("health").classList.toggle("ready", ollamaReady || trainedReady);
    $("provider").querySelector("[value=finetuned]").disabled =
      !h.models.finetuned.available;
    const saved = JSON.parse(
      localStorage.getItem("resolvedesk-last-session") || "null",
    );
    if (saved) {
      try {
        const data = await api(`/api/sessions/${saved.id}`);
        const c = cases.find((x) => x.id === data.session.case_id);
        if (c) {
          selectCase(c);
          $("provider").value = saved.provider;
          $("model-note").hidden = saved.provider !== "finetuned";
          render(data);
        }
      } catch {
        localStorage.removeItem("resolvedesk-last-session");
      }
    }
  } catch (e) {
    $("error").textContent = e.message;
  }
}
init();
