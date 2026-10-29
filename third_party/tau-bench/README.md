# Upstream policy attribution

Source: [Sierra Research τ-bench v2.0.0](https://github.com/sierra-research/tau2-bench/releases/tag/v2.0.0), revision `989d3285ae669e49503324bdb9981bf14fccedeb`.

`retail-policy.md` is the upstream retail policy at `data/tau2/domains/retail/policy.md`; the MIT license and copyright notice are preserved. `source.json` records the copied-file digests.

ResolveDesk implements its own local subset: customer lookup, order/product inspection, single-item return/exchange, pending-order cancellation, clarification, proposal, explicit confirmation, and handoff. It does not implement address changes, pending item modifications, voice or multilingual evaluation. Customer lookup by known demo email is not production authentication. No real payments, card credentials, emails, customer accounts or retailer integrations are used.

The official τ-bench task database, runner, user simulator and model judges were **not** run. ResolveDesk's fixtures, scripted customer replies, grading and results are custom; they are not comparable to official leaderboard scores or official versions. Public attribution is not an endorsement by Sierra.
