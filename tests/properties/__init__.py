"""
tests.properties | Layer: TEST
Properties, not examples.

An example test asserts that one input gives one output. A property asserts something
that must hold for EVERY input, and hands the search for a counterexample to a machine
that is better at it than we are. The three properties here were chosen because each one,
when violated, fails SILENTLY:

  * `test_conservation`      -- a column that vanishes from the result dict inherits no
                                governance classification, and nothing raises (NM-0005)
  * `test_metamorphic`       -- a ranking that moves with input order, or with an
                                unrelated glossary edit, changes which classification a
                                column inherits without changing any input that matters
  * `test_sync_state_machine`-- an incremental index that drifts from a full rebuild keeps
                                answering queries, from stale vectors

Every property in here states its exact-vs-tolerance decision in its own docstring, with
the measured number the tolerance was derived from. A tolerance chosen by taste is how a
gate becomes flaky, and a flaky gate is worse than no gate: it teaches people to re-run
red until it goes green.
"""
