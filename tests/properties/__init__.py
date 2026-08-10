"""
tests.properties | Layer: TEST
Properties, not examples.

An example test asserts that one input gives one output. A property asserts something
that must hold for EVERY input, and hands the search for a counterexample to a machine
that is better at it than we are. The properties here were chosen because each one, when
violated, fails SILENTLY:

  * `test_conservation`      -- a column that vanishes from the result dict inherits no
                                governance classification, and nothing raises (NM-0005)
  * `test_metamorphic`       -- a ranking that moves with input order, or with an
                                unrelated glossary edit, changes which classification a
                                column inherits without changing any input that matters
  * `test_sync_state_machine`-- an incremental index that drifts from a full rebuild keeps
                                answering queries, from stale vectors
  * `test_incremental_work`  -- a `sync` that re-embeds the whole glossary to apply one
                                edit produces the RIGHT index at the wrong cost, so no
                                assertion about the index can see it

`test_incremental_work` exists because a relation cannot pin an absolute. The first three
files compare one run against another, and two HIGH-severity mutations survived all of
them for the same reason: `sync` decides with `content_hash` and the rebuild oracle
re-derives with `content_hash`, so widening that hash moved both sides together (H-004,
recurring inside the code written to prevent H-004); and a full re-embed lands in exactly
the final state the rebuild oracle demands. Both are caught by counting the texts the
embedding provider was actually handed and comparing them against strings written out by
hand. The same lesson produced `TestFusionActuallyFuses` in `test_metamorphic`.

Every property in here states its exact-vs-tolerance decision in its own docstring, with
the measured number the tolerance was derived from. A tolerance chosen by taste is how a
gate becomes flaky, and a flaky gate is worse than no gate: it teaches people to re-run
red until it goes green.
"""
