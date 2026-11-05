# Bash test runs get back the verdict and failing tests their filter cut, and bad test labels explained (port onto the evidence-prompt parent)

hypothesis: The agent runs almost every test through a filter (`2>&1 | tail -3`, `| grep -E "^(Ran|OK|FAILED)"`). The filter often cuts the verdict. Django-style runners print "Destroying test database…" lines after it. sympy's `bin/test` wraps a long "tests finished: …" line in two, so `tail -2` keeps only "in 5.97 seconds / DO *NOT* COMMIT!". Even a visible `FAILED (failures=1)` doesn't say which test failed. So the agent runs the same suite again, often more than once, with a new filter, a longer tail or `grep -B2 -A15 "FAIL:"`, only to see what it already ran. It also guesses labels that name no module, and the runner reports those as ordinary errors. This package adds one bash `tool_call`/`tool_result` hook to the parent, `extensions/test-feedback.ts`, taken from test-runs-show-cut-results-and-bad-label. A `tee` before the first filter keeps the unfiltered output in /tmp. When the filtered result lacks the runner's verdict or the ids of failing tests, the result gets them back along with the log path. Each label of a label-taking runner (`runtests.py`, `-m unittest`) is checked against the files on disk, and a missing label is named along with the closest real ones. I made two fixes for what this parent's runs showed. A summary wrapped over two lines is now read as one, where before it was missed. And `-m …runtests` invocations now count as test runs, as `runtests.py` already did. The parent's prompt, thinking cap and finish review stay as they are. With the outcome of the run it just made in front of it, the agent should skip rerun-only turns, which come late and at high context.

evidence: In the parent's 26 runs, 163 test commands were piped through a filter, and every run had some. 55 of them showed no verdict at all (22 runs), and 38 of those were followed directly by another test run with no edit in between (21 runs, passed and failed alike). About 8 of the 38 were `python -m pytest` on repos without pytest, where the note stays silent.
- django-15987 (failed): at t15, `runtests.py fixtures -v1 | tail -3` showed only three "Destroying" lines. At t16 the agent reran it with a grep and got `FAILED (failures=1)`. At t17 it reran again with `grep -B2 -A15 "FAIL:"` to learn which test it was. The note would have given both at t15.
- sympy-14248 (failed, 141 requests): at t71, `bin/test matrices/ | tail -2` showed only the wrapped summary's tail ("The output got cut off partway through"). It then reran the suite three times, at t72–74, with a different grep each time. It had done the same with `bin/test printing/` at t53–55. The unfixed code missed this wrapped summary. The fixed code returns "tests finished: 627 passed, 1 failed, … 2 exceptions, in 4.21 seconds" and the failing ids.
- Others: django-13028 t23→24, django-15277 t6→7, django-13128 t21→22, django-12273 t35→36, and django-16454 t24→25, where the `admin_scripts` suite was run twice, with a 600 s and then a 900 s timeout.

On the same grandparent, the source variant went from 37 to 34 requests per task, from 21.9k to 18.3k output tokens and from 7.4 to 5.4 minutes, with passes at +2.0 (z 0.58, noise). This parent has not had it.

Replays: I drove the real hooks with a stub `pi` and ran the rewritten commands in bash on a Django-style runner that prints its verdict and then 16 teardown lines.
- `tail -3` came back with "Ran 7 tests / FAILED (failures=1) / FAIL: test_at_tier (…)".
- A grep that kept only the verdict got the failing id and not the verdict a second time.
- A bad class label got the label note and the cut verdict.
- When the verdict was visible, nothing was added. The same held for pytest's visible "no tests ran" and for a visible wrapped sympy summary.
- `${PIPESTATUS[0]}` still held the runner's status.

Live `try-variant` runs:
- After the fixes, `python3 tests/runtests.py 2>&1 | tail -5` showed only teardown lines. The note added "Ran 806 tests … OK", and the agent's next message was "All 806 tests passed", with no rerun.
- In an earlier run, `python3 -m tests.runtests | tail -30` went untapped under the old pattern, and the agent reran it with a grep. That run is what led to the second fix.
- The note stayed silent when `tail -15` already showed "Ran 6 tests / OK", and when pytest's "no tests ran" was visible.
- All 6 runs finished normally with the parent's finish review and an empty pi.err.

predicts: "added the test results a filter cut" should show on about 60–90% of runs, and "explained missing test labels" on about 15–35%, mostly on runtests.py-style repos. Test runs that come right after a filtered run, with no edit in between, should drop well below the parent's 38 in 26 runs. That's about 1–3 fewer requests on runs that test, and more on long Django and sympy runs. Output tokens and minutes should fall by roughly 10–15%, and passes should hold at the parent's level. It's wrong if requests and tokens per run don't fall against the parent, if those reruns stay as common, if the agent spends new turns reading the saved logs, or if passes drop, for example because a real error gets waved off as a label problem.

ports: test-runs-show-cut-results-and-bad-label (which merges explain-missing-test-labels and test-summary-survives-output-filters)
