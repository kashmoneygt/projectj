## Scope of the work

Do what the request asks, verify it, and stop:
- Don't add new tests, documentation, changelog entries or features that the request didn't ask for. If you think one is worth adding, say so in your final message instead.
- Verify a change with a short throwaway script under /tmp that runs the scenario from the request, then run the existing test file(s) covering the code you changed.
- If an existing test asserts the old behavior that the request asks you to change, update that assertion. Otherwise leave the tests alone.
- If existing tests fail, check whether they also fail without your change before investigating, in one command: `git diff HEAD --quiet || (git stash -q && { <only the failing tests>; git stash pop -q; })`. Leave failures that were already there alone.
