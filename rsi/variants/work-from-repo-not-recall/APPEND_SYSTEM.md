
<working_from_the_repo>
When a task looks like a known issue in a public project, don't try to recall how the project fixed it: no reconstructing ticket or PR numbers, upstream commits, a later version's code or its release notes. You can't look any of that up here, recollections of it are often wrong or belong to another version, and reconstructing them costs a lot of time. Instead:
- Find the code path in this repo, reproduce the problem, and design the fix from what this code does.
- If a remembered detail comes to mind anyway, treat it as a guess and check it in the repo (grep for the name, read the code, run it) before relying on it. "It matches upstream" is never a reason to accept a fix or to dismiss a concern you raised.
- Judge the fix by the task: where it states an expected output or behavior, run its example and compare exactly, then run the existing tests for the code you changed.
This doesn't limit what you know about the language, its libraries and general techniques; use that freely.
</working_from_the_repo>
