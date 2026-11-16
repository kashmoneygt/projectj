## Working efficiently

Your reasoning costs as much as your actions, so keep it lean:
- Think only as far as the next step needs, then act. Don't draft whole plans, designs or code in your reasoning; write code straight into the edit.
- Don't copy code you have already read into your reasoning; refer to it by file, function or line.
- When you're unsure what code does or returns, or catch yourself going back and forth on a point, read the code or run it and print the actual values instead of tracing or computing them in your head, then judge the output against the requirement.
- Check the names, signatures and behaviour of the project's and libraries' APIs in the installed source before using them, not from memory.
- If something can't be checked here, don't reconstruct it from memory at length: go with what the code, tests, docs and request support, state your assumption, and move on.
- Put independent lookups in one command or one response instead of one per turn.
- If a check fails unexpectedly, re-read the exact code you wrote (including your own test or script) and the full error before forming theories.
- Run slow commands such as test suites once with the output saved to a file (`cmd > /tmp/out.log 2>&1; tail -n 30 /tmp/out.log`), then grep that file instead of re-running with another filter.
- When the task is done and verified, stop and give your answer or a brief summary of the change.
