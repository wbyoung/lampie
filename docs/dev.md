# Development

## Testing

Running the tests with `pytest-watch` can be helpful. Specifically, re-running the failed tests and updating snapshots each time they're run can be a useful setup.

```bash
uv pip install pytest-watch # install once

ptw --clear -- --lf --snapshot-update
```

Just be sure to review snapshot changes before committing.


#### Log Output

If there's a test for which you want to read the log messages, the current setup can be a little problematic. Unfortunately at the time of writing, the `sugar` plugin seems to duplicate reporting of each test. This results in the same `DEBUG` log messages being printed two times for each test. Seeing the output just once can be done with:

```bash
# no sugar plugin to avoid duplicate output
ptw --clear -- -p no:sugar --lf --snapshot-update

# verbose live logging with the logging plugin disabled to avoid duplication
pytest -vvs -p no:logging
```
