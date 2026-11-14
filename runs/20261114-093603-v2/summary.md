# Harbor runs on the redesigned tasks: `20261114-093603-v2`

Commit 6154794: free-escape goal; `sensors` profile (`tasks/loophole-alpha`) and `planner` profile
(`tasks/loophole-alpha-planner`). Scores from Harbor's own verifier (separate container, replays the log).

| run | task | score | outcome |
|---|---|---|---|
| oracle | loophole-alpha | 0.851 | solved: free after 59 actions, 0 violations |
| noop | loophole-alpha | 0.000 | stopped, 0 actions |
| stop-early | loophole-alpha | 0.000 | stopped after 6 actions, no held progress |
| oracle | loophole-alpha-planner | 0.851 | solved: free after 59 actions, 0 violations |
