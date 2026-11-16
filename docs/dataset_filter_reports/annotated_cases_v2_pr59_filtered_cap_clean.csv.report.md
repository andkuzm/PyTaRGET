| Step | Cases before -> after | Removed | Repositories | Commit pairs | Max cases / pair | Top-3 repos share |
|---|---:|---:|---:|---:|---:|---:|
| initial | 16914 | | 477 | 3748 | 624 | 8.7% |
| same_pair_error_change_cap | 16914 -> 14414 | -2500 (-14.8%) | 477 | 3748 | 182 | 7.7% |
| environment_errors | 14414 -> 14363 | -51 (-0.3%) | 476 | 3739 | 182 | 7.7% |
| non_repairs | 14363 -> 14090 | -273 (-1.9%) | 476 | 3691 | 182 | 7.8% |

same_pair_error_change_cap: 16914 -> 14414
environment_errors: 14414 -> 14363
non_repairs: 14363 -> 14090

Chain: 16914 -> same_pair_error_change_cap: 14414 -> environment_errors: 14363 -> non_repairs: 14090

`same_pair_error_change_cap`: keep at most 5 cases per group of (commit pair, final error line, change pattern); see the module docstring {"groups": 11089, "groups_with_more_than_cap": 269, "largest_group": 366}

`environment_errors`: drop cases whose failing test log ends in a machine-side error: missing third-party module, network error, missing credentials, missing executable, permission / stdin {"dropped_by_reason": {"network": 12, "missing credentials / API key": 11, "missing third-party module": 11, "missing executable": 10, "permission / stdin": 7}, "file_not_found_kept_not_dropped": 135}

`non_repairs`: drop cases whose repair is not a repair of the code under test: it adds skip/xfail/pass, it mostly deletes the test (repaired block under 30% of the broken one), or it rewrites it (more than 40 changed lines) {"cases_dropped": 273, "matches_per_reason (a case can match several)": {"more than 40 changed lines": 119, "mostly deletion": 113, "adds skip/xfail/pass": 48}}
