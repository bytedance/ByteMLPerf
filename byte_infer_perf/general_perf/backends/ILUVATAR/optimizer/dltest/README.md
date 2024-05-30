## CI Test tool for IxRT

### 1. Install dltest tool
    
    python setup.py develop

### 2. Usage

#### 2.1 Fetch log

Commmand:

```shell
ixdltest-fetch args_or_pipe ${log_path}
```

Arguments:

- p or patterns, The pattern of fetch log;
- pn or pattern_names, The name of pattern;
- use_re, Whether use regular expression;
- d or nearest_distance, default=10, The nearest distance of matched pattern;
- start_flag, The flag of start to record log;
- end_flag, The flag of stop to record log;
- split_pattern, The pattern is used to match line, If the line is matched, argument `split_sep` to split the line.
- split_sep, The seperator is used to split line;
- split_idx, The index of split line;
- saved, Save result to path;
- log, Log path.

Example
Analyse from file
```
$ ixdltest-fetch run.log -p "Throughput" -t_bi150 Throughput:100 -t_mr100 Throughput:100
{'results': [{'Throughput': [188.5461778786721]}]}
- Check Throughput on BI150 passed (result vs target): 188.5461778786721>=100.0
```

Analyse from command line pipe
```
$ cat run.log | ixdltest-fetch -p "Throughput" -t_bi150 Throughput:100 -t_mr100 Throughput:100
{'results': [{'Throughput': [188.5461778786721]}]}
- Check Throughput on BI150 passed (result vs target): 188.5461778786721>=100.0
```
