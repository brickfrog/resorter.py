# resorter.py

Python implementation(-ish) of gwern's [resorter](https://www.gwern.net/Resorter) for ranking items based on user input.

# Installation

The simplest option is to use the `uv` to install the package as CLI tool, granting you the `resorter` command.

```bash
uv tool install .
```

See the help for more (up-to-date) information about usage.

```bash
Usage: main.py [OPTIONS]

Options:
  --input TEXT                  input file: a CSV file of items to
                                sort: one per line, with up to two
                                columns. (eg. both 'Akira' and
                                'Akira, 10' are valid)  [required]
  --output TEXT                 output file: a file to write the
                                final results to. Default: printing
                                to stdout.
  --queries INTEGER             Maximum number of questions to ask
                                the user; defaults to N*log(N)
                                comparisons.
  --levels INTEGER              The highest level; rated items will
                                be discretized into 1–l levels.
  --quantiles TEXT              What fraction to allocate to each
                                level; space-separated; overrides
                                `--levels`.
  --progress                    Print the mean standard error to
                                stdout
  --save-state TEXT             Save the current state to this file
  --load-state TEXT             Load the previous state from this
                                file
  --min-confidence FLOAT        Minimum confidence level before
                                stopping (0-1)
  --visualize                   Show ASCII visualization of rankings
  --format [csv|json|markdown]  Output format for the rankings
  --help                        Show this message and exit.
```


## Acknowledgments

Thanks to Gwern for the [original concept](https://gwern.net/resorter) and inspiration.

I primarily wrote it to integrate with other python code, I'd recommending looking at hiAndrewQuinn's [repackaging](https://github.com/hiAndrewQuinn/resorter) if you're looking for the original resorter in an easier-to-install package. Python doesn't have the same kind of statistics packages available, so I had to handroll the comparison functionality. I'm not 100% sure I have it correct, but it passes the smell test insofar as rankings.
