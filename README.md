# resorter.py

Python implementation(-ish) of gwern's [resorter](https://www.gwern.net/Resorter) for ranking items based on user input using the [Bradley-Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) model.

# Installation

The simplest option is to use [uv](https://github.com/astral-sh/uv) to install the package as CLI tool, granting you the `resorter` command.

```bash
git clone https://github.com/brickfrog/resorter.py
cd resorter.py
uv tool install .
```

See the help for more (up-to-date) information about usage.

```bash
Usage: resorter [OPTIONS]

Options:
  --input TEXT                  input file: a CSV file of items to sort: one
                                per line, with up to two columns. (eg. both
                                'Akira' and 'Akira, 10' are valid)  [required]
  --output TEXT                 output file: a file to write the final results
                                to. Default: printing to stdout.
  --queries INTEGER             Maximum number of questions to ask the user;
                                defaults to N*log(N) comparisons.
  --levels INTEGER              The highest level; rated items will be
                                discretized into 1–l levels.
  --quantiles TEXT              What fraction to allocate to each level;
                                space-separated; overrides `--levels`.
  --progress                    Print the mean uncertainty and model diagnostics 
                                during ranking
  --save-state TEXT             Save the current state to this file
  --load-state TEXT             Load the previous state from this file
  --min-confidence FLOAT        Minimum confidence level before stopping (0-1)
  --confidence-intervals        Include confidence intervals in output
  --diagnostics                 Show model diagnostics (AIC, log-likelihood, etc.)
  --visualize                   Show ASCII visualization of rankings
  --format [csv|json|markdown]  Output format for the rankings
  --help                        Show this message and exit.
```

```bash
uv run resorter --input in.csv

Number of queries: 7
Comparison commands: 1=yes, 2=tied, 3=second is better, p=print estimates, s=skip question, u=undo last comparison, q=quit

Comparison 1/7
Is 'Pinky' better than 'Blinky'? 1

Comparison 2/7
Is 'Inky' better than 'Clyde'? 3

Comparison 3/7
Is 'Pinky' better than 'Clyde'? 1

Comparison 4/7
Is 'Inky' better than 'Blinky'? 2

Comparison 5/7
Is 'Blinky' better than 'Clyde'? 1

Comparison 6/7
Is 'Inky' better than 'Pinky'? 3

Comparison 7/7
Is 'Blinky' better than 'Clyde'? 1
Item,Rank,Confidence,Uncertainty
Pinky,0.5581308097197633,0.37499999996018174,0.5
Blinky,0.15703825844408328,0.4480726715134852,0.5
Clyde,0.1444622888556121,0.4057839646372169,0.5
Inky,0.14036864298054136,0.31250070892832205,0.5
```

## Acknowledgments

Thanks to Gwern for the [original concept](https://gwern.net/resorter) and inspiration. I primarily wrote it to integrate with other python code, I'd recommending looking at hiAndrewQuinn's [repackaging](https://github.com/hiAndrewQuinn/resorter) if you're looking for the original resorter in an easier-to-install package. 

Python doesn't have the same kind of statistics packages available, so I had to handroll the comparison functionality. I'm not 100% sure I have it correct, but it passes the smell test insofar as rankings.
