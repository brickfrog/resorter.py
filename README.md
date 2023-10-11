# resorter.py

Python implementation of gwern's resorter, a mechanism using Bradley-Terry models to rank items (originally meant for media).

I'm reccomend looking at hiAndrewQuinn's [implementation](https://github.com/hiAndrewQuinn/resorter) if you're looking for a more 1:1 translation that's easy to install and use, or simply read gwern's [original article](https://www.gwern.net/Resorter) if you're interested in the theory behind it. Python doesn't have the same kind of statistics packages available, so I had to handroll the comparison functionality, I'm not 100% sure I have it correct but it passes the smell test insofar as rankings.

This is primarily for integrating with the rest of the python ecosystem, for example [streamlit](https://streamlit.io/), where I plan to pipe my own media rankings and so on and so forth.
