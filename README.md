# resorter.py

Python implementation of gwern's [resorter](https://www.gwern.net/Resorter) for ranking items based on user input using Bradley-Terry models.

I'm recommend looking at hiAndrewQuinn's [repackaging](https://github.com/hiAndrewQuinn/resorter) if you're looking for the original resorter in an easier-to-install package. Python doesn't have the same kind of statistics packages available, so I had to handroll the comparison functionality, I'm not 100% sure I have it correct but it passes the smell test insofar as rankings.

This is primarily for integrating with the rest of the python ecosystem, for example [streamlit](https://streamlit.io/), where I plan to pipe my own media rankings and so on and so forth.
