# Pending analyses and improvements

## Issues/bugs

- EC_PIP database.duckdb still has the crazy e+12 link between two personas naturales
- HERE update preprocess_rawdata ipynb files.

## Possible directions

### Shock detection

- Can we see the 2011-2012 drop in yield kg/ha due to Black Sigatoka + heavy rainfall? BlackSigatoka2012.ipynb
- How is the micro-behavior changing to adapt to the shock? Is the supply of trucks falling because of the lower production? Look at the aggregated sector connections. Sankey!
- Groundtruth: the entity of the shock is visible from the satellite data of the harvested material. Start from this.
- Assume that the expenses for inputs (e.g. supply of cardboard X) are not affected by the reduced harvest, and that the total production (FAO) Z aggregates at the country level the actual production (affected). If the production drops and the cost for inputs is linear to produced material (1 box for 18 kgs!), Z/X over the years is stable. And then sector in-strength is a proxy for sector production.

### Production volumes/flows

- Sankey diagram for the production network (sanity check if we capture what expected).

### Trade

- Elaborate "ND" in edgelist. Validate with trade data (international import)?

### Industry resolution, or product granularity

- How much approximation are we introducing by using "*A0122 - Growing of tropical and subtropical fruits (avocados, bananas, dates, ...)*" instead of "*Growing of bananas*""? We can quantifing this by comparing the trade of bananas and aggregated tropical fruits. Or use the LUIcube data and compare the harvested mass.
