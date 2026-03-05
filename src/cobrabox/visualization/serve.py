"""Entry point for ``panel serve src/cobrabox/visualization/serve.py``."""

import panel as pn

import cobrabox as cb
from cobrabox.visualization import explore

pn.extension(sizing_mode="stretch_width")

data = cb.dataset("dummy_chain")[0]
app = explore(data)
app.as_template().servable()
