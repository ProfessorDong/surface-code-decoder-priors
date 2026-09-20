# TikZ / pgfplots figure pipeline

The manuscript figures are native pgfplots, not rasterised or embedded PDFs.
They are produced in two steps from the same result CSVs the matplotlib
figures use, so the two always show identical data.

    python simulation/prx/plotting/tikz/capture.py   # -> /tmp/.../captured.json
    python simulation/prx/plotting/tikz/gen.py       # -> manuscript/figures/*.tex
                                                     #    manuscript/figdata/*.dat
                                                     #    manuscript/prxfig.tex

`capture.py` runs `simulation.prx.plotting.figures.main()` with the matplotlib
`Axes` methods instrumented, recording every series, axis limit, tick label and
legend placement. `gen.py` re-emits those series as pgfplots `\addplot` commands
reading plain-text tables from `manuscript/figdata/`.

`manuscript/prxfig.tex` holds the shared colour definitions (Okabe-Ito,
colour-blind safe, as required by the APS accessibility guidelines) and the
`prxaxis` style. The manuscript inputs it once in the preamble and then uses
`\input{figures/figN_name}` inside each float.
