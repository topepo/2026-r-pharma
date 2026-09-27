---
output: github_document
---

# R/Pharma Workshop: Deep Learning and Foundational Models for Tabular Data in R

- Slides: https://topepo.github.io/2026-r-pharma
- Code: https://github.com/topepo/2026-r-pharma

## Software:

The basic are in: 

```r
install.packages(c("tidymodels", "brulee"))
```

After brulee is installed, run this command to download weight files for TabICL:

```r
brulee::tab_icl_download_weights()
```

If you want to tune models, additional downloads are: 

```r
install.packages("pak")
pak::pak(c(""tidymodels/tabby"))
```
