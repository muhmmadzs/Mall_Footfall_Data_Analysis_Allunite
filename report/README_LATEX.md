# LaTeX Report

Main source:

```bash
report/assignment_report.tex
```

Compile from the repository root:

```bash
pdflatex -interaction=nonstopmode -output-directory report report/assignment_report.tex
pdflatex -interaction=nonstopmode -output-directory report report/assignment_report.tex
```

Or compile from the `report/` directory:

```bash
cd report
pdflatex -interaction=nonstopmode assignment_report.tex
pdflatex -interaction=nonstopmode assignment_report.tex
```

The report references existing figures under `../outputs/plots/`.
