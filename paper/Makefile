.PHONY: all clean scientific-assets
all: main.pdf annotated.pdf

INPUTS = $(wildcard *.tex) references.bib $(wildcard generated/*.tex) $(wildcard generated/*.txt) $(wildcard figures/*.pdf)

main.pdf: $(INPUTS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

annotated.pdf: annotated.tex $(INPUTS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error annotated.tex

scientific-assets:
	python3 build_scientific_assets.py

clean:
	latexmk -c main.tex annotated.tex
