.PHONY: all clean
all: main.pdf
main.pdf: main.tex references.bib $(wildcard generated/*.tex) $(wildcard generated/*.txt) $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
clean:
	latexmk -c main.tex
