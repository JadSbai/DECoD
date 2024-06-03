#!/bin/bash
FIRST_PAGE=8
LAST_PAGE=50
PDF_FILE="MPhil_Dissertation-10.pdf"

pdftotext -f $FIRST_PAGE -l $LAST_PAGE $PDF_FILE - | \
grep -E '[A-Za-z]{3}' | wc -w
