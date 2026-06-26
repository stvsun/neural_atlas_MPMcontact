#!/bin/sh
# Render an Asymptote 3D figure to vector PDF using the local TinyTeX engine.
export PATH="$PATH:/Users/wsun/Library/TinyTeX/bin/universal-darwin"
export openout_any=a
export openin_any=a
TEXPATH=/Users/wsun/Library/TinyTeX/bin/universal-darwin
name="$1"
/opt/homebrew/bin/asy -texpath="$TEXPATH" -f pdf -o "$name" "$name.asy"
