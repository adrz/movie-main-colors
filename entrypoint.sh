#!/bin/sh
. env/bin/activate
echo python /app/main-colors.py -i "$1" -a "$2" -c "$3" -n "$4" --normalize "$5" -x "$6" -s "$7" -t "$8" -o "$9"
python /app/main-colors.py -i "$1" -a "$2" -c "$3" -n "$4" --normalize "$5" -x "$6" -s "$7" -t "$8" -o "$9"
