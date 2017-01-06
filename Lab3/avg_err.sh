if [ $# -ne 1 ]; then exit 1; fi
grep -E '^%' $1 | sed -r 's/^% \S+ \S+ average loss: (.*)$/\1/g'
