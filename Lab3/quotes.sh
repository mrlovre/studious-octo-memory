if [ $# -ne 1 ]; then exit 1; fi
sep=$(perl -e "print \"=\" x $(tput cols)")
grep -E '^[^%]' $1 | sed -r "s/(.){1000}/\1\n$sep\n/g"
