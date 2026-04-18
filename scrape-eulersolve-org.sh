#!/usr/bin/env bash
set -eux
d=solvers/eulersolve
mkdir -p "$d"
sleep_time=1
for i in `seq 1 993`; do
  echo $i
  o="$d/$i.cpp"
  if ! [ -s "$o" ]; then
    url="https://eulersolve.org/solutionsCpp/Euler$i.cpp"
    # They started blocking wget user agent after I scraped for the first time. These rich people are funny.
    wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0" -O "$o" "$url"
    sed -i "1s|^|// $url\n\n|" "$o"
    sleep "$sleep_time"
  fi
  o="$d/$i.py"
  if ! [ -s "$o" ]; then
    url="https://eulersolve.org/solutionsPython/Euler$i.py"
    wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0" -O "$o" "$url"
    sed -i "1s|^|# $url\n\n|" "$o"
    sleep "$sleep_time"
  fi
  o="$d/$i.java"
  if ! [ -s "$o" ]; then
    url="https://eulersolve.org/solutionsJava/Euler$i.java"
    wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0" -O "$o" "$url"
    sed -i "1s|^|// $url\n\n|" "$o"
    sleep "$sleep_time"
  fi
  o="$d/$i.html"
  if ! [ -s "$o" ]; then
    url="https://eulersolve.org/mathematicalApproaches/Euler$i"
    wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0" -O "$o" "$url"
    sed -i "1s|^|// $url\n\n|" "$o"
    sleep "$sleep_time"
  fi
done
