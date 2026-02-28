#!/usr/bin/env bash
set -eux
d=tmp/eulersolve
mkdir -p "$d"
for i in `seq 1 986`; do
  echo $i
  wget -O "$d/$i.cpp" https://eulersolve.org/solutionsCpp/Euler$i.cpp
  sed -i "1s|^|// https://eulersolve.org/solutionsCpp/Euler$i.cpp\n\n|" "$d/$i.cpp"
done
