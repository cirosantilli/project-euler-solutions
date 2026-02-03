import ProjectEulerStatements.P34
namespace ProjectEulerSolutions.P34

partial def precomputeFactorials : Array Nat :=
  let rec loop (d f : Nat) (arr : Array Nat) : Array Nat :=
    if d > 9 then
      arr
    else
      let f' := f * d
      loop (d + 1) f' (arr.set! d f')
  loop 1 1 (Array.replicate 10 1)

partial def digitFactorialSum (n : Nat) (facts : Array Nat) : Nat :=
  let rec loop (m acc : Nat) : Nat :=
    if m == 0 then
      acc
    else
      let d := m % 10
      loop (m / 10) (acc + facts[d]!)
  loop n 0

partial def findNBound (nineFact n steps : Nat) : Nat :=
  if steps == 0 then
    n
  else if n * nineFact >= Nat.pow 10 (n - 1) then
    findNBound nineFact (n + 1) (steps - 1)
  else
    n

partial def solve (limit : Nat) : Nat :=
  let facts := precomputeFactorials
  let nineFact := facts[9]!
  let n := findNBound nineFact 1 10
  let upper := Nat.min limit ((n - 1) * nineFact)
  let rec loop (x total : Nat) : Nat :=
    if x > upper then
      total
    else
      let total' := if x == digitFactorialSum x facts then total + x else total
      loop (x + 1) total'
  loop 3 0

def defaultLimit : Nat :=
  let facts := precomputeFactorials
  let nineFact := facts[9]!
  let n := findNBound nineFact 1 10
  (n - 1) * nineFact


theorem equiv (n : Nat) : ProjectEulerStatements.P34.naive n = solve n := sorry
end ProjectEulerSolutions.P34
