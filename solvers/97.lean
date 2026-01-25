import ProjectEulerStatements.P97
namespace ProjectEulerSolutions.P97

partial def powMod (a e mod : Nat) : Nat :=
  let rec loop (base exp acc : Nat) : Nat :=
    if exp == 0 then
      acc
    else
      let acc := if exp % 2 == 1 then (acc * base) % mod else acc
      loop ((base * base) % mod) (exp / 2) acc
  if mod == 1 then 0 else loop (a % mod) e 1

partial def solve (k exp digits : Nat) : Nat :=
  let mod := Nat.pow 10 digits
  (k * powMod 2 exp mod + 1) % mod



theorem equiv : ProjectEulerStatements.P97.naive = solve 28433 7830457 10 := sorry
end ProjectEulerSolutions.P97
open ProjectEulerSolutions.P97

def main : IO Unit := do
  IO.println (solve 28433 7830457 10)
