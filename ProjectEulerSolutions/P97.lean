import ProjectEulerStatements.P97
import ProjectEulerSolutions.Termination.P97
namespace ProjectEulerSolutions.P97

def powMod (a e mod : Nat) : Nat :=
  let rec loop (base exp acc : Nat) : Nat :=
    if exp == 0 then
      acc
    else
      let acc := if exp % 2 == 1 then (acc * base) % mod else acc
      loop ((base * base) % mod) (exp / 2) acc
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  if mod == 1 then 0 else loop (a % mod) e 1

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solveWithParams (k exp digits : Nat) : Nat :=
  let mod := Nat.pow 10 digits
  (k * powMod 2 exp mod + 1) % mod

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve : Nat :=
  solveWithParams 28433 7830457 10
end ProjectEulerSolutions.P97
