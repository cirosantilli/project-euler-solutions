import ProjectEulerStatements.P34
import ProjectEulerSolutions.Termination.P34
namespace ProjectEulerSolutions.P34

def precomputeFactorials : Array Nat :=
  let rec loop (d f : Nat) (arr : Array Nat) : Array Nat :=
    if d > 9 then
      arr
    else
      let f' := f * d
      loop (d + 1) f' (arr.set! d f')
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 1 1 (Array.replicate 10 1)

termination_by 0
decreasing_by all_goals exact Termination.decreases
def digitFactorialSum (n : Nat) (facts : Array Nat) : Nat :=
  let rec loop (m acc : Nat) : Nat :=
    if m == 0 then
      acc
    else
      let d := m % 10
      loop (m / 10) (acc + facts[d]!)
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop n 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def findNBound (nineFact n steps : Nat) : Nat :=
  if steps == 0 then
    n
  else if n * nineFact >= Nat.pow 10 (n - 1) then
    findNBound nineFact (n + 1) (steps - 1)
  else
    n

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (limit : Nat) : Nat :=
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
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop 3 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def defaultLimit : Nat :=
  let facts := precomputeFactorials
  let nineFact := facts[9]!
  let n := findNBound nineFact 1 10
  (n - 1) * nineFact
end ProjectEulerSolutions.P34
