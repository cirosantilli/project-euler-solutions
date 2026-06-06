import ProjectEulerStatements.P40
import ProjectEulerSolutions.Termination.P40
namespace ProjectEulerSolutions.P40

def digitAt (n : Nat) : Nat :=
  let rec loop (n : Nat) (length start count : Nat) : Nat :=
    let blockDigits := count * length
    if n > blockDigits then
      loop (n - blockDigits) (length + 1) (start * 10) (count * 10)
    else
      let idx := (n - 1) / length
      let pos := (n - 1) % length
      let num := start + idx
      let s := toString num
      let ch := s.data[pos]!
      ch.toNat - '0'.toNat
  termination_by 0
  decreasing_by all_goals exact Termination.decreases
  loop n 1 1 9

termination_by 0
decreasing_by all_goals exact Termination.decreases
def totalDigitsUpTo (limit : Nat) : Nat :=
  if limit == 0 then
    0
  else
    let rec loop (length start count acc : Nat) : Nat :=
      if start > limit then
        acc
      else
        let last := Nat.min limit (start + count - 1)
        let nums := last - start + 1
        let acc := acc + nums * length
        loop (length + 1) (start * 10) (count * 10) acc
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    loop 1 1 9 0

termination_by 0
decreasing_by all_goals exact Termination.decreases
def digitAtBounded (n limit : Nat) : Nat :=
  let total := totalDigitsUpTo limit
  if n == 0 || n > total then 0 else digitAt n

termination_by 0
decreasing_by all_goals exact Termination.decreases
def solve (limit : Nat) : Nat :=
  let positions := [1, 10, 100, 1000, 10000, 100000, 1000000]
  positions.foldl (fun acc p => acc * digitAtBounded p limit) 1


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : digitAt 12 = 1 := by
  native_decide

example : digitAt 1 = 1 := by
  native_decide

example : digitAt 10 = 1 := by
  native_decide

example : digitAt 100 = 5 := by
  native_decide

example : digitAt 1000 = 3 := by
  native_decide

example : digitAt 10000 = 7 := by
  native_decide

example : digitAt 100000 = 2 := by
  native_decide

example : digitAt 1000000 = 1 := by
  native_decide
end ProjectEulerSolutions.P40
