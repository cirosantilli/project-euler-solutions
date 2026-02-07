import ProjectEulerStatements.P29
namespace ProjectEulerSolutions.P29

partial def solve (maxA maxB : Nat) : Nat :=
  let rec loopA (a : Nat) (vals : List Nat) : List Nat :=
    if a > maxA then
      vals
    else
      let rec loopB (b : Nat) (vals : List Nat) : List Nat :=
        if b > maxB then
          vals
        else
          let v := a ^ b
          let vals' := if vals.contains v then vals else v :: vals
          loopB (b + 1) vals'
      loopA (a + 1) (loopB 2 vals)
  (loopA 2 []).length


example : solve 5 5 = 15 := by
  native_decide

theorem equiv (aMax bMax : Nat) : ProjectEulerStatements.P29.naive aMax bMax = solve aMax bMax := sorry
end ProjectEulerSolutions.P29
