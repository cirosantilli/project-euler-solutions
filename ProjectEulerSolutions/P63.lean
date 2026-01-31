import ProjectEulerStatements.P63
namespace ProjectEulerSolutions.P63

partial def solve (limit : Nat) : Nat :=
  let rec loopN (n : Nat) (total : Nat) : Nat :=
    if n > limit then
      total
    else
      let len9 := (toString (Nat.pow 9 n)).length
      if len9 < n then
        total
      else
        let rec loopA (a : Nat) (acc : Nat) : Nat :=
          if a > 9 then
            acc
          else
            let len := (toString (Nat.pow a n)).length
            let acc := if len == n then acc + 1 else acc
            loopA (a + 1) acc
        loopN (n + 1) (loopA 1 total)
  loopN 1 0


example : (toString (Nat.pow 7 5)).length = 5 := by
  native_decide

example : (toString (Nat.pow 8 9)).length = 9 := by
  native_decide


theorem equiv (n : Nat) : ProjectEulerStatements.P63.naive n = solve n := sorry
end ProjectEulerSolutions.P63
