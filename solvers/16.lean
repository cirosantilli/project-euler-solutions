namespace ProjectEulerSolutions.P16

partial def digitSum (n : Nat) : Nat :=
  let s := toString n
  s.data.foldl (fun acc c => acc + (c.toNat - '0'.toNat)) 0


def sol : Nat :=
  digitSum (Nat.pow 2 1000)

example : digitSum (Nat.pow 2 15) = 26 := by
  native_decide

end ProjectEulerSolutions.P16
open ProjectEulerSolutions.P16

def main : IO Unit := do
  IO.println sol
