namespace ProjectEulerSolutions.P25

partial def firstFibIndexWithDigits (digits : Nat) : Nat :=
  if digits <= 1 then
    1
  else
    let rec loop (a b n : Nat) : Nat :=
      if (toString b).length >= digits then
        n
      else
        loop b (a + b) (n + 1)
    loop 1 1 2


def sol (digits : Nat) : Nat :=
  firstFibIndexWithDigits digits

example : sol 1 = 1 := by
  native_decide

example : sol 2 = 7 := by
  native_decide

example : sol 3 = 12 := by
  native_decide

end ProjectEulerSolutions.P25
open ProjectEulerSolutions.P25

def main : IO Unit := do
  IO.println (sol 1000)
