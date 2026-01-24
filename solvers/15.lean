namespace ProjectEulerSolutions.P15

partial def binom (n k : Nat) : Nat :=
  if k > n then
    0
  else
    let k' :=
      if k < n - k then k else n - k
    let rec loop (i res : Nat) : Nat :=
      if i > k' then
        res
      else
        loop (i + 1) (res * (n - k' + i) / i)
    loop 1 1

partial def latticePaths (gridN gridM : Nat) : Nat :=
  binom (gridN + gridM) gridN


def sol (gridN gridM : Nat) : Nat :=
  latticePaths gridN gridM

example : sol 2 2 = 6 := by
  native_decide

end ProjectEulerSolutions.P15
open ProjectEulerSolutions.P15

def main : IO Unit := do
  IO.println (sol 20 20)
