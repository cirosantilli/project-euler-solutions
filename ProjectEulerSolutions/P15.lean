import ProjectEulerStatements.P15
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


example : latticePaths 2 2 = 6 := by
  native_decide


def solve (n : Nat) : Nat :=
  latticePaths n n

theorem equiv (n : Nat) : ProjectEulerStatements.P15.naive n = solve n := sorry
end ProjectEulerSolutions.P15
