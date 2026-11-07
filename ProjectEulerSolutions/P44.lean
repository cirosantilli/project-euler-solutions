import ProjectEulerSolutions.Lib.PentagonalSearch
namespace ProjectEulerSolutions.P44
open PentagonalSearch PentagonalDiscriminant

/-- Descending second indices permit pruning as soon as the difference exceeds the best. -/
def scanRow (k : Nat) : Nat → Nat → Nat
  | 0, best => best
  | j+1, best =>
      let diff := difference k (j+1)
      if best ≤ diff then best
      else
        let accepted := discriminant (pentagonal k + pentagonal (j+1)) && discriminant diff
        scanRow k j (if accepted then diff else best)

def solve (limit : Nat) : Nat :=
  (List.range' 2 (limit-1)).foldl (fun best k => scanRow k (k-1) best) cap
end ProjectEulerSolutions.P44
