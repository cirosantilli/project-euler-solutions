import ProjectEulerStatements.P15
import ProjectEulerSolutions.Termination.P15
namespace ProjectEulerSolutions.P15

def binom (n k : Nat) : Nat :=
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
    termination_by 0
    decreasing_by all_goals exact Termination.decreases
    loop 1 1

termination_by 0
decreasing_by all_goals exact Termination.decreases
def latticePaths (gridN gridM : Nat) : Nat :=
  binom (gridN + gridM) gridN


termination_by 0
decreasing_by all_goals exact Termination.decreases
example : latticePaths 2 2 = 6 := by
  native_decide


def solve (n : Nat) : Nat :=
  latticePaths n n
end ProjectEulerSolutions.P15
