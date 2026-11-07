import ProjectEulerStatements.P26
import ProjectEulerStatements.Lib.ArgMaximum
namespace ProjectEulerSolutions.P26

/-- Strip factors 2 and 5, then iterate remainders for at most the denominator. -/
def recurringCycleLength (d : Nat) : Nat := ProjectEulerStatements.P26.cycleLen d

/-- Keep the first denominator when two periods have equal length. -/
def solve (limit : Nat) : Nat :=
  match List.range' 1 (limit - 1) with
  | [] => 0
  | first :: rest => rest.foldl (ProjectEulerStatements.ArgMaximum.select recurringCycleLength) first
end ProjectEulerSolutions.P26
