import solvers.«12.py.lean.d».FactorSource
import lang.py.FormalSW.HeapRuntimeCostReturns

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

abbrev factorTail : List Stmt := (executableBody factorMethod.2).drop 5

/-- The unstripped residue contributes at most one final multiplication. -/
theorem factor_finish (x total : Nat) (heap : Heap) (env : HeapCalls.Env)
    (getX : env "x" = some (.scalar (.int x)))
    (getTotal : env "total" = some (.scalar (.int total))) :
    BlockReturnsIn .word program env factorTail heap 18
      (.scalar (.int (total * ProjectEulerStatements.FactorDivisors.finish x : Nat))) heap := by
  have tested : EvaluatesIn .word program env (.compare (.name "x") [(.gt, .constant (.int 1))]) heap 7
      (.scalar (.bool (decide (1 < x)))) heap := by
    apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup getX) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  by_cases large : 1 < x
  · simp only [ProjectEulerStatements.FactorDivisors.finish, if_pos large]
    apply BlockReturnsIn.branch (cc := 7) (cb := 9) (flag := true) (by simpa [large] using tested) rfl
    apply BlockReturnsIn.augAssign (work := 1) (ca := 1) (cb := 2)
      (value := .scalar (.int (total * 2 : Nat)))
      getTotal EvaluatesIn.constant
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
    · rfl
    · exact BlockReturnsIn.return_value (EvaluatesIn.lookup rfl)
  · simp only [ProjectEulerStatements.FactorDivisors.finish, if_neg large, Nat.mul_one]
    apply BlockReturnsIn.mono (small := 11) (larger := by omega)
    exact BlockReturnsIn.branch (cc := 7) (cb := 2) (flag := false) (by simpa [large] using tested) rfl
      (BlockReturnsIn.return_value (EvaluatesIn.lookup getTotal))

end solvers.«12».runtimeCost
