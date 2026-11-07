import solvers.«12.py.lean.d».SearchSource
import solvers.«12.py.lean.d».CacheRuntime

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

theorem split_cost (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (.scalar (.int n))) :
    BlockContinuesIn .word program env [searchBody[0]'(by decide)] heap 30 (splitEnv env n) heap := by
  have tested : EvaluatesIn .word program env
      (.compare (.binOp (.name "n") .mod (.constant (.int 2))) [(.eq, .constant (.int 0))]) heap 12
      (.scalar (.bool (decide (n % 2 = 0)))) heap := by
    apply EvaluatesIn.compare (work := 2) (av := .scalar (.int (n % 2))) (middle := heap)
      (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup getN) EvaluatesIn.constant
        (by simp [Mutable.binary, scalarBinary]) rfl) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      omega
    · rfl
  by_cases even : n % 2 = 0
  · apply BlockContinuesIn.branch (cc := 12) (cb := 16) (flag := true) (by simpa [even] using tested) rfl
    apply BlockContinuesIn.append (ca := 8) (cb := 8)
      (body := [.assign [.name "a"] (.binOp (.name "n") .floorDiv (.constant (.int 2)))])
      (middleEnv := HeapCalls.setLocal env "a" (.scalar (.int (n / 2 : Nat)))) (middleHeap := heap)
    · apply BlockContinuesIn.assign
      exact EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup getN) EvaluatesIn.constant
        (by simp [Mutable.binary, scalarBinary]) rfl
    · simp only [splitEnv, ProjectEulerStatements.P12.splitLeft, ProjectEulerStatements.P12.splitRight, if_pos even]
      apply BlockContinuesIn.assign
      exact EvaluatesIn.binary (work := 1)
        (EvaluatesIn.lookup (by simpa [HeapCalls.setLocal] using getN)) EvaluatesIn.constant
        (by simp [Mutable.binary, scalarBinary]) rfl
  · apply BlockContinuesIn.branch (cc := 12) (cb := 16) (flag := false) (by simpa [even] using tested) rfl
    apply BlockContinuesIn.append (ca := 3) (cb := 13) (body := [.assign [.name "a"] (.name "n")])
      (middleEnv := HeapCalls.setLocal env "a" (.scalar (.int n))) (middleHeap := heap)
    · exact BlockContinuesIn.assign (EvaluatesIn.lookup getN)
    · simp only [splitEnv, ProjectEulerStatements.P12.splitLeft, ProjectEulerStatements.P12.splitRight, if_neg even]
      apply BlockContinuesIn.assign
      apply EvaluatesIn.binary (ca := 6) (cb := 1) (work := 1) (av := .int (n + 1 : Nat)) (middle := heap)
        (bv := .int 2) (afterRight := heap)
      · exact EvaluatesIn.binary (work := 1)
          (EvaluatesIn.lookup (by simpa [HeapCalls.setLocal] using getN)) EvaluatesIn.constant
          (by simp [Mutable.binary, scalarBinary]) rfl
      · exact EvaluatesIn.constant
      · simp [Mutable.binary, scalarBinary]
      · rfl

theorem triangle_cost (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (.scalar (.int n))) :
    EvaluatesIn .word program env
      (.binOp (.binOp (.name "n") .mult (.binOp (.name "n") .add (.constant (.int 1))))
        .floorDiv (.constant (.int 2))) heap 16 (.scalar (.int (ProjectEulerStatements.P12.triangle n))) heap := by
  apply EvaluatesIn.binary (ca := 11) (cb := 1) (work := 1) (av := .int (n * (n + 1) : Nat)) (middle := heap)
    (bv := .int 2) (afterRight := heap)
  · apply EvaluatesIn.binary (ca := 1) (cb := 6) (work := 1) (middle := heap) (afterRight := heap) (EvaluatesIn.lookup getN)
      (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup getN) EvaluatesIn.constant
        (value := .scalar (.int (n + 1 : Nat))) (afterRight := heap) (by simp [Mutable.binary, scalarBinary]) rfl)
    · simp [Mutable.binary, scalarBinary]
    · rfl
  · exact EvaluatesIn.constant
  · simp [Mutable.binary, scalarBinary, ProjectEulerStatements.P12.triangle]
  · rfl


end solvers.«12».runtimeCost
