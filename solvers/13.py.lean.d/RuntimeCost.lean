import solvers.«13.py.lean.d».SourceRuntime
import lang.py.FormalSW.SumCalls

namespace solvers.«13»
open FormalSW.PyAst
open HeapCalls.Runtime.NestedIntList (encode)

theorem solve.«BigO(n, n = sum(1 + value.bit_length() for value in nums))» :
    HeapCalls.Runtime.IsBigO .word Solvers.P13Ast.module "solve"
      (fun input : List Int × Int => [.ref 0, .scalar (.int input.2)])
      (fun input : List Int × Int => [.list (encode input.1)])
      (fun input : List Int × Int =>
        HeapCalls.Condition Solvers.P13Ast.module
          (fun name => if name = "nums" then some (.ref 0) else none)
          Solvers.P13Ast.requires_solve_0 [.list (encode input.1)] ∧
        HeapCalls.Condition Solvers.P13Ast.module
          (fun name => if name = "k" then some (.scalar (.int input.2)) else none)
          Solvers.P13Ast.requires_solve_1 [.list (encode input.1)])
      (fun input : List Int × Int => IntegerListWork.size input.1) (fun n : Nat => n) := by
  refine ⟨141, 1, ?_⟩
  rintro ⟨items, k⟩ ⟨validSum, validCount⟩ large
  have checkedSum : HeapCalls.Evaluates Solvers.P13Ast.module
      (fun name => if name = "nums" then some (.ref 0) else none)
      Solvers.P13Ast.requires_solve_0 [.list (encode items)]
      (.scalar (.bool (decide (0 ≤ items.foldl Int.add 0)))) [.list (encode items)] := by
    apply HeapCalls.Evaluates.compare (bv := .scalar (.int 0))
      (HeapCalls.Evaluates.sum_integers (values := items)
        (.cons (HeapCalls.Evaluates.lookup rfl) (.cons HeapCalls.Evaluates.constant .nil))
        (by simp [Mutable.sequence, encode])) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have checkedCount : HeapCalls.Evaluates Solvers.P13Ast.module
      (fun name => if name = "k" then some (.scalar (.int k)) else none)
      Solvers.P13Ast.requires_solve_1 [.list (encode items)]
      (.scalar (.bool (decide (0 ≤ k)))) [.list (encode items)] := by
    apply HeapCalls.Evaluates.compare (HeapCalls.Evaluates.lookup rfl) HeapCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have nonnegative : 0 ≤ items.foldl Int.add 0 := by simpa using checkedSum.condition_iff.mp validSum
  have count : 0 ≤ k := by simpa using checkedCount.condition_iff.mp validCount
  have bounded := runtimeCost.source_bound items k.toNat nonnegative
  rw [Int.toNat_of_nonneg count] at bounded
  apply bounded.mono
  change 1 ≤ IntegerListWork.size items at large
  change 40*IntegerListWork.size items+101 ≤ 141*IntegerListWork.size items
  omega

end solvers.«13»
