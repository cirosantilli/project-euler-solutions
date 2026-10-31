import solvers.«92.py.lean.d».Answer
import solvers.«92.py.lean.d».Distribution

namespace solvers.«92».solve
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := table.number
abbrev spec := initialization.spec

/-- Every positive decimal bound builds its exact distribution and counts the states reaching 89. -/
theorem positive_result (bound : Nat) (positive : 0 < bound) (heap : Mutable.Heap)
    (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program spec "solve" [number (bound + 1)] heap
      (fun value _ => value = number (count (bound + 1 : Nat))) globals := by
  apply HeapMemoCalls.BlockReturnsRel.function (by change "solve" ∉ ([] : List String); simp) rfl rfl
  apply HeapMemoCalls.BlockReturnsRel.branch
    (HeapMemoCalls.Evaluates.from_heap (HeapCalls.Extends.refl program) answer.disjoint
      (Evaluates.compare (value := .scalar (.bool false)) (finalHeap := heap) (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, Nat.ne_of_gt positive]))
      (by intro cache valid; exact valid)).rel
  rintro value middle ⟨rfl, same⟩
  subst middle
  refine ⟨false, by simp [Mutable.truth, scalarTruth], ?_⟩
  change HeapMemoCalls.BlockReturnsRel program spec _ (distribution.body ++ answer.body) heap
    (fun value _ => value = number (count (bound + 1 : Nat))) globals
  intro anchors stack
  apply (distribution.verified bound heap globals _ (by cases globals <;> rfl) answer.body anchors stack).bind
  rintro state ⟨locals, finalHeap, address, entries, extra, rfl, _, counts, found, keys, exact⟩
  exact answer.verified finalHeap locals globals bound address entries counts found keys exact anchors stack

/-- Signed bounds at most one return immediately; larger bounds use the complete decimal traversal. -/
theorem result (limit : Int) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program spec "solve" [.scalar (.int limit)] heap
      (fun value _ => value = number (count limit)) globals := by
  by_cases small : limit ≤ 1
  · have raw : ReturnsValue program "solve" [.scalar (.int limit)] heap (number 0) heap globals := by
      apply BlockReturns.function rfl rfl
      apply BlockReturns.branch (flag := true)
        (Evaluates.compare (value := .scalar (.bool true)) (finalHeap := heap) (Evaluates.lookup (by cases globals <;> rfl)) Evaluates.constant
          (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small]))
        (by simp [Mutable.truth, scalarTruth])
      exact BlockReturns.return_value Evaluates.constant
    have lifted := HeapMemoCalls.ReturnsValue.from_heap (HeapCalls.Extends.refl program) answer.disjoint raw
      (spec := spec) (by intro cache valid; exact valid)
    apply lifted.rel.consequence
    intro value finalHeap same
    simpa [count_nonpositive limit small] using same.1
  · let bound := (limit - 1).toNat
    have positive : 0 < bound := by dsimp [bound]; omega
    have equal : ((bound + 1 : Nat) : Int) = limit := by dsimp [bound]; omega
    simpa only [number, table.number, next_term.number, equal] using positive_result bound positive heap globals

end solvers.«92».solve
