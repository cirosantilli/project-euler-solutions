import solvers.«93.py.lean.d».StatePreparation
import ProjectEulerSolutions.Lib.FilteredRangeCollection
import ProjectEulerSolutions.Lib.IndexedFilterPairs

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.HeapCollection
set_option Elab.async false

abbrev remainderCondition : Expr := .boolOp .and
  [.compare (.name "k") [(.notEq, .name "i")], .compare (.name "k") [(.notEq, .name "j")]]
abbrev remainderElement : Expr := .subscript (.name "state") (.name "k")
abbrev remainderBounds : List Expr := [.builtinCall .len [.name "state"]]
abbrev remainderExpr : Expr :=
  .listComp remainderElement [("k", .call (.name "range") remainderBounds, [remainderCondition])]
abbrev keepOperand := ProjectEulerSolutions.IndexedFilter.exceptTwo

private def forBody : Option Stmt → List Stmt
  | some (.forIn _ _ body _) => body
  | _ => []

theorem source_remainder : (forBody ((forBody recBody[3]?)[0]?))[0]? =
    some (.assign [.name "rest"] remainderExpr) := rfl

/-- The first rejected index short-circuits the second comparison. -/
theorem remainder_condition (heap : Mutable.Heap) (i j index : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getI : env "i" = some (.scalar (.int i))) (getJ : env "j" = some (.scalar (.int j)))
    (getIndex : env "k" = some (.scalar (.int index))) :
    Evaluates program env remainderCondition heap (.scalar (.bool (keepOperand i j index))) heap globals := by
  have compared (name : String) (value : Nat) (found : env name = some (.scalar (.int value))) :
      Evaluates program env (.compare (.name "k") [(.notEq, .name name)]) heap
        (.scalar (.bool (decide (index ≠ value)))) heap globals := by
    apply Evaluates.compare (Evaluates.lookup getIndex) (Evaluates.lookup found)
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  have first := compared "i" i getI
  have second := compared "j" j getJ
  by_cases same : index = i
  · have stopped := Evaluates.boolean_stop (op := BoolOp.and)
      (rest := [.compare (.name "k") [(.notEq, .name "j")]]) first
      (flag := false) (by simp [Mutable.truth, scalarTruth, same]) rfl
    simpa [remainderCondition, keepOperand, ProjectEulerSolutions.IndexedFilter.exceptTwo, same] using stopped
  · have last := Evaluates.boolean_last (op := BoolOp.and) second
      (flag := decide (index ≠ j)) (by simp [Mutable.truth, scalarTruth])
    have continued := Evaluates.boolean_next (op := BoolOp.and) first
      (flag := true) (by simp [Mutable.truth, scalarTruth, same]) rfl last
    simpa [remainderCondition, keepOperand, ProjectEulerSolutions.IndexedFilter.exceptTwo, same] using continued

/-- The actual comprehension skips the two positions without deduplicating equal operands. -/
theorem remainder_expression (heap : Mutable.Heap) (state : Nat) (items : List Mutable.Value) (i j : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[state]? = some (.tuple items)) (getState : env "state" = some (.ref state))
    (getI : env "i" = some (.scalar (.int i))) (getJ : env "j" = some (.scalar (.int j))) :
    Evaluates program env remainderExpr heap (.ref heap.length)
      (heap ++ [.list (ProjectEulerSolutions.IndexedFilter.select (keepOperand i j) 0 items)]) globals := by
  have lengthExpr : Evaluates program env (.builtinCall .len [.name "state"]) heap
      (.scalar (.int items.length)) heap globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getState) .nil)
    simp [primitive, Mutable.length, found]
  apply ProjectEulerSolutions.FilteredRangeCollection.collect program heap env "k" remainderElement
    remainderCondition items (keepOperand i j) remainderBounds [.scalar (.int items.length)]
    (.cons lengthExpr .nil) rfl
  · intro collected index inside
    apply Evaluates.subscript (by intro a b c impossible; cases impossible)
      (Evaluates.lookup (value := .ref state) (by simp [comprehensionEnv, setLocal, getState]))
      (Evaluates.lookup (value := .scalar (.int index)) (by simp [setLocal]))
    have kept : (working heap collected [])[state]? = some (.tuple items) :=
      prefix_found heap [.list collected] state _ found
    have nonnegative : ¬ (index : Int) < 0 := by omega
    simp [Mutable.getItem, kept, Mutable.index, nonnegative, Nat.not_le.mpr inside,
      List.getElem?_eq_getElem inside, Except.map]
  · intro collected index _
    apply remainder_condition
    · simp [comprehensionEnv, setLocal, getI]
    · simp [comprehensionEnv, setLocal, getJ]
    · simp [setLocal]

theorem remainder_rows (heap : Mutable.Heap) (items : List Mutable.Value) (pairs : List Pair) (i j : Nat)
    (rows : Heap.Rows heap items pairs) :
    Heap.Rows (heap ++ [.list (ProjectEulerSolutions.IndexedFilter.select (keepOperand i j) 0 items)])
      (ProjectEulerSolutions.IndexedFilter.select (keepOperand i j) 0 items)
      (ProjectEulerSolutions.IndexedFilter.select (keepOperand i j) 0 pairs) := by
  have selected : Heap.Rows heap (ProjectEulerSolutions.IndexedFilter.select (keepOperand i j) 0 items)
      (ProjectEulerSolutions.IndexedFilter.select (keepOperand i j) 0 pairs) :=
    ProjectEulerSolutions.IndexedFilter.forall₂ rows (keepOperand i j) 0
  exact selected.append _

end solvers.«93».all_results
