import solvers.«68.permutations.source»
import ProjectEulerStatements.Lib.MagicRingPruning
import lang.py.FormalSW.SumCalls

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRing MagicRingTextCalls
set_option Elab.async false

abbrev widthElement : Expr := .builtinCall .len [.builtinCall .str [.name "value"]]
abbrev widthSumExpression : Expr := .sumGenerator widthElement
  [("value", .name "inner_set", [])] (.constant (.int 0))
abbrev widthCondition : Expr := .compare
  (.binOp (.name "total_digits") .add widthSumExpression) [(.notEq, .name "digits")]
abbrev innerSumExpression : Expr := .sumSequence (.name "inner_set") (.constant (.int 0))
abbrev totalExpression : Expr := .binOp (.name "total") .add (.name "inner_sum")
abbrev divisibilityCondition : Expr :=
  .compare (.binOp totalExpression .mod (.name "n")) [(.notEq, .constant (.int 0))]
abbrev targetExpression : Expr := .binOp totalExpression .floorDiv (.name "n")

theorem combination_code_source : combinationBody =
    [.ifThen widthCondition [.continueLoop] [],
     .assign [.name "inner_sum"] innerSumExpression,
     .ifThen divisibilityCondition [.continueLoop] [],
     .assign [.name "target"] targetExpression,
     .assign [.name "remaining_outer"] complementExpression, innerLoop] := rfl

private theorem weighted_fold (values : List Nat) (weight : Nat → Nat) (initial : Int) :
    values.foldl (fun current value => current + (weight value : Int)) initial =
      initial + ((values.map weight).sum : Int) := by
  induction values generalizing initial with
  | nil => simp
  | cons first rest ih =>
      simp only [List.foldl_cons, List.map_cons, List.sum_cons, Nat.cast_add, ih]
      omega

theorem width_element_result (value : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getValue : env "value" = some (number value)) :
    Evaluates program env widthElement heap (number (digitWidth value)) heap :=
  Evaluates.builtin
    (.cons (Evaluates.builtin (.cons (Evaluates.lookup getValue) .nil) rfl) .nil) rfl

theorem width_sum_result (selected : List Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (getSelected : heap[address]? = some (.tuple (encode selected)))
    (localSelected : env "inner_set" = some (.ref address)) :
    Evaluates program env widthSumExpression heap (number (selected.map digitWidth).sum) heap := by
  have sequence : Mutable.sequence (.ref address) heap = some (.ok (encode selected), heap) := by
    simp [Mutable.sequence, getSelected]
  apply Evaluates.sum_generator_sequence (by intros expressions impossible; cases impossible)
    (Evaluates.lookup localSelected) sequence Evaluates.constant rfl
  have summed := sum_map_loop (module := program) (values := selected)
    (encode := MagicRingTextCalls.number) (number := fun value => (digitWidth value : Int))
    (source := sequenceSource (.ref address)) (position := 0) (initial := 0)
    (env := setLocal (comprehensionEnv env [("value", .name "inner_set", [])])
      sumAccumulator (.scalar (.int 0))) selected (by decide) sequence
    (fun current value found => width_element_result value heap current found) rfl
    (show (setLocal _ sumAccumulator (.scalar (.int 0))) sumAccumulator =
      some (.scalar (.int 0)) from rfl)
  simpa only [weighted_fold, Int.zero_add] using summed

theorem inner_sum_result (selected : List Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (getSelected : heap[address]? = some (.tuple (encode selected)))
    (localSelected : env "inner_set" = some (.ref address)) :
    Evaluates program env innerSumExpression heap (number selected.sum) heap := by
  have total : (ints selected).foldl Int.add 0 = (selected.sum : Int) := by
    rw [ints, List.foldl_map]
    simpa only [id_eq, List.map_id, Int.zero_add] using weighted_fold selected id 0
  have summed := Evaluates.sum_integers (module := program) (globals := [])
    (values := ints selected)
    (heap := heap) (middle := heap) (env := env) (initial := 0)
    (.cons (Evaluates.lookup localSelected) (.cons Evaluates.constant .nil))
    (by simp [Mutable.sequence, getSelected, ints, encode, List.map_map, Function.comp_def])
  simpa only [total] using summed

theorem width_condition_result (n : Nat) (selected : List Nat) (digits : Int)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat)
    (getSelected : heap[address]? = some (.tuple (encode selected)))
    (localSelected : env "inner_set" = some (.ref address))
    (getWidth : env "total_digits" = some (number ((labels n).map digitWidth).sum))
    (getDigits : env "digits" = some (.scalar (.int digits))) :
    Evaluates program env widthCondition heap
      (.scalar (.bool (decide ((requiredDigits n selected : Int) ≠ digits)))) heap := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (av := number (requiredDigits n selected)) (bv := .scalar (.int digits))
  · exact Evaluates.binary (middle := heap) (afterRight := heap)
      (Evaluates.lookup getWidth)
      (width_sum_result selected heap env address getSelected localSelected)
      (by simp [Mutable.binary, scalarBinary, requiredDigits])
  · exact Evaluates.lookup getDigits
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]

theorem total_expression_result (n : Nat) (selected : List Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getTotal : env "total" = some (number (n * (2 * n + 1))))
    (getSum : env "inner_sum" = some (number selected.sum)) :
    Evaluates program env totalExpression heap (number (n * (2 * n + 1) + selected.sum)) heap :=
  Evaluates.binary (middle := heap) (afterRight := heap)
    (Evaluates.lookup getTotal) (Evaluates.lookup getSum)
    (by simp [Mutable.binary, scalarBinary])

theorem divisibility_condition_result (n : Nat) (positive : 0 < n) (selected : List Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getTotal : env "total" = some (number (n * (2 * n + 1))))
    (getSum : env "inner_sum" = some (number selected.sum))
    (getN : env "n" = some (number n)) :
    Evaluates program env divisibilityCondition heap
      (.scalar (.bool (decide ((n * (2 * n + 1) + selected.sum) % n ≠ 0)))) heap := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (av := number ((n * (2 * n + 1) + selected.sum) % n)) (bv := number 0)
  · exact Evaluates.binary (middle := heap) (afterRight := heap)
      (total_expression_result n selected heap env getTotal getSum) (Evaluates.lookup getN)
      (by simp [Mutable.binary, scalarBinary, positive, show n ≠ 0 by omega])
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    norm_cast

theorem target_expression_result (n : Nat) (positive : 0 < n) (selected : List Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getTotal : env "total" = some (number (n * (2 * n + 1))))
    (getSum : env "inner_sum" = some (number selected.sum))
    (getN : env "n" = some (number n)) :
    Evaluates program env targetExpression heap (number (target n selected)) heap :=
  Evaluates.binary (middle := heap) (afterRight := heap)
    (total_expression_result n selected heap env getTotal getSum) (Evaluates.lookup getN)
    (by simp [Mutable.binary, scalarBinary, target, positive, show n ≠ 0 by omega])

end solvers.«68»
