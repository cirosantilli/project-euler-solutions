import solvers.«68.outer.source»
import ProjectEulerStatements.Lib.MagicRingSetValues

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRingTextCalls
set_option Elab.async false

abbrev complementExpression : Expr := .binOp (.name "nums") .sub
  (.builtinCall .set [.name "inner_set"])
abbrev validationExpression : Expr := .boolOp .or
  [.compare (.builtinCall .set [.name "outer"]) [(.notEq, .name "remaining_outer")],
   .compare (.builtinCall .len [.builtinCall .set [.name "outer"]]) [(.notEq, .name "n")]]

theorem complement_expression_source : combinationBody[4]? =
    some (.assign [.name "remaining_outer"] complementExpression) := rfl

theorem validation_expression_source : innerPermutationBody[3]? =
    some (.ifThen validationExpression [.continueLoop] []) := rfl

theorem complement_result (heap : Mutable.Heap) (env : HeapCalls.Env)
    (nums selected : List Int) (numsAddress selectedAddress : Nat)
    (getNums : heap[numsAddress]? = some (.set (IntegerSets.entries nums)))
    (getSelected : heap[selectedAddress]? = some (.tuple (IntegerSets.encoded selected)))
    (localNums : env "nums" = some (.ref numsAddress))
    (localSelected : env "inner_set" = some (.ref selectedAddress)) :
    Evaluates program env complementExpression heap (.ref (heap.length + 1))
      (heap ++ [.set (IntegerSets.entries selected),
        .set (ScalarSet.difference
          (IntegerSets.entries nums) (IntegerSets.entries selected))]) := by
  let middle := heap ++ [.set (IntegerSets.entries selected)]
  have selectedSet : Evaluates program env (.builtinCall .set [.name "inner_set"])
      heap (.ref heap.length) middle :=
    Evaluates.set_integer_sequence (tuple := true) (items := selected)
      (Evaluates.lookup localSelected)
      (by simpa [Mutable.scalarSequence, IntegerSets.encoded, List.map_map, Function.comp_def]
        using getSelected)
  have numsFound : middle[numsAddress]? = some (.set (IntegerSets.entries nums)) := by
    dsimp [middle]
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp getNums).1]
    exact getNums
  have result := Evaluates.difference_integer_sets (Evaluates.lookup localNums) selectedSet
    numsFound
    (show middle[heap.length]? = some (.set (IntegerSets.entries selected)) by simp [middle])
  simpa only [complementExpression, middle, List.length_append, List.length_singleton,
    List.append_assoc, List.singleton_append] using result

def validationFlag (n : Nat) (outer : List Int) (remaining : List Constant) : Bool :=
  !(ScalarSet.compare .eq (IntegerSets.entries outer) remaining) ||
    decide ((IntegerSets.entries outer).length ≠ n)

theorem validation_flag_false (n : Nat) (outer : List Int) (remaining : List Constant) :
    validationFlag n outer remaining = false ↔
      ScalarSet.compare .eq (IntegerSets.entries outer) remaining = true ∧
        (IntegerSets.entries outer).length = n := by
  simp [validationFlag]

theorem validation_result (n : Nat) (outer : List Int) (remaining : List Constant)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (outerAddress remainingAddress : Nat)
    (getOuter : heap[outerAddress]? = some (.list (IntegerSets.encoded outer)))
    (getRemaining : heap[remainingAddress]? = some (.set remaining))
    (localOuter : env "outer" = some (.ref outerAddress))
    (localRemaining : env "remaining_outer" = some (.ref remainingAddress))
    (getN : env "n" = some (number n)) :
    ∃ extra, Evaluates program env validationExpression heap
      (.scalar (.bool (validationFlag n outer remaining))) (heap ++ extra) := by
  let contents := IntegerSets.entries outer
  let heapOne := heap ++ [.set contents]
  let heapTwo := heapOne ++ [.set contents]
  let same := ScalarSet.compare .eq contents remaining
  have remainingOne : heapOne[remainingAddress]? = some (.set remaining) := by
    dsimp [heapOne]
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp getRemaining).1]
    exact getRemaining
  have outerOne : heapOne[outerAddress]? = some (.list (IntegerSets.encoded outer)) := by
    dsimp [heapOne]
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp getOuter).1]
    exact getOuter
  have firstSet : Evaluates program env (.builtinCall .set [.name "outer"])
      heap (.ref heap.length) heapOne :=
    Evaluates.set_integer_sequence (tuple := false) (items := outer) (Evaluates.lookup localOuter)
      (by simpa [Mutable.scalarSequence, IntegerSets.encoded, List.map_map, Function.comp_def]
        using getOuter)
  have secondSet : Evaluates program env (.builtinCall .set [.name "outer"])
      heapOne (.ref heapOne.length) heapTwo :=
    Evaluates.set_integer_sequence (tuple := false) (items := outer) (Evaluates.lookup localOuter)
      (by simpa [Mutable.scalarSequence, IntegerSets.encoded, List.map_map, Function.comp_def]
        using outerOne)
  have compared : Evaluates program env
      (.compare (.builtinCall .set [.name "outer"]) [(.notEq, .name "remaining_outer")]) heap
      (.scalar (.bool (!same))) heapOne := by
    exact Evaluates.compare (middle := heapOne) (afterRight := heapOne)
      firstSet (Evaluates.lookup localRemaining)
      (by simp [HeapCalls.compare, heapOne, remainingOne, same, ScalarSet.compare])
  have counted : Evaluates program env
      (.compare (.builtinCall .len [.builtinCall .set [.name "outer"]]) [(.notEq, .name "n")])
      heapOne (.scalar (.bool (decide (contents.length ≠ n)))) heapTwo := by
    apply Evaluates.compare (middle := heapTwo) (afterRight := heapTwo)
      (av := number contents.length) (bv := number n)
    · exact Evaluates.builtin (.cons secondSet .nil)
        (by simp [HeapCalls.primitive, Mutable.length, heapTwo])
    · exact Evaluates.lookup getN
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
  by_cases aligned : same = true
  · refine ⟨[.set contents, .set contents], ?_⟩
    change Evaluates program env (.boolOp .or _) heap _ _
    apply Evaluates.boolean_next (flag := false)
      (by simpa only [aligned, Bool.not_true] using compared) rfl rfl
    simpa only [validationFlag, same, aligned, Bool.not_true, Bool.false_or,
      contents, heapTwo, heapOne, List.append_assoc, List.singleton_append] using
      Evaluates.boolean_last counted rfl
  · have different : same = false := Bool.eq_false_iff.mpr aligned
    have mismatch : ScalarSet.compare .eq (IntegerSets.entries outer) remaining = false := different
    refine ⟨[.set contents], ?_⟩
    simpa only [validationFlag, mismatch, Bool.not_false, Bool.true_or, heapOne] using
      (Evaluates.boolean_stop (op := .or) (value := .scalar (.bool true))
        (rest := [.compare (.builtinCall .len [.builtinCall .set [.name "outer"]])
          [(.notEq, .name "n")]])
        (by simpa only [different, Bool.not_false] using compared) rfl rfl)

end solvers.«68»
