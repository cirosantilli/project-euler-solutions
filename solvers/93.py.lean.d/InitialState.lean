import solvers.«93.py.lean.d».StatePreparation
import lang.py.FormalSW.GeneratorCalls

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.HeapCollection
set_option Elab.async false

abbrev integerPairs (digits : List Int) : List Pair := digits.map (fun digit => (digit, 1))
abbrev digitPairExpr : Expr := .tuple [.name "digit", .constant (.int 1)]
abbrev digitGenerators : List (Expr × Expr × List Expr) := [(.name "digit", .name "digits", [])]
abbrev sortedDigitsExpr : Expr := .consumeGenerator (.builtinCall .sorted []) digitPairExpr digitGenerators
abbrev startExpr : Expr := .builtinCall .tuple [sortedDigitsExpr]

/-- Materializing the integer-pair generator retains all allocated tuples through collector writes. -/
theorem collect_digits (base : Mutable.Heap) (address : Nat) (digits pending : List Int)
    (found : base[address]? = some (.tuple (digits.map Heap.number))) (position : Nat)
    (suffix : digits.drop position = pending) (items : List Mutable.Value) (pairs : List Pair)
    (garbage : Mutable.Heap) (rows : Heap.Rows (working base items garbage) items pairs)
    (env : HeapCalls.Env) (globals : Mutable.Globals) :
    ∃ finalEnv finalItems finalGarbage,
      BlockContinues program env
        [.sequenceStepLoop "digit" (.ref address) position [.comprehensionAppend base.length digitPairExpr] []]
        (working base items garbage) finalEnv (working base finalItems finalGarbage) globals ∧
      Heap.Rows (working base finalItems finalGarbage) finalItems (pairs ++ integerPairs pending) := by
  have sequence (items : List Mutable.Value) (garbage : Mutable.Heap) :
      Mutable.sequence (.ref address) (working base items garbage) =
        some (.ok (digits.map Heap.number), working base items garbage) := by
    have original := prefix_found base (.list items :: garbage) address _ found
    simp [Mutable.sequence, original, working]
  induction pending generalizing position items pairs garbage env with
  | nil =>
    refine ⟨env, items, garbage, BlockContinues.sequence_done (source := .ref address) (sequence items garbage) ?_ .nil, ?_⟩
    · simpa [List.getElem?_map, suffix] using congrArg (Option.map Heap.number) (List.head?_drop (l := digits) (i := position)).symm
    · simpa using rows
  | cons digit rest ih =>
    let bound := setLocal env "digit" (Heap.number digit)
    let value := Mutable.Value.ref (working base items garbage).length
    let nextItems := items ++ [value]
    let nextGarbage := garbage ++ [Heap.object (digit, 1)]
    have tuple : Evaluates program bound digitPairExpr (working base items garbage) value
        (working base items garbage ++ [Heap.object (digit, 1)]) globals := by
      apply Evaluates.tuple (.cons (Evaluates.lookup (by simp [bound, setLocal])) (.cons Evaluates.constant .nil))
      simp
    have emitted := emit base items garbage [Heap.object (digit, 1)] value tuple
    have nextRow : Heap.Row (working base nextItems nextGarbage) value (digit, 1) := by
      refine ⟨(working base items garbage).length, rfl, ?_⟩
      simp [working, nextGarbage]
    have nextRows : Heap.Rows (working base nextItems nextGarbage) nextItems (pairs ++ [(digit, 1)]) :=
      List.rel_append (rows.collector nextItems [Heap.object (digit, 1)]) (.cons nextRow .nil)
    obtain ⟨finalEnv, finalItems, finalGarbage, continued, finalRows⟩ :=
      ih (position + 1) (by simp [List.drop_add_one_eq_tail_drop, suffix]) nextItems (pairs ++ [(digit, 1)]) nextGarbage nextRows bound
    refine ⟨finalEnv, finalItems, finalGarbage, ?_, ?_⟩
    · apply BlockContinues.sequence_step (source := .ref address) (value := Heap.number digit) (sequence items garbage)
      · simpa [List.getElem?_map, suffix] using congrArg (Option.map Heap.number) (List.head?_drop (l := digits) (i := position)).symm
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_or, Bool.false_eq_true, if_false] using emitted.append continued
    · simpa only [integerPairs, List.map_cons, List.append_assoc, List.singleton_append] using finalRows

/-- Sorting the generator changes only the order of the represented integer operands. -/
theorem initial_state (heap : Mutable.Heap) (address : Nat) (digits : List Int)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[address]? = some (.tuple (digits.map Heap.number)))
    (getDigits : env "digits" = some (.ref address)) :
    ∃ extra state items pairs,
      Evaluates program env startExpr heap (.ref state) (heap ++ extra) globals ∧
      (heap ++ extra)[state]? = some (.tuple items) ∧ Heap.Rows (heap ++ extra) items pairs ∧
      pairs.Perm (integerPairs digits) ∧ (∀ pair ∈ pairs, Canonical pair) := by
  let inside := comprehensionTargetsEnv env digitGenerators
  obtain ⟨finalEnv, items, garbage, collected, rows⟩ :=
    collect_digits heap address digits digits found 0 rfl [] [] [] .nil inside globals
  obtain ⟨sortedItems, sortedPairs, sorted, sortedRows, pairPermutation, _⟩ := rows.sort
  let gathered := working heap items garbage
  let ordered := gathered ++ [Mutable.Object.list sortedItems]
  have sortedExpression : Evaluates program env sortedDigitsExpr heap (.ref gathered.length) ordered globals := by
    apply Evaluates.consume_generator_sorted
    apply GeneratorCollects.sequence (target := .name "digit") (iterable := .name "digits")
      (filters := []) (generators := []) (finalEnv := finalEnv) (items := digits.map Heap.number)
      (source := .ref address) (middle := heap) (collectedHeap := gathered)
      (by intro expressions impossible; cases impossible) (by intro expressions impossible; cases impossible)
      (by intro expressions impossible; cases impossible) (Evaluates.lookup getDigits)
      (by simp [Mutable.sequence, found])
    · exact collected
    · exact primitive_sorted_values gathered heap.length items sortedItems (ProjectEulerSolutions.HeapCollection.found ..) sorted
  have orderedFound : ordered[gathered.length]? = some (.list sortedItems) := by simp [ordered]
  obtain ⟨state, extra, allocated, stateFound⟩ := allocate_tuple sortedItems ordered
  have executed : Evaluates program env startExpr heap (.ref state) (ordered ++ extra) globals := by
    apply Evaluates.builtin (.cons sortedExpression .nil)
    simpa [primitive, Mutable.sequence, orderedFound, Except.map] using allocated
  have finalRows := (sortedRows.append [.list sortedItems]).append extra
  refine ⟨(.list items :: garbage) ++ [.list sortedItems] ++ extra, state, sortedItems, sortedPairs, ?_, ?_, ?_, ?_, ?_⟩
  · simpa only [ordered, gathered, working, List.append_assoc] using executed
  · simpa only [ordered, gathered, working, List.append_assoc] using stateFound
  · simpa only [working, List.append_assoc] using finalRows
  · simpa using pairPermutation
  · intro pair member
    have original := pairPermutation.mem_iff.mp member
    obtain ⟨digit, _, rfl⟩ := List.mem_map.mp original
    simp [Canonical, value]

end solvers.«93».all_results
