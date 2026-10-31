import solvers.«93.py.lean.d».SetFilter
import lang.py.FormalSW.MembershipCalls

namespace solvers.«93».consecutive_length
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.PositiveRun
set_option Elab.async false

abbrev increment : List Stmt := [.augAssign (.name "n") .add (.constant (.int 1))]
abbrev inIntegers : Expr := .membership (.name "n") (.name "integers") false
abbrev scanTail : List Stmt :=
  [.whileDo inIntegers increment [],
    .return (.tuple [.binOp (.name "n") .sub (.constant (.int 1)), .name "integers"])]
abbrev scanCode : List Stmt := .assign [.name "n"] (.constant (.int 1)) :: scanTail

theorem source_scan : ((findFunctionIn program.body "consecutive_length").getD ([], [])).2.drop 2 = scanCode := rfl

/-- The actual set-membership loop stops at the least missing positive integer. -/
theorem scan_loop (xs : List Int) (heap : Mutable.Heap) (address n : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[address]? = some (.set (xs.map Constant.int)))
    (positiveN : 0 < n) (within : n ≤ firstMissing xs)
    (getOut : env "integers" = some (.ref address)) (getN : env "n" = some (number n)) :
    BlockReturns program env scanTail heap (.ref heap.length)
      (heap ++ [.tuple [number (length xs), .ref address]]) globals := by
  induction measure : firstMissing xs - n using Nat.strong_induction_on generalizing n env with
  | h remaining ih =>
    have tested : Evaluates program env inIntegers heap
        (.scalar (.bool (decide ((n : Int) ∈ xs)))) heap globals :=
      Evaluates.membership (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getOut) .nil))
        (by simpa only [Bool.false_eq_true, if_false, ScalarSet.contains_ints] using
          membership_set heap address (.int n) (xs.map Constant.int) found false)
    by_cases finished : n = firstMissing xs
    · have absent : (n : Int) ∉ xs := finished ▸ missing xs
      apply BlockReturns.while_loop (middle := heap) (flag := false) tested
        (by simp [absent, Mutable.truth, scalarTruth])
      apply BlockReturns.return_value
      apply Evaluates.tuple (values := [number (length xs), .ref address])
      · apply Evaluates.Arguments.cons (middle := heap)
        · apply Evaluates.binary (Evaluates.lookup getN) Evaluates.constant
          have count : (n : Int) - 1 = (length xs : Int) := by
            have := firstMissing_eq_length xs
            omega
          simp [Mutable.binary, scalarBinary, count, number]
        · exact .cons (Evaluates.lookup getOut) .nil
      · simp
    · have earlier : n < firstMissing xs := by omega
      have present := before xs n positiveN earlier
      let next := setLocal env "n" (number (n + 1))
      have continued := ih (firstMissing xs - (n + 1)) (by omega) (n + 1) next
        (by omega) (by omega) (by simpa [next, setLocal] using getOut) (by simp [next, setLocal]) rfl
      have advanced := BlockReturns.augAssign getN (Evaluates.constant (value := .int 1))
        (show inplace .add (number n) (number 1) heap =
          some (.ok (number (n + 1)), heap) by simp [inplace, Mutable.binary, scalarBinary, number]) continued
      apply BlockReturns.while_loop (middle := heap) (flag := true) tested
        (by simp [present, Mutable.truth, scalarTruth])
      simpa only [scanTail, increment, Calls.loopContinuation, Calls.loopControl,
        Calls.loopControlStmt, Bool.false_or, Bool.false_eq_true, if_false] using advanced

/-- The returned tuple retains the same integer set, including any caller aliases. -/
theorem scan_result (xs : List Int) (heap : Mutable.Heap) (address : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[address]? = some (.set (xs.map Constant.int)))
    (getOut : env "integers" = some (.ref address)) :
    BlockReturns program env scanCode heap (.ref heap.length)
      (heap ++ [.tuple [number (length xs), .ref address]]) globals := by
  apply BlockReturns.assign Evaluates.constant
  apply scan_loop xs heap address 1 _ globals found (by decide) (positive xs)
  · simpa [setLocal] using getOut
  · simp [setLocal, number]

theorem collected_run (choices : ChoiceSchedule) (used : Nat) (remaining : List SequenceSource)
    (interpret : SequenceSource → Pair) :
    length (((SetCalls.selectionOrder choices used remaining).map interpret).foldl collectCanonicalInteger []) =
      length (canonicalIntegerResults (remaining.map interpret)) := by
  apply ProjectEulerSolutions.PositiveRun.congr
  intro n _
  simpa using collected_membership choices used remaining interpret [] n

/-- Filtering and the run scan compose through the real continuation of `all_results`. -/
theorem finish_key_set (choices : ChoiceSchedule) (heap : Mutable.Heap) (source address : Nat)
    (original : List (HashKey × Mutable.Value)) (interpret : SequenceSource → Pair)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (sourceFound : heap[source]? = some (.keySet original))
    (found : heap[address]? = some (.set []))
    (rows : ∀ entry ∈ original, Heap.Row heap entry.2 (interpret (sequenceSource entry.2)))
    (getOut : env "integers" = some (.ref address))
    (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ (integers : List Int) (finalUsed : Nat),
      (∀ n, n ∈ integers ↔ n ∈ canonicalIntegerResults
        ((original.map (fun entry => sequenceSource entry.2)).map interpret)) ∧
      SetCalls.Reaches choices program anchors
        ⟨⟨.value (.ref source), .unpackSequence ["numerator", "denominator"] filterBody [] scanCode env :: stack,
          heap⟩, used⟩
        ⟨⟨.value (.ref heap.length), stack,
          heap.set address (.set (integers.map Constant.int)) ++
            [.tuple [number (length (canonicalIntegerResults
              ((original.map (fun entry => sequenceSource entry.2)).map interpret))), .ref address]]⟩,
          finalUsed⟩ (globals := globals) := by
  let remaining := original.map (fun entry => sequenceSource entry.2)
  let integers := ((SetCalls.selectionOrder choices used remaining).map interpret).foldl collectCanonicalInteger []
  let after := heap.set address (.set (integers.map Constant.int))
  obtain ⟨finalEnv, finalUsed, finalOut, filtered⟩ := filter_key_set choices heap source address original interpret []
    env globals sourceFound found rows getOut scanCode anchors stack used
  have collected : after[address]? = some (.set (integers.map Constant.int)) := by
    simp [after, (List.getElem?_eq_some_iff.mp found).1]
  have scanned := scan_result integers after address finalEnv globals collected finalOut
  refine ⟨integers, finalUsed, ?_, ?_⟩
  · intro n
    simpa [integers, remaining] using collected_membership choices used remaining interpret [] n
  · have path := filtered.trans (SetCalls.Reaches.of_heap (scanned anchors stack) choices finalUsed)
    simpa only [after, integers, List.length_set, collected_run, remaining] using path

/-- Empty scalar storage follows the same source continuation and returns the unchanged collector. -/
theorem finish_empty (choices : ChoiceSchedule) (heap : Mutable.Heap) (source address : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (sourceFound : heap[source]? = some (.set [])) (found : heap[address]? = some (.set []))
    (getOut : env "integers" = some (.ref address))
    (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.value (.ref source), .unpackSequence ["numerator", "denominator"] filterBody [] scanCode env :: stack,
        heap⟩, used⟩
      ⟨⟨.value (.ref heap.length), stack, heap ++ [.tuple [number (length []), .ref address]]⟩, used⟩
      (globals := globals) :=
  (filter_empty choices heap source env globals sourceFound scanCode anchors stack used).trans
    (SetCalls.Reaches.of_heap (scan_result [] heap address env globals found getOut anchors stack) choices used)

end solvers.«93».consecutive_length
