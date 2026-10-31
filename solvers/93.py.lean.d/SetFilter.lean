import solvers.«93.py.lean.d».IntegerFilter
import ProjectEulerSolutions.Lib.RationalHeap
import lang.py.FormalSW.SetChoiceCalls

namespace solvers.«93».consecutive_length
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
set_option Elab.async false

abbrev unpackFilter : List Stmt :=
  comprehensionTargetBody (.tuple [.name "numerator", .name "denominator"]) filterBody

private theorem set_same (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨inside, entry⟩ := List.getElem?_eq_some_iff.mp found
  rw [← entry, List.set_getElem_self inside]

theorem collect_ints (heap : Mutable.Heap) (address : Nat) (before : List Int) (pair : Pair)
    (found : heap[address]? = some (.set (before.map Constant.int))) :
    collect heap address (before.map Constant.int) pair =
      heap.set address (.set ((collectCanonicalInteger before pair).map Constant.int)) := by
  by_cases accepted : pair.2 = 1 ∧ 0 < pair.1
  · simp only [collect, collectCanonicalInteger, if_pos accepted, ScalarSet.insert_ints]
  · simpa only [collect, collectCanonicalInteger, if_neg accepted] using
      (set_same heap address _ found).symm

/-- The hidden iterator binding is unpacked before testing and collecting the rational value. -/
theorem unpack_filter_result (source : SequenceSource) (pair : Pair) (heap : Mutable.Heap)
    (address : Nat) (before : List Int) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (row : Heap.Row heap (sequenceValue source) pair)
    (found : heap[address]? = some (.set (before.map Constant.int)))
    (getOut : env "integers" = some (.ref address)) :
    BlockContinues program (setLocal env "$formalsw.unpack" (sequenceValue source)) unpackFilter heap
      (assignNames (setLocal env "$formalsw.unpack" (sequenceValue source))
        ["numerator", "denominator"] [number pair.1, number pair.2])
      (heap.set address (.set ((collectCanonicalInteger before pair).map Constant.int))) globals := by
  obtain ⟨tupleAddress, value, tupleFound⟩ := row
  have unpacked := BlockContinues.unpack (module := program) (globals := globals)
    (heap := heap) (middle := heap) (finalHeap := heap) (expressions := [.name "numerator", .name "denominator"])
    (names := ["numerator", "denominator"]) (items := [number pair.1, number pair.2]) rfl
    (Evaluates.lookup (show (setLocal env "$formalsw.unpack" (sequenceValue source)) "$formalsw.unpack" =
      some (sequenceValue source) by simp [setLocal]))
    (by simp [value, Mutable.sequence, tupleFound, Heap.object, Heap.number, number]) rfl
  have filtered := filter_result pair heap address (before.map Constant.int)
    (assignNames (setLocal env "$formalsw.unpack" (sequenceValue source))
      ["numerator", "denominator"] [number pair.1, number pair.2]) globals found
    (by simp [assignNames, setLocal, getOut]) (by simp [assignNames, setLocal]) (by simp [assignNames, setLocal])
  rw [collect_ints heap address before pair found] at filtered
  exact unpacked.append filtered

/-- Updating the integer collector cannot change any of the rational tuple objects. -/
private theorem row_set (heap : Mutable.Heap) (address : Nat) (before after : List Int)
    (found : heap[address]? = some (.set (before.map Constant.int)))
    (source : SequenceSource) (pair : Pair) (row : Heap.Row heap (sequenceValue source) pair) :
    Heap.Row (heap.set address (.set (after.map Constant.int))) (sequenceValue source) pair := by
  obtain ⟨tupleAddress, value, tupleFound⟩ := row
  have different : address ≠ tupleAddress := by
    intro equal
    subst tupleAddress
    simp [found, Heap.object] at tupleFound
  refine ⟨tupleAddress, value, ?_⟩
  simpa [List.getElem?_set, different, Ne.symm different] using tupleFound

/-- Every traversal order executes the actual unpacking/filtering loop and changes only its collector. -/
theorem filter_loop (choices : ChoiceSchedule) (remaining : List SequenceSource)
    (interpret : SequenceSource → Pair) (heap : Mutable.Heap) (source address : Nat)
    (original : List (HashKey × Mutable.Value)) (before : List Int)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (sourceFound : heap[source]? = some (.keySet original))
    (found : heap[address]? = some (.set (before.map Constant.int)))
    (rows : ∀ item ∈ remaining, Heap.Row heap (sequenceValue item) (interpret item))
    (getOut : env "integers" = some (.ref address))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ finalEnv finalUsed,
      finalEnv "integers" = some (.ref address) ∧
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (.keySetStepLoop "$formalsw.unpack" source original.length remaining unpackFilter [] :: following)
          env, stack, heap⟩, used⟩
        ⟨⟨.block following finalEnv, stack,
          heap.set address (.set ((((SetCalls.selectionOrder choices used remaining).map interpret).foldl
            collectCanonicalInteger before).map Constant.int))⟩, finalUsed⟩ (globals := globals) := by
  cases remaining with
  | nil =>
    refine ⟨env, used, getOut, ?_⟩
    simp only [SetCalls.selectionOrder, List.map_nil, List.foldl_nil, set_same heap address _ found]
    exact .next (by simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, Mutable.keyEntries, sourceFound]) .refl
  | cons head tail =>
    let index := choices used % (head :: tail).length
    have bound : index < (head :: tail).length := Nat.mod_lt _ (by simp)
    let item := (head :: tail)[index]
    let rest := (head :: tail).eraseIdx index
    let pair := interpret item
    let inserted := collectCanonicalInteger before pair
    let middle := heap.set address (.set (inserted.map Constant.int))
    let boundEnv := setLocal env "$formalsw.unpack" (sequenceValue item)
    let nextEnv := assignNames boundEnv ["numerator", "denominator"] [number pair.1, number pair.2]
    have different : address ≠ source := by
      intro equal
      subst source
      simp [found] at sourceFound
    have nextFound : middle[address]? = some (.set (inserted.map Constant.int)) := by
      simp [middle, (List.getElem?_eq_some_iff.mp found).1]
    have nextSource : middle[source]? = some (.keySet original) := by
      simpa [middle, List.getElem?_set, different, Ne.symm different] using sourceFound
    have nextRows : ∀ value ∈ rest, Heap.Row middle (sequenceValue value) (interpret value) := by
      intro value member
      exact row_set heap address before inserted found value (interpret value)
        (rows value (List.mem_of_mem_eraseIdx member))
    have nextOut : nextEnv "integers" = some (.ref address) := by
      simp [nextEnv, boundEnv, assignNames, setLocal, getOut]
    obtain ⟨finalEnv, finalUsed, finalOut, continued⟩ := filter_loop choices rest interpret middle source address
      original inserted nextEnv globals nextSource nextFound nextRows nextOut following anchors stack (used + 1)
    refine ⟨finalEnv, finalUsed, finalOut, ?_⟩
    have emitted := unpack_filter_result item pair heap address before env globals
      (rows item (List.getElem_mem bound)) found getOut
    have selected : (head :: tail)[choices used % (tail.length + 1)]? = some item := List.getElem?_eq_getElem bound
    apply SetCalls.Reaches.next (following := ⟨⟨.block
      (unpackFilter ++ [.keySetStepLoop "$formalsw.unpack" source original.length rest unpackFilter []] ++ following)
      boundEnv, stack, heap⟩, used + 1⟩)
    · simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, Mutable.keyEntries, sourceFound,
        selected, index, rest, boundEnv, unpackFilter, comprehensionTargetBody, filterBody, emit,
        Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt]
    · have bodyPath := SetCalls.Reaches.of_heap
        (emitted (.keySetStepLoop "$formalsw.unpack" source original.length rest unpackFilter [] :: following)
          anchors stack) choices (used + 1)
      have path := bodyPath.trans continued
      simpa [SetCalls.selectionOrder, index, rest, item, pair, inserted, middle, nextEnv, boundEnv,
        List.map_cons, List.foldl_cons, List.append_assoc] using path
termination_by remaining.length
decreasing_by
  simp_wf
  subst remaining
  change ((head :: tail).eraseIdx index).length < (head :: tail).length
  rw [List.length_eraseIdx_of_lt bound]
  simp

/-- The resulting integer membership is independent of the selected traversal order. -/
theorem collected_membership (choices : ChoiceSchedule) (used : Nat) (remaining : List SequenceSource)
    (interpret : SequenceSource → Pair) (before : List Int) (n : Int) :
    n ∈ ((SetCalls.selectionOrder choices used remaining).map interpret).foldl collectCanonicalInteger before ↔
      n ∈ before ∨ n ∈ canonicalIntegerResults (remaining.map interpret) := by
  rw [mem_fold_collectCanonicalInteger]
  have same : (canonicalIntegerResults ((SetCalls.selectionOrder choices used remaining).map interpret)).Perm
      (canonicalIntegerResults (remaining.map interpret)) :=
    ((SetCalls.selectionOrder_perm choices used remaining).map interpret).flatMap
      (fun pair _ => List.Perm.refl (canonicalIntegerResult pair))
  exact or_congr Iff.rfl same.mem_iff

/-- Canonical rational keys contribute exactly their positive integer values. -/
theorem collected_values (choices : ChoiceSchedule) (used : Nat) (remaining : List SequenceSource)
    (interpret : SequenceSource → Pair) (before : List Int) (n : Int)
    (canonical : ∀ item ∈ remaining, Canonical (interpret item)) :
    n ∈ ((SetCalls.selectionOrder choices used remaining).map interpret).foldl collectCanonicalInteger before ↔
      n ∈ before ∨ (0 < n ∧ (n : Rat) ∈ (remaining.map interpret).map value) := by
  rw [collected_membership, mem_canonicalIntegerResults]
  intro pair member
  obtain ⟨item, present, rfl⟩ := List.mem_map.mp member
  exact canonical item present

/-- Resume the actual tuple-unpacking continuation after the result-producing call returns. -/
theorem filter_key_set (choices : ChoiceSchedule) (heap : Mutable.Heap) (source address : Nat)
    (original : List (HashKey × Mutable.Value)) (interpret : SequenceSource → Pair) (before : List Int)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (sourceFound : heap[source]? = some (.keySet original))
    (found : heap[address]? = some (.set (before.map Constant.int)))
    (rows : ∀ entry ∈ original, Heap.Row heap entry.2 (interpret (sequenceSource entry.2)))
    (getOut : env "integers" = some (.ref address))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ finalEnv finalUsed,
      finalEnv "integers" = some (.ref address) ∧
      SetCalls.Reaches choices program anchors
        ⟨⟨.value (.ref source), .unpackSequence ["numerator", "denominator"] filterBody [] following env :: stack,
          heap⟩, used⟩
        ⟨⟨.block following finalEnv, stack,
          heap.set address (.set ((((SetCalls.selectionOrder choices used
            (original.map (fun entry => sequenceSource entry.2))).map interpret).foldl collectCanonicalInteger before).map Constant.int))⟩, finalUsed⟩ (globals := globals) := by
  have available : ∀ item ∈ original.map (fun entry => sequenceSource entry.2),
      Heap.Row heap (sequenceValue item) (interpret item) := by
    intro item member
    obtain ⟨entry, present, rfl⟩ := List.mem_map.mp member
    simpa using rows entry present
  obtain ⟨finalEnv, finalUsed, finalOut, loop⟩ := filter_loop choices _ interpret heap source address original before
    env globals sourceFound found available getOut following anchors stack used
  refine ⟨finalEnv, finalUsed, finalOut, .next ?_ loop⟩
  simp [SetCalls.step, HeapCalls.step, HeapCalls.run, Mutable.sequence, sourceFound,
    SetCalls.iteratorStep, Mutable.unsupported, Mutable.keyEntries, unpackFilter]

/-- An empty result uses scalar-set storage and leaves the integer collector unchanged. -/
theorem filter_empty (choices : ChoiceSchedule) (heap : Mutable.Heap) (source : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : heap[source]? = some (.set []))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.value (.ref source), .unpackSequence ["numerator", "denominator"] filterBody [] following env :: stack,
        heap⟩, used⟩
      ⟨⟨.block following env, stack, heap⟩, used⟩ (globals := globals) := by
  apply SetCalls.Reaches.next (following := ⟨⟨.block
    (.keySetStepLoop "$formalsw.unpack" source 0 [] unpackFilter [] :: following) env, stack, heap⟩, used⟩)
  · simp [SetCalls.step, HeapCalls.step, HeapCalls.run, Mutable.sequence, found,
      SetCalls.iteratorStep, Mutable.unsupported, Mutable.keyEntries, unpackFilter]
  · exact .next (by simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, Mutable.keyEntries, found]) .refl

end solvers.«93».consecutive_length
