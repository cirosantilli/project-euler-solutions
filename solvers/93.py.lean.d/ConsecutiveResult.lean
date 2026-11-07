import solvers.«93.py.lean.d».ResultAgreement
import solvers.«93.py.lean.d».Consecutive

namespace solvers.«93».consecutive_length
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.HeapCollection
set_option Elab.async false

/-- Decode the rational represented by an iterator's retained tuple object. -/
def readPair (heap : Mutable.Heap) (source : SequenceSource) : Pair :=
  match sequenceValue source with
  | .ref address => match heap[address]? with
    | some (.tuple [.scalar (.int a), .scalar (.int b)]) => (a, b)
    | _ => (0, 1)
  | _ => (0, 1)

theorem readPair_row (heap : Mutable.Heap) (entry : Heap.SetEntry)
    (found : heap[entry.2]? = some (Heap.object entry.1)) :
    readPair heap (sequenceSource (.ref entry.2)) = entry.1 := by
  simp [readPair, sequenceSource, sequenceValue, found, Heap.object]

/-- Filtering a complete result set yields the reference score, in every set order. -/
theorem finish_result (digits : List Int) (heap : Mutable.Heap) (source address : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (complete : Heap.ResultAt heap source (all_results.integerPairs digits))
    (found : heap[address]? = some (.set [])) (getOut : env "integers" = some (.ref address))
    (choices : ChoiceSchedule) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ (integers : List Int) (finalUsed : Nat),
      SetCalls.Reaches choices program anchors
        ⟨⟨.value (.ref source), .unpackSequence ["numerator", "denominator"] filterBody [] scanCode env :: stack,
          heap⟩, used⟩
        ⟨⟨.value (.ref heap.length), stack,
          heap.set address (.set (integers.map Constant.int)) ++
            [.tuple [number (all_results.Reference.runLength digits), .ref address]]⟩, finalUsed⟩
        (globals := globals) := by
  obtain ⟨entries, stored, members⟩ := complete
  have decoded : ((entries.map Heap.encodeEntry).map (fun entry => sequenceSource entry.2)).map (readPair heap) =
      entries.map Prod.fst := by
    simp only [List.map_map]
    apply List.map_congr_left
    intro entry member
    exact readPair_row heap entry (stored.rows entry member)
  rcases stored.stored with keySet | ⟨rfl, empty⟩
  · have rows : ∀ entry ∈ entries.map Heap.encodeEntry,
        Heap.Row heap entry.2 (readPair heap (sequenceSource entry.2)) := by
      intro entry member
      obtain ⟨original, present, rfl⟩ := List.mem_map.mp member
      rw [show readPair heap (sequenceSource (Heap.encodeEntry original).2) = original.1 from
        readPair_row heap original (stored.rows original present)]
      exact ⟨original.2, rfl, stored.rows original present⟩
    obtain ⟨integers, finalUsed, _, finished⟩ := finish_key_set choices heap source address
      (entries.map Heap.encodeEntry) (readPair heap) env globals keySet found rows getOut anchors stack used
    refine ⟨integers, finalUsed, ?_⟩
    simpa only [decoded, all_results.run_length_agrees digits (entries.map Prod.fst) members] using finished
  · have score : ProjectEulerSolutions.PositiveRun.length [] = all_results.Reference.runLength digits := by
      simpa only [canonicalIntegerResults, List.flatMap_nil] using all_results.run_length_agrees digits [] members
    have unchanged : heap.set address (.set []) = heap := by
      obtain ⟨inside, entry⟩ := List.getElem?_eq_some_iff.mp found
      rw [← entry, List.set_getElem_self inside]
    refine ⟨[], used, ?_⟩
    simpa only [List.map_nil, unchanged, score] using
      finish_empty choices heap source address env globals empty found getOut anchors stack used

abbrev sourceArgs := ((findFunctionIn program.body "consecutive_length").getD ([], [])).1
abbrev sourceBody := ((findFunctionIn program.body "consecutive_length").getD ([], [])).2
abbrev filterCode : List Stmt :=
  .forIn (.tuple [.name "numerator", .name "denominator"])
    (.call (.name "all_results") [.name "digits"]) filterBody [] :: scanCode

theorem source_body : executableBody sourceBody =
    .assign [.name "integers"] (.builtinCall .set []) :: filterCode := rfl

/-- Complete scoring call: only its new collector is mutated; all caller objects survive. -/
theorem result (digits : List Int) (heap : Mutable.Heap) (address : Nat) (globals : Mutable.Globals)
    (input : heap[address]? = some (.tuple (digits.map Heap.number)))
    (choices : ChoiceSchedule) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ (extra : Mutable.Heap) (tupleAddress setAddress finalUsed : Nat),
      (heap ++ extra)[tupleAddress]? =
        some (.tuple [number (all_results.Reference.runLength digits), .ref setAddress]) ∧
      SetCalls.Reaches choices program anchors
        ⟨⟨.operation (.call "consecutive_length") [.ref address], stack, heap⟩, used⟩
        ⟨⟨.value (.ref tupleAddress), stack, heap ++ extra⟩, finalUsed⟩ (globals := globals) := by
  let entry := functionEnv sourceArgs sourceBody [.ref address] globals
  let initialized := setLocal entry "integers" (.ref heap.length)
  let ready := heap ++ [.set []]
  have getDigits : initialized "digits" = some (.ref address) := by cases globals <;> rfl
  obtain ⟨source, garbage, _, generated, complete⟩ :=
    all_results.result digits address ready globals (prefix_found heap [.set []] address _ input)
  let middle := ready ++ garbage
  have collector : middle[heap.length]? = some (.set []) := by simp [middle, ready, List.append_assoc]
  have callResult : Evaluates program initialized (.call (.name "all_results") [.name "digits"])
      ready (.ref source) middle globals :=
    Evaluates.call (.cons (Evaluates.lookup getDigits) .nil) generated
  obtain ⟨integers, finalUsed, finished⟩ := finish_result digits middle source heap.length initialized globals complete
    collector (by simp [initialized, setLocal]) choices anchors stack used
  have replaced : middle.set heap.length (.set (integers.map Constant.int)) =
      heap ++ .set (integers.map Constant.int) :: garbage := by
    simp [middle, ready, List.append_assoc]
  refine ⟨.set (integers.map Constant.int) :: garbage ++
      [.tuple [number (all_results.Reference.runLength digits), .ref heap.length]], middle.length, heap.length,
      finalUsed, ?_, ?_⟩
  · simp [middle, ready, List.length_append]
  · apply SetCalls.Reaches.next (following := ⟨⟨.block (executableBody sourceBody) entry, stack, heap⟩, used⟩)
    · rfl
    · have allocated := BlockContinues.assign (name := "integers") (module := program) (env := entry) (globals := globals)
        (Evaluates.builtin .nil (show primitive (.builtin .set) [] heap =
          some (.ok (.ref heap.length), ready) from rfl))
      apply (SetCalls.Reaches.of_heap (allocated filterCode anchors stack) choices used).trans
      apply SetCalls.Reaches.next (following := ⟨⟨.expr (.call (.name "all_results") [.name "digits"])
        initialized, .unpackSequence ["numerator", "denominator"] filterBody [] scanCode initialized :: stack,
        ready⟩, used⟩) rfl
      apply (SetCalls.Reaches.of_heap (callResult anchors _) choices used).trans
      simpa only [replaced, List.append_assoc, List.cons_append] using finished

end solvers.«93».consecutive_length
