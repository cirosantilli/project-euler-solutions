import solvers.«68.combination.source»
import lang.py.FormalSW.CombinationCalls

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRing MagicRingTextCalls
set_option Elab.async false

abbrev solverArgs : List Arg := ((findFunctionIn program.body "solve").getD ([], [])).1
abbrev solverBody : List Stmt := ((findFunctionIn program.body "solve").getD ([], [])).2
abbrev solverCode : List Stmt := executableBody solverBody
abbrev selectionLoop : Stmt := .forIn (.name "inner_set")
  (.combinationsIterable [.name "labels", .name "n"]) combinationBody []

def capturedSelections (rows : List (List Nat)) : List (List SequenceSource) :=
  rows.map (fun row => (encode row).map sequenceSource)

theorem selection_loop_source : solverCode[6]? = some selectionLoop := rfl

theorem selections_result (n : Nat) (positive : 0 < n) (digits : Int)
    (rows : List (List Nat)) (valid : ∀ row ∈ rows, row.Sublist (labels n) ∧ row.length = n)
    (best : String) (heap : Mutable.Heap) (env : HeapCalls.Env) (labelsAddress numsAddress : Nat)
    (getNums : heap[numsAddress]? = some (.set (IntegerSets.entries (ints (labels n)))))
    (ready : OuterReady n digits best labelsAddress numsAddress env) :
    ∃ finalHeap, BlockReturns program env
      [.permutationStepLoop "inner_set" (capturedSelections rows) combinationBody [],
       .return (.name "best")]
      heap (string (prunedSelectionScan n digits rows best)) finalHeap := by
  induction rows generalizing best heap env with
  | nil =>
      exact ⟨heap, BlockReturns.permutation_done
        (BlockReturns.return_value (Evaluates.lookup ready.getBest))⟩
  | cons selected rest ih =>
      obtain ⟨selectedSublist, size⟩ := valid selected List.mem_cons_self
      have distinct : selected.Nodup := selectedSublist.nodup List.nodup_range'
      have subset : ∀ value ∈ selected, value ∈ labels n := selectedSublist.subset
      have nonempty : (encode selected).isEmpty = false := by
        apply List.isEmpty_eq_false_iff.mpr
        intro empty
        have length := congrArg List.length empty
        simp only [encode, List.length_map, List.length_nil] at length
        omega
      let tupleHeap := heap ++ [.tuple (encode selected)]
      let selectedEnv := setLocal env "inner_set" (.ref heap.length)
      let next : Stmt :=
        .permutationStepLoop "inner_set" (capturedSelections rest) combinationBody []
      have selectedReady : OuterReady n digits best labelsAddress numsAddress selectedEnv :=
        ready.set_auxiliary "inner_set" _ (by decide)
      have tupleNums : tupleHeap[numsAddress]? =
          some (.set (IntegerSets.entries (ints (labels n)))) :=
        ProjectEulerSolutions.HeapCollection.prefix_found heap _ _ _ getNums
      have post (nextEnv : HeapCalls.Env) (extra : Mutable.Heap)
          (nextReady : OuterReady n digits (selectionVisit n digits best selected)
            labelsAddress numsAddress nextEnv) :
          ∃ finalHeap, BlockReturns program nextEnv [next, .return (.name "best")]
            (tupleHeap ++ extra) (string (prunedSelectionScan n digits (selected :: rest) best))
            finalHeap := by
        obtain ⟨finalHeap, continued⟩ := ih
          (fun entry member => valid entry (List.mem_cons_of_mem _ member))
          (selectionVisit n digits best selected) (tupleHeap ++ extra) nextEnv
          (ProjectEulerSolutions.HeapCollection.prefix_found tupleHeap extra _ _ tupleNums)
          nextReady
        exact ⟨finalHeap, continued⟩
      obtain ⟨finalHeap, advanced⟩ := combination_result n positive digits best selected
        size distinct subset tupleHeap selectedEnv labelsAddress numsAddress heap.length
        tupleNums (by simp [tupleHeap]) rfl selectedReady next [.return (.name "best")]
        (string (prunedSelectionScan n digits (selected :: rest) best)) post
      refine ⟨finalHeap, ?_⟩
      apply BlockReturns.permutation_step (tuple := .ref heap.length) (middle := tupleHeap)
      · simp [List.map_map, Function.comp_def, Mutable.allocateTuple, nonempty, tupleHeap,
          Except.map]
      · exact advanced

theorem selection_loop_result (n digits : Nat) (positive : 0 < n)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (labelsAddress numsAddress : Nat)
    (getLabels : heap[labelsAddress]? = some (.list (encode (labels n))))
    (getNums : heap[numsAddress]? = some (.set (IntegerSets.entries (ints (labels n)))))
    (ready : OuterReady n digits "" labelsAddress numsAddress env) :
    ∃ finalHeap, BlockReturns program env [selectionLoop, .return (.name "best")]
      heap (string (optimizedAnswer n digits)) finalHeap := by
  obtain ⟨finalHeap, scanned⟩ := selections_result n positive digits
    (Combinations.rows n (labels n))
    (fun row member => (Combinations.mem_rows_iff _ _ _).mp member) ""
    heap env labelsAddress numsAddress getNums ready
  rw [prunedSelectionScan_answer] at scanned
  have acquired : combinationRows [.ref labelsAddress, number n] heap =
      some (.ok (capturedSelections (Combinations.rows n (labels n))), heap) := by
    simpa only [capturedSelections, encode, Combinations.rows_map,
      List.map_map, Function.comp_def] using
      combinationRows_list heap labelsAddress n (encode (labels n)) getLabels
  exact ⟨finalHeap, BlockReturns.for_combinations
    (.cons (Evaluates.lookup ready.getLabels) (.cons (Evaluates.lookup ready.getN) .nil))
    acquired scanned⟩

end solvers.«68»
