import solvers.«96.py.lean.d».CandidatePeers
import ProjectEulerSolutions.Lib.FilteredRangeCollection
import lang.py.FormalSW.HeapqContents

namespace solvers.«96».reduce_cands.queue
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

abbrev candidate : Expr := .subscript (.name "cands") (.name "i")
abbrev singleTest : Expr := .compare (.builtinCall .intBitCount [candidate]) [(.eq, .constant (.int 1))]
abbrev collection : Expr := .listComp (.name "i") [("i", .call (.name "range") [.constant (.int 81)], [singleTest])]
abbrev prepare : List Stmt :=
  [.assign [.name "changed"] (.constant (.bool false)), .assign [.name "queue"] collection]
abbrev takeLast : List Stmt :=
  [.assign [.name "i"] (.call (.attribute (.name "queue") "pop") []),
   .assign [.name "val"] candidate]

theorem source_prepare : outerBody.take 2 = prepare := rfl
theorem source_take : queueBody.take 2 = takeLast := rfl

def single (candidates : Candidates) (index : Nat) : Bool :=
  if inside : index < 81 then decide ((candidates (Encoding.cell ⟨index, inside⟩)).card = 1) else false

def initial (candidates : Candidates) : List Mutable.Value :=
  IndexedFilter.select (single candidates) 0 ((List.range 81).map fun n : Nat => number n)

theorem initial_filter (candidates : Candidates) :
    initial candidates = ((List.range 81).filter (single candidates)).map fun n : Nat => number n := by
  have filtered (first count : Nat) :
      IndexedFilter.select (single candidates) first ((List.range' first count).map fun n : Nat => number n) =
        ((List.range' first count).filter (single candidates)).map fun n : Nat => number n := by
    induction count generalizing first with
    | zero => rfl
    | succ count ih =>
        simp only [List.range'_succ, List.map_cons, IndexedFilter.select, List.filter_cons]
        split <;> simp [ih]
  simpa [initial] using filtered 0 81

theorem initial_mem (candidates : Candidates) (position : Cell) :
    number (Encoding.index position).val ∈ initial candidates ↔ (candidates position).card = 1 := by
  rw [initial_filter, List.mem_map]
  constructor
  · rintro ⟨index, member, equal⟩
    have same : index = (Encoding.index position).val := Int.ofNat.inj (by simpa [number] using equal)
    subst index
    have selected := (List.mem_filter.mp member).2
    simpa [single, (Encoding.index position).isLt, Encoding.cell_index] using selected
  · intro selected
    refine ⟨(Encoding.index position).val, ?_, rfl⟩
    simp [single, (Encoding.index position).isLt, Encoding.cell_index, selected]

theorem initial_bound (candidates : Candidates) : (initial candidates).length ≤ 81 := by
  simpa using IndexedFilter.length_le (single candidates) 0 ((List.range 81).map fun n : Nat => number n)

theorem read (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (slot : Fin 81)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) (getIndex : env "i" = some (number slot.val)) :
    Evaluates program env candidate heap (number (Masks.encode (candidates (Encoding.cell slot)))) heap globals :=
  Evaluates.subscript (by intros; simp) (Evaluates.lookup getCandidates) (Evaluates.lookup getIndex)
    (get_mask address heap candidates slot found)

theorem single_test (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (slot : Fin 81)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) (getIndex : env "i" = some (number slot.val)) :
    Evaluates program env singleTest heap (.scalar (.bool (single candidates slot.val))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap)
    (av := number (candidates (Encoding.cell slot)).card) (bv := number 1)
  · apply Evaluates.builtin (.cons (read address heap candidates slot env globals found getCandidates getIndex) .nil)
    change (some (.ok (number (FormalSW.Py.Builtin.Int.bitCount (Masks.encode (candidates (Encoding.cell slot))))), heap) :
      Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
    rw [Masks.bitCount]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, single, slot.isLt]
    omega

theorem collect (address : Nat) (heap : Mutable.Heap) (candidates : Candidates)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) :
    Evaluates program env collection heap (.ref heap.length) (heap ++ [.list (initial candidates)]) globals := by
  apply FilteredRangeCollection.collect program heap env "i" (.name "i") singleTest
    ((List.range 81).map fun n : Nat => number n) (single candidates)
    [.constant (.int 81)] [number 81] (.cons Evaluates.constant .nil)
  · rfl
  · intro collected index inside
    simpa [List.getElem_map, List.getElem_range, setLocal] using
      (Evaluates.lookup (module := program) (globals := globals) (heap := HeapCollection.working heap collected [])
        (show setLocal (comprehensionEnv env [("i", .call (.name "range") [.constant (.int 81)], [singleTest])])
          "i" (number index) "i" = some (number index) by simp [setLocal]))
  · intro collected index inside
    have bound : index < 81 := by simpa using inside
    apply single_test address (HeapCollection.working heap collected []) candidates ⟨index, bound⟩
    · exact HeapCollection.prefix_found heap _ address _ found
    · simpa [setLocal, comprehensionEnv] using getCandidates
    · simp [setLocal]

theorem prepared (address : Nat) (heap : Mutable.Heap) (candidates : Candidates)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (found : View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) :
    BlockContinues program env prepare heap
      (setLocal (setLocal env "changed" (.scalar (.bool false))) "queue" (.ref heap.length))
      (heap ++ [.list (initial candidates)]) globals := by
  apply (BlockContinues.assign (name := "changed") (module := program) (globals := globals) Evaluates.constant).append
  exact BlockContinues.assign (collect address heap candidates _ globals found (by simpa [setLocal] using getCandidates))

theorem take_result (address queue : Nat) (heap : Mutable.Heap) (candidates : Candidates) (slot : Fin 81)
    (items : List Mutable.Value) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : View address heap candidates) (queued : heap[queue]? = some (.list (items ++ [number slot.val])))
    (different : address ≠ queue) (getCandidates : env "cands" = some (.ref address))
    (getQueue : env "queue" = some (.ref queue)) :
    BlockContinues program env takeLast heap
      (setLocal (setLocal env "i" (number slot.val)) "val" (number (Masks.encode (candidates (Encoding.cell slot)))))
      (heap.set queue (.list items)) globals := by
  have popped : Evaluates program env (.call (.attribute (.name "queue") "pop") []) heap (number slot.val)
      (heap.set queue (.list items)) globals := by
    apply Evaluates.pop getQueue
    · simp [bindMethod, queued]
    · exact .nil
    · change Mutable.popItem queue (.scalar (.int (-1))) heap = _
      exact Heapq.popItem_last heap queue items (number slot.val) queued
  apply (BlockContinues.assign (name := "i") popped).append
  apply BlockContinues.assign
  apply read address (heap.set queue (.list items)) candidates slot
  · exact (List.getElem?_set_ne different.symm).trans found
  · simpa [setLocal] using getCandidates
  · simp [setLocal]

end
end solvers.«96».reduce_cands.queue
