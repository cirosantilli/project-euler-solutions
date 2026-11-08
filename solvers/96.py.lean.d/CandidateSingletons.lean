import solvers.«96.py.lean.d».CandidateQueueLoop

namespace solvers.«96».reduce_cands.singletons
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku
set_option Elab.async false
noncomputable section

def cells (candidates : Candidates) : List Cell :=
  ((List.finRange 81).filter fun slot => queue.single candidates slot.val).map Encoding.cell

def initial (candidates : Candidates) : Propagation.State := ⟨candidates, cells candidates, false⟩

theorem encoding (candidates : Candidates) : queue.initial candidates = peerLoop.encoded (cells candidates) := by
  have indices : (List.finRange 81).map Fin.val = List.range 81 := by
    apply List.ext_getElem
    · simp
    · intros
      simp
  rw [queue.initial_filter, ← indices, List.filter_map, List.map_map]
  simp only [cells, peerLoop.encoded, List.map_map, Function.comp_def, peerLoop.atom, Encoding.index_cell, number]

theorem singles (candidates : Candidates) : Propagation.Singles (initial candidates) := by
  intro cell member
  obtain ⟨slot, selected, rfl⟩ := List.mem_map.mp member
  have single := (List.mem_filter.mp selected).2
  simpa [queue.single, slot.isLt] using single

theorem separation (table : PeerTable) (address : Nat) (heap : Mutable.Heap) (candidates : Candidates)
    (view : table.View heap) (found : reduce_cands.View address heap candidates) :
    address ≠ heap.length ∧ table.Separate address heap.length := by
  have inside := (List.getElem?_eq_some_iff.mp found).1
  refine ⟨by omega, ⟨?_, ?_, ?_, ?_⟩⟩
  · intro same
    have objects := view.root
    rw [← same, found] at objects
    have lists : values candidates = table.references := by simpa using objects
    have candidateHead : (values candidates)[0]? = some (number (Masks.encode (candidates (Encoding.cell 0)))) := by
      simp only [values, List.getElem?_map]
      rfl
    have referenceHead : table.references[0]? = some (.ref (table.address (Encoding.cell 0))) := by
      simp only [PeerTable.references, List.getElem?_ofFn, dif_pos (by decide : 0 < 81)]
      rfl
    have heads : (values candidates)[0]? = table.references[0]? := congrArg (fun items : List Mutable.Value => items[0]?) lists
    rw [candidateHead, referenceHead] at heads
    cases heads
  · have bound := (List.getElem?_eq_some_iff.mp view.root).1
    omega
  · intro cell same
    have objects := view.sets cell
    rw [← same, found] at objects
    cases objects
  · intro cell
    have bound := (List.getElem?_eq_some_iff.mp (view.sets cell)).1
    omega

def ready (env : HeapCalls.Env) (heap : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal env "changed" (.scalar (.bool false))) "queue" (.ref heap.length)

def work (heap : Mutable.Heap) (candidates : Candidates) : Mutable.Heap :=
  heap ++ [.list (peerLoop.encoded (cells candidates))]

theorem prepare (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (found : reduce_cands.View address heap candidates)
    (getCandidates : env "cands" = some (.ref address)) :
    BlockContinues program env queue.prepare heap (ready env heap) (work heap candidates) globals := by
  simpa only [encoding] using queue.prepared address heap candidates env globals found getCandidates

theorem view (address : Nat) (heap : Mutable.Heap) (candidates : Candidates)
    (found : reduce_cands.View address heap candidates) :
    peerLoop.View address heap.length (work heap candidates) (initial candidates) :=
  ⟨HeapCollection.prefix_found heap _ address _ found, by simp [work, initial]⟩

theorem locals (table : PeerTable) (address : Nat) (heap : Mutable.Heap) (candidates : Candidates) (env : HeapCalls.Env)
    (getCandidates : env "cands" = some (.ref address)) (getPeers : env "PEERS" = some (.ref table.root)) :
    queueLoop.Locals table address heap.length (initial candidates) (ready env heap) := by
  constructor
  · simpa [ready, setLocal] using getCandidates
  · simp [ready, setLocal]
  · simp [ready, setLocal, initial]
  · simpa [ready, setLocal] using getPeers

theorem table_view (table : PeerTable) (heap : Mutable.Heap) (candidates : Candidates)
    (view : table.View heap) : table.View (work heap candidates) :=
  ⟨HeapCollection.prefix_found heap _ table.root _ view.root,
    fun cell => HeapCollection.prefix_found heap _ (table.address cell) _ (view.sets cell)⟩

theorem source : outerBody.take 3 = queue.prepare ++ [queueLoop.statement] := rfl

/-- Initialize the queue and execute the complete singleton phase from the actual outer-loop prefix. -/
theorem execute (choices : ChoiceSchedule) (globals : Mutable.Globals) (table : PeerTable) (address : Nat)
    (candidates : Candidates) (heap : Mutable.Heap) (env : HeapCalls.Env) (used : Nat)
    (found : reduce_cands.View address heap candidates) (tableView : table.View heap)
    (getCandidates : env "cands" = some (.ref address)) (getPeers : env "PEERS" = some (.ref table.root))
    (nonempty : ∀ cell, candidates cell ≠ ∅)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    ∃ finalHeap finalEnv finalUsed result, finalHeap.length = heap.length + 1 ∧
      (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
      (∀ name, name ∉ ["queue", "changed", "i", "val", "p"] → finalEnv name = env name) ∧
      queueLoop.outcome choices globals table address heap.length (initial candidates) result
        ⟨⟨.block (outerBody.take 3 ++ following) env, stack, heap⟩, used⟩
        finalEnv finalHeap finalUsed following anchors stack := by
  obtain ⟨different, separate⟩ := separation table address heap candidates tableView found
  obtain ⟨finalHeap, finalEnv, finalUsed, result, frame, envFrame, executed⟩ :=
    queueLoop.run choices globals table address heap.length (initial candidates) (work heap candidates) (ready env heap) used
      (view address heap candidates found) (locals table address heap candidates env getCandidates getPeers)
      (table_view table heap candidates tableView) separate different nonempty (singles candidates) following anchors stack
  have entry : SetCalls.Reaches choices program anchors
      ⟨⟨.block (outerBody.take 3 ++ following) env, stack, heap⟩, used⟩
      ⟨⟨.block (queueLoop.statement :: following) (ready env heap), stack, work heap candidates⟩, used⟩ globals := by
    simpa only [source, List.append_assoc, List.singleton_append] using
      SetCalls.Reaches.of_heap (prepare address heap candidates env globals found getCandidates
        (queueLoop.statement :: following) anchors stack) choices used
  refine ⟨finalHeap, finalEnv, finalUsed, result, ?_, ?_, ?_, ?_⟩
  · simpa [work] using frame.length
  · intro index inside different
    exact (frame.outside index different (by omega)).trans (List.getElem?_append_left inside)
  · intro name outside
    have away : name ≠ "queue" ∧ name ≠ "changed" ∧ name ≠ "i" ∧ name ≠ "val" ∧ name ≠ "p" := by simpa using outside
    rw [envFrame name (by simp [away.2.2.1, away.2.2.2.1, away.2.2.2.2, away.2.1])]
    simp [ready, setLocal, away.1, away.2.1]
  · cases result with
    | none => exact ⟨executed.1, entry.trans executed.2⟩
    | some after =>
        rcases executed with ⟨finalView, finalLocals, progress, finalNonempty, finalSingles, empty, equivalent, path⟩
        exact ⟨finalView, finalLocals, progress, finalNonempty, finalSingles, empty, equivalent, entry.trans path⟩

end
end solvers.«96».reduce_cands.singletons
