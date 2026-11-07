import solvers.«96.py.lean.d».PeerFrame

namespace solvers.«96».build_peers_units.peerBuild
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

/-- Every cell in the current unit receives its own in-place peer-set update. -/
theorem unit_loop (root unitAddress setAddress : Nat) (cells : Fin 9 → Cell)
    (unitRelated : ∀ i j, i ≠ j → Peer (cells i) (cells j)) (before remaining : List (Fin 9))
    (keys : Keys) (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (known : View root heap keys) (allRelated : ∀ cell, Peers.Related cell (keys cell))
    (found : heap[unitAddress]? = some (.list (peerLoop.encoded ((before ++ remaining).map cells))))
    (setFound : heap[setAddress]? = some (.set (unitKeys cells))) (separate : ∀ cell, address root cell ≠ setAddress)
    (getPeers : env "peers" = some (.ref root)) (getSet : env "s" = some (.ref setAddress)) :
    ∃ finalHeap finalEnv finalKeys,
      BlockContinues program env [.sequenceStepLoop "i" (.ref unitAddress) before.length [statement] []]
        heap finalEnv finalHeap globals ∧
      View root finalHeap finalKeys ∧ (∀ cell, Peers.Related cell (finalKeys cell)) ∧
      Frame root heap finalHeap ∧ (∀ name, name ≠ "i" → finalEnv name = env name) := by
  induction remaining generalizing before keys heap env with
  | nil =>
      refine ⟨heap, env, keys, ?_, known, allRelated, .refl root heap, fun _ _ => rfl⟩
      apply BlockContinues.sequence_done (items := peerLoop.encoded (before.map cells))
      · simpa [Mutable.sequence, sequenceValue, List.append_nil] using
          (show Mutable.sequence (.ref unitAddress) heap = some (.ok (peerLoop.encoded (before.map cells)), heap) by
            simp only [List.append_nil] at found
            simp [Mutable.sequence, found])
      · simp [peerLoop.encoded]
      · exact .nil
  | cons index rest ih =>
      let cell := cells index
      let ready := setLocal env "i" (number (Encoding.index cell).val)
      let afterHeap := work root heap keys cell (unitKeys cells)
      let afterKeys := Function.update keys cell (replacement keys cell (unitKeys cells))
      have peersLookup : ready "peers" = some (.ref root) := by simpa [ready, setLocal] using getPeers
      have setLookup : ready "s" = some (.ref setAddress) := by simpa [ready, setLocal] using getSet
      have ran := execute root keys cell setAddress (unitKeys cells) ready heap globals known setFound peersLookup setLookup
        (by simp [ready, setLocal])
      have updatedView : View root afterHeap afterKeys := view root heap keys cell (unitKeys cells) known
      have updatedFrame : Frame root heap afterHeap := update_frame root heap keys cell (unitKeys cells)
      have updatedRelated : ∀ current, Peers.Related current (afterKeys current) :=
        related keys cells (unitKeys cells) index allRelated (unit_within cells) unitRelated
      have unitKept : afterHeap[unitAddress]? = some (.list (peerLoop.encoded (((before ++ [index]) ++ rest).map cells))) := by
        simpa only [List.append_assoc, List.singleton_append] using updatedFrame.list known unitAddress _ found
      have setKept : afterHeap[setAddress]? = some (.set (unitKeys cells)) :=
        (updatedFrame.outside setAddress (List.getElem?_eq_some_iff.mp setFound).1 separate).trans setFound
      obtain ⟨finalHeap, finalEnv, finalKeys, continued, finalView, finalRelated, finalFrame, names⟩ :=
        ih (before ++ [index]) afterKeys afterHeap ready updatedView updatedRelated unitKept setKept peersLookup setLookup
      refine ⟨finalHeap, finalEnv, finalKeys, ?_, finalView, finalRelated, updatedFrame.trans finalFrame, ?_⟩
      · apply BlockContinues.sequence_step (items := peerLoop.encoded ((before ++ index :: rest).map cells))
          (value := number (Encoding.index cell).val)
        · simp [Mutable.sequence, sequenceValue, found]
        · simp [peerLoop.encoded, cell]
        · simpa only [Calls.loopContinuation, show Calls.loopControl [statement] = false from rfl,
            Bool.false_eq_true, if_false, List.length_append, List.length_singleton] using ran.append continued
      · intro name different
        rw [names name different]
        simp [ready, setLocal, different]

/-- Constructing s and processing all nine unit cells preserves all non-peer objects. -/
theorem unit (root : Nat) (entry : UnitEntry) (keys : Keys) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (known : View root heap keys) (allRelated : ∀ cell, Peers.Related cell (keys cell))
    (found : heap[entry.address]? = some (.list (peerLoop.encoded ((List.finRange 9).map entry.cells))))
    (getPeers : env "peers" = some (.ref root)) (getUnit : env "u" = some (.ref entry.address)) :
    ∃ finalHeap finalEnv finalKeys, BlockContinues program env unitBody heap finalEnv finalHeap globals ∧
      View root finalHeap finalKeys ∧ (∀ cell, Peers.Related cell (finalKeys cell)) ∧ Frame root heap finalHeap ∧
      (∀ name, name ∉ ["s", "i"] → finalEnv name = env name) := by
  let ready := setLocal env "s" (.ref heap.length)
  let after := heap ++ [.set (unitKeys entry.cells)]
  have built := unit_set entry.cells entry.address env heap globals found getUnit
  have unitFound : after[entry.address]? = some (.list (peerLoop.encoded ((List.finRange 9).map entry.cells))) :=
    HeapCollection.prefix_found heap _ _ _ found
  have unitLookup : ready "u" = some (.ref entry.address) := by simpa [ready, setLocal] using getUnit
  obtain ⟨finalHeap, finalEnv, finalKeys, scanned, finalView, finalRelated, finalFrame, names⟩ :=
    unit_loop root entry.address heap.length entry.cells entry.related [] (List.finRange 9) keys after ready globals
      (known.append _) allRelated unitFound (by simp [after]) known.fresh
      (by simpa [ready, setLocal] using getPeers) (by simp [ready, setLocal])
  have traversed : BlockContinues program ready [.forIn (.name "i") (.name "u") [statement] []]
      after finalEnv finalHeap globals := by
    apply BlockContinues.for_sequence (by intros; simp) (Evaluates.lookup unitLookup)
      (items := peerLoop.encoded ((List.finRange 9).map entry.cells))
    · simp [Mutable.sequence, unitFound]
    · exact scanned
  refine ⟨finalHeap, finalEnv, finalKeys, ?_, finalView, finalRelated, (Frame.append root heap _).trans finalFrame, ?_⟩
  · rw [unit_source]
    exact (BlockContinues.assign built).append traversed
  · intro name outside
    simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at outside
    rw [names name outside.2]
    simp [ready, setLocal, outside.1]

end
end solvers.«96».build_peers_units.peerBuild
