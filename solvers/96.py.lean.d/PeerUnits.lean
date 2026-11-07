import solvers.«96.py.lean.d».PeerUnit
import solvers.«96.py.lean.d».CandidatePeerTable

namespace solvers.«96».build_peers_units.peerBuild
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

abbrev outer : Stmt := .forIn (.name "u") (.name "units") unitBody []
theorem outer_source : definition.2[5]? = some outer := rfl

/-- The outer source loop visits every represented unit, retaining its lists and table root. -/
theorem units_loop (root : Nat) (units : UnitTable) (before remaining : List UnitEntry)
    (suffix : units.units = before ++ remaining) (keys : Keys) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (known : View root heap keys) (allRelated : ∀ cell, Peers.Related cell (keys cell))
    (unitView : units.View heap) (getPeers : env "peers" = some (.ref root)) :
    ∃ finalHeap finalEnv finalKeys,
      BlockContinues program env [.sequenceStepLoop "u" (.ref units.root) before.length unitBody []]
        heap finalEnv finalHeap globals ∧
      View root finalHeap finalKeys ∧ (∀ cell, Peers.Related cell (finalKeys cell)) ∧ Frame root heap finalHeap ∧
      (∀ name, name ∉ ["u", "s", "i"] → finalEnv name = env name) := by
  induction remaining generalizing before keys heap env with
  | nil =>
      refine ⟨heap, env, keys, ?_, known, allRelated, .refl root heap, fun _ _ => rfl⟩
      have found := unitView.root
      simp only [suffix, List.append_nil] at found
      apply BlockContinues.sequence_done (items := UnitTable.references before)
      · simp [Mutable.sequence, sequenceValue, found]
      · simp [UnitTable.references]
      · exact .nil
  | cons entry rest ih =>
      let ready := setLocal env "u" (.ref entry.address)
      have member : entry ∈ units.units := by rw [suffix]; simp
      obtain ⟨middleHeap, middleEnv, middleKeys, ran, middleView, middleRelated, middleFrame, names⟩ :=
        unit root entry keys heap ready globals known allRelated (unitView.units entry member)
          (by simpa [ready, setLocal] using getPeers) (by simp [ready, setLocal])
      obtain ⟨finalHeap, finalEnv, finalKeys, continued, finalView, finalRelated, finalFrame, finalNames⟩ :=
        ih (before ++ [entry]) (by simpa only [List.append_assoc, List.singleton_append] using suffix)
          middleKeys middleHeap middleEnv middleView middleRelated (middleFrame.units known units unitView)
          ((names "peers" (by decide)).trans (by simpa [ready, setLocal] using getPeers))
      refine ⟨finalHeap, finalEnv, finalKeys, ?_, finalView, finalRelated, middleFrame.trans finalFrame, ?_⟩
      · apply BlockContinues.sequence_step (items := UnitTable.references (before ++ entry :: rest)) (value := .ref entry.address)
        · simp [Mutable.sequence, sequenceValue, unitView.root, suffix]
        · simp [UnitTable.references]
        · simpa only [Calls.loopContinuation, show Calls.loopControl unitBody = false from rfl,
            Bool.false_eq_true, if_false, List.length_append, List.length_singleton] using ran.append continued
      · intro name outside
        rw [finalNames name outside, names name (by simpa using (show name ∉ ["s", "i"] from
          fun member => outside (List.mem_cons_of_mem _ member)))]
        have different : name ≠ "u" := fun same => outside (by simp [same])
        simp [ready, setLocal, different]

theorem all_units (root : Nat) (units : UnitTable) (keys : Keys) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) (known : View root heap keys) (allRelated : ∀ cell, Peers.Related cell (keys cell))
    (unitView : units.View heap) (getPeers : env "peers" = some (.ref root)) (getUnits : env "units" = some (.ref units.root)) :
    ∃ finalHeap finalEnv finalKeys, BlockContinues program env [outer] heap finalEnv finalHeap globals ∧
      View root finalHeap finalKeys ∧ (∀ cell, Peers.Related cell (finalKeys cell)) ∧ Frame root heap finalHeap ∧
      finalEnv "peers" = some (.ref root) ∧ finalEnv "units" = some (.ref units.root) := by
  obtain ⟨finalHeap, finalEnv, finalKeys, ran, finalView, finalRelated, frame, names⟩ :=
    units_loop root units [] units.units rfl keys heap env globals known allRelated unitView getPeers
  refine ⟨finalHeap, finalEnv, finalKeys, ?_, finalView, finalRelated, frame,
    (names "peers" (by decide)).trans getPeers, (names "units" (by decide)).trans getUnits⟩
  apply BlockContinues.for_sequence (by intros; simp) (Evaluates.lookup getUnits)
    (items := UnitTable.references units.units)
  · simp [Mutable.sequence, unitView.root]
  · exact ran

/-- The constructed sets give the cell-list representation needed for every possible set iteration order. -/
theorem table_view (root : Nat) (keys : Keys) (heap : Mutable.Heap) (known : View root heap keys)
    (allRelated : ∀ cell, Peers.Related cell (keys cell)) :
    ∃ peers : PeerTable, peers.root = root ∧ peers.View heap ∧ (∀ cell, peers.address cell = address root cell) := by
  choose cells encoded related using fun cell => Peers.decode cell (keys cell) (allRelated cell)
  let peers : PeerTable := ⟨root, address root, cells, related⟩
  have referencesEq : references root 81 = peers.references := by
    apply List.ext_getElem?
    intro index
    simp only [references, PeerTable.references, peers, List.getElem?_map, List.getElem?_ofFn]
    by_cases inside : index < 81
    · simp only [List.getElem?_range inside, dif_pos inside, Option.map_some, address, Encoding.index_cell]
    · simp only [List.getElem?_eq_none (show (List.range 81).length ≤ index by simp; omega), dif_neg inside, Option.map_none]
  refine ⟨peers, rfl, ⟨?_, ?_⟩, fun _ => rfl⟩
  · simpa only [referencesEq] using known.root
  · intro cell
    simpa only [encoded cell] using known.sets cell

end
end solvers.«96».build_peers_units.peerBuild
