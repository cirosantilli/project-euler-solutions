import solvers.«96.py.lean.d».PeerUnits

namespace solvers.«96».build_peers_units
open FormalSW.PyAst HeapCalls ProjectEulerSolutions ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

theorem source : definition.2 = definition.2.take 4 ++
    [.assign [.name "peers"] peerBuild.collection, peerBuild.outer, .return (.tuple [.name "peers", .name "units"])] := rfl

private theorem units_append (units : UnitTable) (heap extra : Mutable.Heap) (view : units.View heap) : units.View (heap ++ extra) :=
  ⟨HeapCollection.prefix_found heap extra _ _ view.root,
    fun unit member => HeapCollection.prefix_found heap extra _ _ (view.units unit member)⟩

/-- The complete constructor returns concrete peer and unit tables, preserving the caller's heap. -/
theorem result (heap : Mutable.Heap) (globals : Mutable.Globals) :
    ∃ (finalHeap : Mutable.Heap) (tupleAddress : Nat) (peers : PeerTable),
      ReturnsValue program "build_peers_units" [] heap (.ref tupleAddress) finalHeap globals ∧
      finalHeap[tupleAddress]? = some (.tuple [.ref peers.root, .ref (table heap.length).root]) ∧
      peers.View finalHeap ∧ (table heap.length).View finalHeap ∧
      heap.length ≤ finalHeap.length ∧ (∀ index < heap.length, finalHeap[index]? = heap[index]?) := by
  let env := functionEnv definition.1 definition.2 [] globals
  let unitHeap := work heap ((List.finRange 27).map Units.cells)
  obtain ⟨afterUnits, builtUnits, unitsLookup⟩ := units heap env globals
  let ready := setLocal afterUnits "peers" (.ref unitHeap.length)
  let initialHeap := peerBuild.emptyWork unitHeap 81
  have initialized : BlockContinues program afterUnits [.assign [.name "peers"] peerBuild.collection]
      unitHeap ready initialHeap globals := BlockContinues.assign (peerBuild.allocate afterUnits unitHeap globals)
  have initialView : peerBuild.View unitHeap.length initialHeap (fun _ => []) := peerBuild.empty_view unitHeap
  have initialUnits : (table heap.length).View initialHeap := units_append (table heap.length) unitHeap _ (table_view heap)
  obtain ⟨afterHeap, finalEnv, keys, ran, peerView, related, frame, getPeers, getUnits⟩ :=
    peerBuild.all_units unitHeap.length (table heap.length) (fun _ => []) initialHeap ready globals initialView
      (by simp [Peers.Related]) initialUnits (by simp [ready, setLocal])
      (by simpa [ready, setLocal] using unitsLookup)
  obtain ⟨peers, rootEq, represented, _⟩ := peerBuild.table_view unitHeap.length keys afterHeap peerView related
  let finalHeap := afterHeap ++ [.tuple [.ref peers.root, .ref (table heap.length).root]]
  have finalUnits : (table heap.length).View afterHeap := frame.units initialView (table heap.length) initialUnits
  have returned : BlockReturns program finalEnv [.return (.tuple [.name "peers", .name "units"])] afterHeap
      (.ref afterHeap.length) finalHeap globals := by
    apply BlockReturns.return_value
    apply Evaluates.tuple (.cons (Evaluates.lookup (by simpa only [rootEq] using getPeers))
      (.cons (Evaluates.lookup getUnits) .nil)) (by simp)
  have originalLength : heap.length ≤ unitHeap.length := by simp [unitHeap, work, HeapCollection.working]
  have initialLength : unitHeap.length ≤ initialHeap.length := by simp [initialHeap, peerBuild.emptyWork, HeapCollection.working]
  have prefixFrame (index : Nat) (inside : index < heap.length) : initialHeap[index]? = heap[index]? := by
    change (unitHeap ++ _)[index]? = _
    rw [List.getElem?_append_left (inside.trans_le originalLength)]
    exact List.getElem?_append_left inside
  refine ⟨finalHeap, afterHeap.length, peers, ?_, by simp [finalHeap], ?_,
    units_append (table heap.length) afterHeap _ finalUnits, ?_, ?_⟩
  · apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
    rw [source]
    exact builtUnits.returns (initialized.returns (ran.returns returned))
  · exact ⟨HeapCollection.prefix_found afterHeap _ _ _ represented.root,
      fun cell => HeapCollection.prefix_found afterHeap _ _ _ (represented.sets cell)⟩
  · exact (originalLength.trans (initialLength.trans frame.length)).trans (by simp [finalHeap])
  · intro index inside
    have within := inside.trans_le (originalLength.trans (initialLength.trans frame.length))
    rw [List.getElem?_append_left within]
    rw [frame.outside index (inside.trans_le (originalLength.trans initialLength))
      (by intro cell; unfold peerBuild.address; omega)]
    exact prefixFrame index inside

theorem «Returns» (heap : Mutable.Heap) (globals : Mutable.Globals) :
    (HeapCalls.call program "build_peers_units" [] heap globals).Returns := by
  obtain ⟨_, _, _, returned, _⟩ := result heap globals
  exact returned.returns

end
end solvers.«96».build_peers_units
