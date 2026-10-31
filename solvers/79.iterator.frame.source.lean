import solvers.«79.valid.source»
import lang.py.FormalSW.TupleRelocation

/-! Graph updates preserve arbitrary disjoint caller cells, including iterator allocations
interleaved with the graph's own storage. No unchanged-prefix assumption is needed. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

structure PreservesCells (keep : Nat → Prop) (before after : Mutable.Heap) : Prop where
  length : before.length ≤ after.length
  read : ∀ address, keep address → address < before.length →
    after[address]? = before[address]?

theorem PreservesCells.refl (keep : Nat → Prop) (heap : Mutable.Heap) :
    PreservesCells keep heap heap := ⟨Nat.le_refl _, fun _ _ _ => rfl⟩

theorem PreservesCells.trans {keep : Nat → Prop} {first middle last : Mutable.Heap}
    (left : PreservesCells keep first middle)
    (right : PreservesCells keep middle last) : PreservesCells keep first last :=
  ⟨left.length.trans right.length, fun address selected inside =>
    (right.read address selected (Nat.lt_of_lt_of_le inside left.length)).trans
      (left.read address selected inside)⟩

theorem PreservesCells.append (keep : Nat → Prop) (heap extra : Mutable.Heap) :
    PreservesCells keep heap (heap ++ extra) := by
  constructor
  · simp
  · intro address _ inside
    exact List.getElem?_append_left inside

theorem PreservesCells.set (keep : Nat → Prop) (heap : Mutable.Heap)
    (address : Nat) (object : Mutable.Object) (separate : ¬ keep address) :
    PreservesCells keep heap (heap.set address object) := by
  constructor
  · simp
  · intro observed selected _
    have different : address ≠ observed := by
      intro same
      exact separate (same ▸ selected)
    simp [different]

/-- The solver's private writes preserve a relocated caller graph, including cycles. -/
theorem PreservesCells.layout {keep : Nat → Prop} {before after logical : Mutable.Heap}
    {rename : Nat → Nat} (retained : PreservesCells keep before after)
    (layout : Mutable.Relocation.HeapLayout logical before rename)
    (selected : ∀ index < logical.length, keep (rename index)) :
    Mutable.Relocation.HeapLayout logical after rename := by
  apply layout.preserves
  intro index inside
  have found : (logical.map Mutable.Relocation.graphNode)[index]? =
      some (Mutable.Relocation.graphNode logical[index]) := by
    simp [List.getElem?_eq_getElem inside]
  have lookup := layout.lookup index _ found
  have bound : rename index < before.length := by
    simpa only [List.length_map] using (List.getElem?_eq_some_iff.mp lookup).1
  exact retained.read (rename index) (selected index inside) bound

/-- Private graph updates also retain the metadata needed by later iterator methods. -/
theorem PreservesCells.runtimeLayout {keep : Nat → Prop} {before after logical : Mutable.Heap}
    {rename : Nat → Nat} (retained : PreservesCells keep before after)
    (layout : Mutable.Relocation.RuntimeLayout logical before rename)
    (selected : ∀ index < logical.length, keep (rename index)) :
    Mutable.Relocation.RuntimeLayout logical after rename := by
  apply layout.preserves
  intro index inside
  exact retained.read (rename index) (selected index inside) (layout.bound index inside)

/-- Protected objects may occur anywhere in memory, but cannot be graph-owned cells. -/
structure ProtectedCells (keep : Nat → Prop) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) : Prop where
  inside : ∀ address, keep address → address < heap.length
  nodes : ¬ keep roots.nodes
  adjacency : ¬ keep roots.adjacency
  degrees : ¬ keep roots.degrees
  neighbors : ∀ vertex ∈ state.vertices, ¬ keep (state.address vertex)

theorem ProtectedCells.append {keep : Nat → Prop} {roots : Roots} {state : Storage}
    {heap : Mutable.Heap} (saved : ProtectedCells keep roots state heap)
    (extra : Mutable.Heap) : ProtectedCells keep roots state (heap ++ extra) :=
  { saved with inside := fun address selected =>
      Nat.lt_of_lt_of_le (saved.inside address selected) (by simp) }

theorem ProtectedCells.node_preserves {keep : Nat → Prop} {roots : Roots}
    {state : Storage} {heap : Mutable.Heap}
    (saved : ProtectedCells keep roots state heap) (vertex : String) :
    PreservesCells keep heap
      (nodeHeap heap roots.nodes roots.adjacency roots.degrees
        state.vertices state.adjacency state.degrees vertex) := by
  by_cases present : vertex ∈ state.vertices
  · simpa [nodeHeap, present] using PreservesCells.refl keep heap
  · simp only [nodeHeap, if_neg present]
    exact (PreservesCells.set keep heap roots.nodes _ saved.nodes).trans
      ((PreservesCells.append keep _ _).trans
        ((PreservesCells.set keep _ roots.adjacency _ saved.adjacency).trans
          (PreservesCells.set keep _ roots.degrees _ saved.degrees)))

theorem ProtectedCells.withNode {keep : Nat → Prop} {roots : Roots} {state : Storage}
    {heap : Mutable.Heap} (saved : ProtectedCells keep roots state heap) (vertex : String) :
    ProtectedCells keep roots (state.withNode vertex heap.length)
      (nodeHeap heap roots.nodes roots.adjacency roots.degrees
        state.vertices state.adjacency state.degrees vertex) := by
  by_cases present : vertex ∈ state.vertices
  · simpa [Storage.withNode, nodeHeap, present] using saved
  · constructor
    · intro address selected
      exact Nat.lt_of_lt_of_le (saved.inside address selected) (saved.node_preserves vertex).length
    · exact saved.nodes
    · exact saved.adjacency
    · exact saved.degrees
    · intro other member selected
      by_cases same : other = vertex
      · subst other
        have fresh : keep heap.length := by simpa [Storage.withNode, present] using selected
        exact Nat.lt_irrefl _ (saved.inside _ fresh)
      · have old : other ∈ state.vertices := by
          simpa [Storage.withNode, present, same] using member
        exact saved.neighbors other old (by simpa [Storage.withNode, present, same] using selected)

theorem ProtectedCells.edge_preserves {keep : Nat → Prop} {roots : Roots}
    {state : Storage} {heap : Mutable.Heap}
    (saved : ProtectedCells keep roots state heap) (source target : String)
    (member : source ∈ state.vertices) :
    PreservesCells keep heap
      (edgeHeap heap (state.address source) roots.degrees (state.neighbors source) state.degrees
        target (ProjectEulerStatements.TopologicalOrder.indegree target state.edges)) := by
  by_cases present : target ∈ state.neighbors source
  · simpa [edgeHeap, present] using PreservesCells.refl keep heap
  · simp only [edgeHeap, if_neg present]
    exact (PreservesCells.set keep heap (state.address source) _
      (saved.neighbors source member)).trans
      (PreservesCells.set keep _ roots.degrees _ saved.degrees)

theorem ProtectedCells.withEdge {keep : Nat → Prop} {roots : Roots} {state : Storage}
    {heap : Mutable.Heap} (saved : ProtectedCells keep roots state heap)
    (source target : String) (member : source ∈ state.vertices) :
    ProtectedCells keep roots (state.withEdge source target)
      (edgeHeap heap (state.address source) roots.degrees (state.neighbors source) state.degrees
        target (ProjectEulerStatements.TopologicalOrder.indegree target state.edges)) := by
  refine { saved with inside := ?_ }
  intro address selected
  exact Nat.lt_of_lt_of_le (saved.inside address selected)
    (saved.edge_preserves source target member).length


/-- Private graph construction never changes the allocator's empty-tuple choice. -/
theorem Represents.node_preserves_emptyTuple (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap) (vertex : String) :
    Mutable.Relocation.emptyTupleAddress
      (nodeHeap heap roots.nodes roots.adjacency roots.degrees
        state.vertices state.adjacency state.degrees vertex) =
      Mutable.Relocation.emptyTupleAddress heap := by
  by_cases present : vertex ∈ state.vertices
  · simp [nodeHeap, present]
  · simp only [nodeHeap, if_neg present]
    have adjacencyInside := (List.getElem?_eq_some_iff.mp represented.adjacencyFound).1
    have degreesInside := (List.getElem?_eq_some_iff.mp represented.degreesFound).1
    rw [Mutable.Relocation.emptyTupleAddress_set_nonempty _ roots.degrees _
      (by
        simp only [List.getElem?_set, if_neg represented.tablesDistinct]
        rw [List.getElem?_append_left (by simpa using degreesInside)]
        simp [represented.nodes_not_degrees roots state heap,
          represented.degreesFound]) (by simp)]
    rw [Mutable.Relocation.emptyTupleAddress_set_nonempty _ roots.adjacency _
      (by
        rw [List.getElem?_append_left (by simpa using adjacencyInside)]
        simp [represented.nodes_not_adjacency roots state heap,
          represented.adjacencyFound]) (by simp)]
    rw [Mutable.Relocation.emptyTupleAddress_append_nonempty _ _ (by simp)]
    exact Mutable.Relocation.emptyTupleAddress_set _ _ _ _
      represented.nodesFound (by simp) (by simp)

theorem Represents.edge_preserves_emptyTuple (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap)
    (source target : String) (member : source ∈ state.vertices) :
    Mutable.Relocation.emptyTupleAddress
      (edgeHeap heap (state.address source) roots.degrees (state.neighbors source) state.degrees
        target (ProjectEulerStatements.TopologicalOrder.indegree target state.edges)) =
      Mutable.Relocation.emptyTupleAddress heap := by
  by_cases present : target ∈ state.neighbors source
  · simp [edgeHeap, present]
  · simp only [edgeHeap, if_neg present]
    rw [Mutable.Relocation.emptyTupleAddress_set_nonempty _ roots.degrees _
      (by simp [List.getElem?_set, represented.degreesFound, ite_eq_iff]) (by simp)]
    exact Mutable.Relocation.emptyTupleAddress_set _ _ _ _
      (represented.neighborsFound source member) (by simp) (by simp)

end solvers.«79».graph
