import solvers.«79.initialize.source»

/-! Graph storage is allocated beyond the caller's objects, which graph updates preserve. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

structure CallerPreserved (caller : Mutable.Heap) (roots : Roots)
    (state : Storage) (heap : Mutable.Heap) : Prop where
  unchanged : caller <+: heap
  nodesPrivate : caller.length ≤ roots.nodes
  adjacencyPrivate : caller.length ≤ roots.adjacency
  degreesPrivate : caller.length ≤ roots.degrees
  neighborsPrivate : ∀ vertex ∈ state.vertices, caller.length ≤ state.address vertex

theorem prefix_append (caller heap extra : Mutable.Heap) (saved : caller <+: heap) :
    caller <+: heap ++ extra := by
  obtain ⟨suffix, rfl⟩ := saved
  exact ⟨suffix ++ extra, (List.append_assoc caller suffix extra).symm⟩

theorem prefix_set (caller heap : Mutable.Heap) (saved : caller <+: heap)
    (index : Nat) (value : Mutable.Object) (outside : caller.length ≤ index) :
    caller <+: heap.set index value := by
  obtain ⟨suffix, rfl⟩ := saved
  exact ⟨suffix.set (index - caller.length) value,
    (List.set_append_right index value outside).symm⟩

theorem CallerPreserved.read (caller : Mutable.Heap) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (saved : CallerPreserved caller roots state heap)
    (index : Nat) (value : Mutable.Object) (found : caller[index]? = some value) :
    heap[index]? = some value := by
  obtain ⟨suffix, equal⟩ := saved.unchanged
  rw [← equal, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

theorem CallerPreserved.append (caller : Mutable.Heap) (roots : Roots) (state : Storage)
    (heap extra : Mutable.Heap) (saved : CallerPreserved caller roots state heap) :
    CallerPreserved caller roots state (heap ++ extra) :=
  { saved with unchanged := prefix_append caller heap extra saved.unchanged }

theorem initial_caller (caller : Mutable.Heap) :
    CallerPreserved caller (initialRoots caller) emptyStorage (initialHeap caller) := by
  constructor
  · exact ⟨_, rfl⟩
  · exact Nat.le_refl _
  · simp [initialRoots]
  · simp [initialRoots]
  · simp [emptyStorage]

theorem CallerPreserved.withNode (caller : Mutable.Heap) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (saved : CallerPreserved caller roots state heap) (vertex : String) :
    CallerPreserved caller roots (state.withNode vertex heap.length)
      (nodeHeap heap roots.nodes roots.adjacency roots.degrees
        state.vertices state.adjacency state.degrees vertex) := by
  by_cases present : vertex ∈ state.vertices
  · simpa only [Storage.withNode, nodeHeap, if_pos present] using saved
  · simp only [Storage.withNode, nodeHeap, if_neg present]
    constructor
    · apply prefix_set _ _ _ _ _ saved.degreesPrivate
      apply prefix_set _ _ _ _ _ saved.adjacencyPrivate
      apply prefix_append
      exact prefix_set _ _ saved.unchanged _ _ saved.nodesPrivate
    · exact saved.nodesPrivate
    · exact saved.adjacencyPrivate
    · exact saved.degreesPrivate
    · intro other member
      by_cases same : other = vertex
      · simpa only [if_pos same] using saved.unchanged.length_le
      · have old : other ∈ state.vertices := by simpa [same] using member
        simpa only [if_neg same] using saved.neighborsPrivate other old

theorem CallerPreserved.withEdge (caller : Mutable.Heap) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (saved : CallerPreserved caller roots state heap)
    (source target : String) (sourceMember : source ∈ state.vertices) :
    CallerPreserved caller roots (state.withEdge source target)
      (edgeHeap heap (state.address source) roots.degrees (state.neighbors source) state.degrees
        target (ProjectEulerStatements.TopologicalOrder.indegree target state.edges)) := by
  constructor
  · by_cases present : target ∈ state.neighbors source
    · simpa only [edgeHeap, if_pos present] using saved.unchanged
    · simp only [edgeHeap, if_neg present]
      apply prefix_set _ _ _ _ _ saved.degreesPrivate
      exact prefix_set _ _ saved.unchanged _ _ (saved.neighborsPrivate source sourceMember)
  · exact saved.nodesPrivate
  · exact saved.adjacencyPrivate
  · exact saved.degreesPrivate
  · exact saved.neighborsPrivate

end solvers.«79».graph
