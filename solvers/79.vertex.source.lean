import solvers.«79.state.source»

/-! Fresh vertex allocation preserves the shared graph representation. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

theorem Represents.nodes_not_adjacency (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) : roots.nodes ≠ roots.adjacency := by
  intro same
  have found := represented.nodesFound
  rw [same, represented.adjacencyFound] at found
  cases found

def Storage.withNode (state : Storage) (vertex : String) (fresh : Nat) : Storage :=
  if vertex ∈ state.vertices then state else
    { state with
      vertices := state.vertices ++ [vertex]
      adjacency := ScalarDict.insert state.adjacency (.string vertex) (.ref fresh)
      degrees := ScalarDict.insert state.degrees (.string vertex) (number 0)
      address := fun other => if other = vertex then fresh else state.address other
      neighbors := fun other => if other = vertex then [] else state.neighbors other }

theorem Represents.withNode (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (vertex : String) :
    Represents roots (state.withNode vertex heap.length)
      (nodeHeap heap roots.nodes roots.adjacency roots.degrees
        state.vertices state.adjacency state.degrees vertex) := by
  by_cases present : vertex ∈ state.vertices
  · simpa only [Storage.withNode, nodeHeap, if_pos present] using represented
  · have nodesInside := (List.getElem?_eq_some_iff.mp represented.nodesFound).1
    have adjacencyInside := (List.getElem?_eq_some_iff.mp represented.adjacencyFound).1
    have degreesInside := (List.getElem?_eq_some_iff.mp represented.degreesFound).1
    have nodesAdjacency := represented.nodes_not_adjacency roots state heap
    have nodesDegrees := represented.nodes_not_degrees roots state heap
    have addressesInside : ∀ other ∈ state.vertices, state.address other < heap.length := by
      intro other member
      exact (List.getElem?_eq_some_iff.mp (represented.neighborsFound other member)).1
    have noSuccessors : successors vertex state.edges = [] := by
      apply List.eq_nil_iff_forall_not_mem.mpr
      intro other member
      exact present (represented.endpoints (vertex, other)
        ((mem_successors vertex other state.edges).mp member)).1
    have zeroDegree : indegree vertex state.edges = 0 := by
      apply (indegree_zero_iff vertex state.edges).mpr
      intro edge member same
      exact present (same ▸ (represented.endpoints edge member).2)
    let added := heap.set roots.nodes (.set (stringSet (state.vertices ++ [vertex])))
    let allocated := added ++ [Mutable.Object.set []]
    let linked := allocated.set roots.adjacency
      (.dict (ScalarDict.insert state.adjacency (.string vertex) (.ref heap.length)))
    let after := linked.set roots.degrees
      (.dict (ScalarDict.insert state.degrees (.string vertex) (number 0)))
    have allocatedOld (index : Nat) (inside : index < heap.length) :
        allocated[index]? = added[index]? := by
      exact List.getElem?_append_left (by simpa [added] using inside)
    have allocatedFresh : allocated[heap.length]? = some (.set []) := by
      simp [allocated, added]
    have allocatedLength : allocated.length = heap.length + 1 := by
      simp [allocated, added]
    have afterOld (index : Nat) (inside : index < heap.length)
        (notNodes : roots.nodes ≠ index) (notAdjacency : roots.adjacency ≠ index)
        (notDegrees : roots.degrees ≠ index) : after[index]? = heap[index]? := by
      simp only [after, linked, List.getElem?_set, notDegrees, notAdjacency,
        if_false, allocatedOld index inside]
      simp [added, notNodes]
    simp only [Storage.withNode, nodeHeap, if_neg present]
    change Represents roots _ after
    constructor
    · simp [after, linked, Ne.symm nodesDegrees, Ne.symm nodesAdjacency,
        allocatedOld roots.nodes nodesInside, added, nodesInside]
    · simp [after, linked, Ne.symm represented.tablesDistinct, allocatedLength,
        show roots.adjacency < heap.length + 1 by omega]
    · simp [after, linked, allocatedLength, show roots.degrees < heap.length + 1 by omega]
    · exact represented.tablesDistinct
    · simpa using represented.verticesUnique.concat present
    · exact represented.edgesUnique
    · intro edge member
      obtain ⟨source, target⟩ := represented.endpoints edge member
      exact ⟨List.mem_append_left _ source, List.mem_append_left _ target⟩
    · intro other member
      by_cases same : other = vertex
      · subst other
        simp [ScalarDict.lookup_insert_self]
      · have old : other ∈ state.vertices := by simpa [same] using member
        simpa only [if_neg same,
          lookup_insert_other_string state.adjacency vertex other _ (Ne.symm same)]
          using represented.neighborsLookup other old
    · intro other member
      by_cases same : other = vertex
      · subst other
        simp [after, linked, ne_of_lt adjacencyInside, ne_of_lt degreesInside,
          allocatedFresh, stringSet]
      · have old : other ∈ state.vertices := by simpa [same] using member
        have unchanged := afterOld (state.address other) (addressesInside other old)
          (Ne.symm (represented.notNodes other old))
          (Ne.symm (represented.neighbor_not_adjacency roots state heap other old))
          (Ne.symm (represented.neighbor_not_degrees roots state heap other old))
        simpa only [if_neg same, unchanged] using represented.neighborsFound other old
    · intro other member
      by_cases same : other = vertex
      · subst other
        simp [noSuccessors]
      · have old : other ∈ state.vertices := by simpa [same] using member
        simpa only [if_neg same] using represented.contents other old
    · intro other member
      by_cases same : other = vertex
      · subst other
        simp [ScalarDict.lookup_insert_self, zeroDegree]
      · have old : other ∈ state.vertices := by simpa [same] using member
        simpa only [lookup_insert_other_string state.degrees vertex other _ (Ne.symm same)]
          using represented.counts other old
    · intro first firstMember second secondMember equal
      by_cases firstNew : first = vertex <;> by_cases secondNew : second = vertex
      · exact firstNew.trans secondNew.symm
      · have old : second ∈ state.vertices := by simpa [secondNew] using secondMember
        have inside := addressesInside second old
        simp only [if_pos firstNew, if_neg secondNew] at equal
        omega
      · have old : first ∈ state.vertices := by simpa [firstNew] using firstMember
        have inside := addressesInside first old
        simp only [if_neg firstNew, if_pos secondNew] at equal
        omega
      · have firstOld : first ∈ state.vertices := by simpa [firstNew] using firstMember
        have secondOld : second ∈ state.vertices := by simpa [secondNew] using secondMember
        exact represented.distinctSets first firstOld second secondOld
          (by simpa only [if_neg firstNew, if_neg secondNew] using equal)
    · intro other member
      by_cases same : other = vertex
      · simpa only [if_pos same] using (ne_of_lt nodesInside).symm
      · have old : other ∈ state.vertices := by simpa [same] using member
        simpa only [if_neg same] using represented.notNodes other old

theorem addNode_preserves (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (vertex : String) :
    ∃ after, ReturnsValue program "__formalsw_local_build_graph_add_node"
        [text vertex, .ref roots.nodes, .ref roots.adjacency, .ref roots.degrees]
        heap (.scalar .none) after ∧
      Represents roots (state.withNode vertex heap.length) after := by
  refine ⟨_, ?_, represented.withNode roots state heap vertex⟩
  exact addNode_result heap vertex roots.nodes roots.adjacency roots.degrees
    state.vertices state.adjacency state.degrees represented.nodesFound
    represented.adjacencyFound represented.degreesFound represented.tablesDistinct

end solvers.«79».graph
