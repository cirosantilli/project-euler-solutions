import solvers.«79.nodes.source»
import solvers.«79.edges.source»

/-! Shared graph storage connects the node set, adjacency references, and indegree table.
Successor sets have distinct addresses, so updating one vertex cannot change another's edges. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

structure Roots where
  nodes : Nat
  adjacency : Nat
  degrees : Nat

structure Storage where
  vertices : List String
  edges : List (String × String)
  adjacency : List (Constant × Mutable.Value)
  degrees : List (Constant × Mutable.Value)
  address : String → Nat
  neighbors : String → List String

structure Represents (roots : Roots) (state : Storage) (heap : Mutable.Heap) : Prop where
  nodesFound : heap[roots.nodes]? = some (.set (stringSet state.vertices))
  adjacencyFound : heap[roots.adjacency]? = some (.dict state.adjacency)
  degreesFound : heap[roots.degrees]? = some (.dict state.degrees)
  tablesDistinct : roots.adjacency ≠ roots.degrees
  verticesUnique : state.vertices.Nodup
  edgesUnique : state.edges.Nodup
  endpoints : ∀ edge ∈ state.edges, edge.1 ∈ state.vertices ∧ edge.2 ∈ state.vertices
  neighborsLookup : ∀ vertex ∈ state.vertices,
    ScalarDict.lookup state.adjacency (.string vertex) = some (.ref (state.address vertex))
  neighborsFound : ∀ vertex ∈ state.vertices,
    heap[state.address vertex]? = some (.set (stringSet (state.neighbors vertex)))
  contents : ∀ vertex ∈ state.vertices,
    (state.neighbors vertex).Perm (successors vertex state.edges)
  counts : ∀ vertex ∈ state.vertices,
    ScalarDict.lookup state.degrees (.string vertex) = some (number (indegree vertex state.edges))
  distinctSets : ∀ first ∈ state.vertices, ∀ second ∈ state.vertices,
    state.address first = state.address second → first = second
  notNodes : ∀ vertex ∈ state.vertices, state.address vertex ≠ roots.nodes

theorem Represents.neighbor_not_adjacency (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (vertex : String)
    (member : vertex ∈ state.vertices) :
    state.address vertex ≠ roots.adjacency := by
  intro same
  have found := represented.neighborsFound vertex member
  rw [same, represented.adjacencyFound] at found
  cases found

theorem Represents.neighbor_not_degrees (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (vertex : String)
    (member : vertex ∈ state.vertices) :
    state.address vertex ≠ roots.degrees := by
  intro same
  have found := represented.neighborsFound vertex member
  rw [same, represented.degreesFound] at found
  cases found

theorem Represents.nodes_not_degrees (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) : roots.nodes ≠ roots.degrees := by
  intro same
  have found := represented.nodesFound
  rw [same, represented.degreesFound] at found
  cases found

def Storage.withEdge (state : Storage) (source target : String) : Storage :=
  { state with
    edges := insertEdge source target state.edges
    degrees := insertDegree source target state.edges state.degrees
    neighbors := fun vertex => if vertex = source then
      insertNeighbor target (state.neighbors source) else state.neighbors vertex }

theorem Storage.withEdge_present (state : Storage) (source target : String)
    (present : (source, target) ∈ state.edges)
    (contents : (state.neighbors source).Perm (successors source state.edges)) :
    state.withEdge source target = state := by
  have member : target ∈ state.neighbors source :=
    contents.mem_iff.mpr ((mem_successors source target state.edges).mpr present)
  have sameNeighbors :
      (fun vertex => if vertex = source then
        insertNeighbor target (state.neighbors source) else state.neighbors vertex) =
      state.neighbors := by
    funext vertex
    by_cases same : vertex = source <;> simp [same, insertNeighbor, member]
  simp only [Storage.withEdge, insertEdge, insertDegree, if_pos present, sameNeighbors]

/-- Adding an existing or new edge retains the complete shared graph representation. -/
theorem Represents.withEdge (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (source target : String)
    (sourceMember : source ∈ state.vertices) (targetMember : target ∈ state.vertices) :
    Represents roots (state.withEdge source target)
      (edgeHeap heap (state.address source) roots.degrees (state.neighbors source) state.degrees
        target (indegree target state.edges)) := by
  have contents := represented.contents source sourceMember
  have member : target ∈ state.neighbors source ↔ (source, target) ∈ state.edges :=
    contents.mem_iff.trans (mem_successors source target state.edges)
  by_cases present : target ∈ state.neighbors source
  · rw [Storage.withEdge_present state source target (member.mp present) contents]
    simpa only [edgeHeap, if_pos present] using represented
  · have absent : (source, target) ∉ state.edges := by simpa only [← member] using present
    have neighborInside :=
      (List.getElem?_eq_some_iff.mp (represented.neighborsFound source sourceMember)).1
    have degreeInside := (List.getElem?_eq_some_iff.mp represented.degreesFound).1
    have notAdjacency := represented.neighbor_not_adjacency roots state heap source sourceMember
    have notDegrees := represented.neighbor_not_degrees roots state heap source sourceMember
    have notNodes := represented.notNodes source sourceMember
    have nodesNotDegrees := represented.nodes_not_degrees roots state heap
    let after := (heap.set (state.address source)
      (.set (stringSet (state.neighbors source ++ [target])))).set roots.degrees
      (.dict (ScalarDict.insert state.degrees (.string target)
        (number ((indegree target state.edges : Int) + 1))))
    simp only [edgeHeap, if_neg present]
    change Represents roots (state.withEdge source target) after
    constructor
    · simpa [after, Storage.withEdge, Ne.symm nodesNotDegrees, notNodes]
        using represented.nodesFound
    · simpa [after, Storage.withEdge, Ne.symm represented.tablesDistinct, notAdjacency]
        using represented.adjacencyFound
    · simp [after, Storage.withEdge, insertDegree, absent, degreeInside]
    · exact represented.tablesDistinct
    · exact represented.verticesUnique
    · simpa [Storage.withEdge, insertEdge, absent] using
        List.nodup_cons.mpr ⟨absent, represented.edgesUnique⟩
    · intro edge edgeMember
      change edge ∈ insertEdge source target state.edges at edgeMember
      rw [insertEdge, if_neg absent] at edgeMember
      rcases List.mem_cons.mp edgeMember with same | old
      · subst edge
        exact ⟨sourceMember, targetMember⟩
      · exact represented.endpoints edge old
    · exact represented.neighborsLookup
    · intro vertex vertexMember
      by_cases same : vertex = source
      · subst vertex
        simp [after, Storage.withEdge, insertNeighbor, present,
          Ne.symm notDegrees, neighborInside]
      · have separate : state.address source ≠ state.address vertex := by
          intro equal
          exact same (represented.distinctSets vertex vertexMember source sourceMember equal.symm)
        have vertexNotDegrees :=
          represented.neighbor_not_degrees roots state heap vertex vertexMember
        simpa [after, Storage.withEdge, same, separate, Ne.symm vertexNotDegrees]
          using represented.neighborsFound vertex vertexMember
    · intro vertex vertexMember
      by_cases same : vertex = source
      · subst vertex
        simpa only [Storage.withEdge, if_pos rfl] using
          insertNeighbor_contents source target state.edges (state.neighbors source) contents
      · have unchanged : successors vertex (insertEdge source target state.edges) =
            successors vertex state.edges := by
          simp [insertEdge, absent, successors, Ne.symm same]
        simpa only [Storage.withEdge, if_neg same, unchanged] using
          represented.contents vertex vertexMember
    · intro vertex vertexMember
      exact insertDegree_counts source target state.edges state.degrees vertex
        (represented.counts vertex vertexMember)
    · exact represented.distinctSets
    · exact represented.notNodes

theorem addEdge_preserves (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (source target : String)
    (sourceMember : source ∈ state.vertices) (targetMember : target ∈ state.vertices) :
    ∃ after, ReturnsValue program "__formalsw_local_build_graph_add_edge"
        [text source, text target, .ref roots.adjacency, .ref roots.degrees]
        heap (.scalar .none) after ∧ Represents roots (state.withEdge source target) after := by
  refine ⟨_, ?_, represented.withEdge roots state heap source target sourceMember targetMember⟩
  exact addEdge_result heap source target roots.adjacency roots.degrees (state.address source)
    state.adjacency state.degrees (state.neighbors source) (indegree target state.edges)
    represented.adjacencyFound (represented.neighborsFound source sourceMember)
    represented.degreesFound (represented.neighborsLookup source sourceMember)
    (represented.counts target targetMember)

end solvers.«79».graph
