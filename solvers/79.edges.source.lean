import solvers.«79.graph.source»
import ProjectEulerStatements.Lib.Indegree
import Mathlib.Data.String.Basic

/-! The source edge update maintains abstract successor contents and every vertex's indegree. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

theorem scalar_string_eq (stored : Constant) (key : String) :
    (scalarCompare .eq stored (.string key)).getD false = decide (stored = .string key) := by
  cases stored <;> simp [scalarCompare]

theorem lookup_insert_other_string (entries : List (Constant × Mutable.Value))
    (key other : String) (value : Mutable.Value) (different : key ≠ other) :
    ScalarDict.lookup (ScalarDict.insert entries (.string key) value) (.string other) =
      ScalarDict.lookup entries (.string other) := by
  induction entries with
  | nil => simp [ScalarDict.insert, ScalarDict.lookup, scalar_string_eq, different]
  | cons entry rest ih =>
      obtain ⟨stored, previous⟩ := entry
      by_cases same : stored = .string key
      · subst stored
        simp [ScalarDict.insert, ScalarDict.lookup, scalar_string_eq, different]
      · simp only [ScalarDict.insert, scalar_string_eq, same, decide_false, Bool.false_eq_true,
          if_false, ScalarDict.lookup]
        by_cases otherKey : stored = .string other <;>
          simp [otherKey, ih]

def insertEdge (source target : String) (edges : List (String × String)) : List (String × String) :=
  if (source, target) ∈ edges then edges else (source, target) :: edges

def insertNeighbor (target : String) (neighbors : List String) : List String :=
  if target ∈ neighbors then neighbors else neighbors ++ [target]

def insertDegree (source target : String) (edges : List (String × String))
    (degrees : List (Constant × Mutable.Value)) : List (Constant × Mutable.Value) :=
  if (source, target) ∈ edges then degrees else
    ScalarDict.insert degrees (.string target) (number ((indegree target edges : Int) + 1))

theorem insertNeighbor_contents (source target : String) (edges : List (String × String))
    (neighbors : List String) (contents : neighbors.Perm (successors source edges)) :
    (insertNeighbor target neighbors).Perm
      (successors source (insertEdge source target edges)) := by
  have member : target ∈ neighbors ↔ (source, target) ∈ edges :=
    contents.mem_iff.trans (mem_successors source target edges)
  by_cases present : target ∈ neighbors
  · simpa [insertNeighbor, insertEdge, present, member.mp present] using contents
  · have absent : (source, target) ∉ edges := by simpa only [← member] using present
    have moved : (neighbors ++ [target]).Perm (target :: neighbors) :=
      List.perm_append_comm
    have result := moved.trans (contents.cons target)
    simpa [insertNeighbor, insertEdge, present, absent, successors] using result

theorem insertDegree_counts (source target : String) (edges : List (String × String))
    (degrees : List (Constant × Mutable.Value)) (vertex : String)
    (counted : ScalarDict.lookup degrees (.string vertex) = some (number (indegree vertex edges))) :
    ScalarDict.lookup (insertDegree source target edges degrees) (.string vertex) =
      some (number (indegree vertex (insertEdge source target edges))) := by
  by_cases present : (source, target) ∈ edges
  · simpa [insertDegree, insertEdge, present] using counted
  · have updated := indegree_insert source target vertex edges
    by_cases same : target = vertex
    · subst vertex
      simp only [insertDegree, present, if_false, ScalarDict.lookup_insert_self]
      have newCount : indegree target ((source, target) :: edges) = indegree target edges + 1 := by
        simpa [present] using updated
      simp [insertEdge, present, newCount, number]
    · rw [insertDegree, if_neg present,
        lookup_insert_other_string degrees target vertex _ same]
      have unchanged : indegree vertex (insertEdge source target edges) =
          indegree vertex edges := by
        simpa [insertEdge, present, same] using updated
      rw [unchanged]
      exact counted

/-- The concrete helper returns the graph model's updated successor set and count table. -/
theorem addEdge_refines (heap : Mutable.Heap) (source target : String)
    (adjacencyAddress degreesAddress neighborsAddress : Nat)
    (adjacency degrees : List (Constant × Mutable.Value))
    (edges : List (String × String)) (neighbors vertices : List String)
    (adjacencyFound : heap[adjacencyAddress]? = some (.dict adjacency))
    (neighborsFound : heap[neighborsAddress]? = some (.set (stringSet neighbors)))
    (degreesFound : heap[degreesAddress]? = some (.dict degrees))
    (neighborsLookup : ScalarDict.lookup adjacency (.string source) = some (.ref neighborsAddress))
    (targetPresent : target ∈ vertices)
    (contents : neighbors.Perm (successors source edges))
    (counted : ∀ vertex ∈ vertices, ScalarDict.lookup degrees (.string vertex) =
      some (number (indegree vertex edges))) :
    ∃ after, ReturnsValue program "__formalsw_local_build_graph_add_edge"
        [text source, text target, .ref adjacencyAddress, .ref degreesAddress]
        heap (.scalar .none) after ∧
      after[neighborsAddress]? = some (.set (stringSet (insertNeighbor target neighbors))) ∧
      after[degreesAddress]? = some (.dict (insertDegree source target edges degrees)) ∧
      (insertNeighbor target neighbors).Perm (successors source (insertEdge source target edges)) ∧
      (∀ vertex ∈ vertices,
        ScalarDict.lookup (insertDegree source target edges degrees) (.string vertex) =
          some (number (indegree vertex (insertEdge source target edges)))) := by
  refine ⟨edgeHeap heap neighborsAddress degreesAddress neighbors degrees target
    (indegree target edges), ?_, ?_, ?_,
    insertNeighbor_contents source target edges neighbors contents,
    fun vertex member =>
      insertDegree_counts source target edges degrees vertex (counted vertex member)⟩
  · exact addEdge_result heap source target adjacencyAddress degreesAddress neighborsAddress
      adjacency degrees neighbors (indegree target edges) adjacencyFound neighborsFound degreesFound
      neighborsLookup (counted target targetPresent)
  · have distinct : neighborsAddress ≠ degreesAddress := by
      intro same
      subst degreesAddress
      rw [neighborsFound] at degreesFound
      cases degreesFound
    have inside := (List.getElem?_eq_some_iff.mp neighborsFound).1
    by_cases present : target ∈ neighbors
    · simpa only [edgeHeap, insertNeighbor, if_pos present] using neighborsFound
    · simp [edgeHeap, insertNeighbor, present, Ne.symm distinct, inside]
  · have member : target ∈ neighbors ↔ (source, target) ∈ edges :=
      contents.mem_iff.trans (mem_successors source target edges)
    have inside := (List.getElem?_eq_some_iff.mp degreesFound).1
    by_cases present : target ∈ neighbors
    · simp [edgeHeap, insertDegree, present, member.mp present, degreesFound]
    · have absent : (source, target) ∉ edges := by simpa only [← member] using present
      simp [edgeHeap, insertDegree, present, absent, inside]

end solvers.«79».graph
