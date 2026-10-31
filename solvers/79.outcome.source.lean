import solvers.«79.solve.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p79.naive.source»

/-! Both graph representations impose the same ordering and reject the same first bad attempt. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».topo
open ProjectEulerStatements.TopologicalOrder
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

theorem insertVertex_mem (vertex value : String) (vertices : List String) :
    value ∈ insertVertex vertex vertices ↔ value ∈ vertices ∨ value = vertex := by
  by_cases present : vertex ∈ vertices <;> by_cases same : value = vertex <;>
    simp [insertVertex, present, same]

theorem insertVertex_unique (vertex : String) (vertices : List String)
    (unique : vertices.Nodup) : (insertVertex vertex vertices).Nodup := by
  by_cases present : vertex ∈ vertices
  · simpa [insertVertex, present] using unique
  · simp [insertVertex, present, List.nodup_append, unique]
    intro value member same
    exact present (same ▸ member)

theorem insertEdge_mem (source target : String) (edge : String × String)
    (edges : List (String × String)) :
    edge ∈ insertEdge source target edges ↔ edge ∈ edges ∨ edge = (source, target) := by
  by_cases present : (source, target) ∈ edges <;> by_cases same : edge = (source, target) <;>
    simp [insertEdge, present, same]

theorem reference_node_contents (attempts : List String) (value : String) :
    value ∈ p79.nodeValues attempts ↔ value ∈ p79.attemptCharacters attempts := by
  simpa [p79.nodeValues] using p79.dedupBy_keys id [] (p79.attemptCharacters attempts) value

/-- Vertex order and duplicate edges are irrelevant; the reference adds a redundant third edge. -/
structure RepresentsTriples (data : GraphData) (triples : List (Char × Char × Char)) : Prop where
  unique : data.vertices.Nodup
  vertices : ∀ value, value ∈ data.vertices ↔
    value ∈ p79.attemptCharacters (triples.map p79.tripleText)
  edges : ∀ edge, edge ∈ data.edges ↔ edge ∈ adjacentEdges (triples.map p79.tripleSymbols)

theorem RepresentsTriples.empty : RepresentsTriples emptyStorage.graphData [] := by
  constructor <;> simp [Storage.graphData, emptyStorage, p79.attemptCharacters, adjacentEdges]

theorem RepresentsTriples.extend (data : GraphData) (triples : List (Char × Char × Char))
    (represented : RepresentsTriples data triples) (triple : Char × Char × Char) :
    RepresentsTriples (data.extend (p79.tripleText triple)) (triples ++ [triple]) := by
  rcases triple with ⟨a, b, c⟩
  constructor
  · simpa [GraphData.extend, p79.tripleText] using
      insertVertex_unique (character c) _
        (insertVertex_unique (character b) _
          (insertVertex_unique (character a) _ represented.unique))
  · intro value
    simp only [GraphData.extend, p79.tripleText, String.toList_ofList, insertVertex_mem,
      represented.vertices]
    simp [p79.attemptCharacters, p79.tripleText, List.flatMap_append, character, or_assoc]
  · intro edge
    simp only [GraphData.extend, p79.tripleText, String.toList_ofList, insertEdge_mem,
      represented.edges]
    simp [adjacentEdges, p79.tripleSymbols, List.flatMap_append, character,
      p79.character, or_assoc]

theorem RepresentsTriples.order (data : GraphData) (triples : List (Char × Char × Char))
    (represented : RepresentsTriples data triples) :
    order data.vertices data.edges =
      order (p79.nodeValues (triples.map p79.tripleText))
        (p79.attemptEdges (triples.map p79.tripleText)) := by
  rw [p79.reference_order_adjacent]
  have permutation : data.vertices.Perm (p79.nodeValues (triples.map p79.tripleText)) :=
    (List.perm_ext_iff_of_nodup represented.unique (p79.node_values_unique _)).mpr
      (fun value => (represented.vertices value).trans (reference_node_contents _ value).symm)
  exact (order_nodes_perm _ _ _ permutation).trans (order_edges_congr _ _ _ represented.edges)

def processedOutcome (data : GraphData) (attempts : List String) :
    Except ExceptionObject Mutable.Value :=
  match data.process attempts with
  | .error error => .error error
  | .ok next => resultOutcome (order next.vertices next.edges)

theorem processed_outcome (data : GraphData) (triples : List (Char × Char × Char))
    (represented : RepresentsTriples data triples) (attempts : List String) :
    processedOutcome data attempts =
      p79.validatedOutcome (triples.map p79.tripleText) attempts := by
  induction attempts generalizing data triples with
  | nil =>
      simp only [processedOutcome, GraphData.process, p79.validatedOutcome, p79.checkedOutcome]
      rw [represented.order]
      cases order (p79.nodeValues (triples.map p79.tripleText))
        (p79.attemptEdges (triples.map p79.tripleText)) <;> rfl
  | cons s rest ih =>
      by_cases valid : ValidAttempt s
      · obtain ⟨a, b, c, chars⟩ := List.length_eq_three.mp
          (show s.toList.length = 3 by simpa using valid.1)
        have same : s = p79.tripleText (a, b, c) := by
          apply String.toList_injective
          simpa [p79.tripleText] using chars
        have extended := represented.extend data triples (a, b, c)
        rw [← same] at extended
        have continued := ih (data.extend s) (triples ++ [(a, b, c)]) extended
        simp only [List.map_append, List.map_cons, List.map_nil, ← same] at continued
        simpa only [processedOutcome, GraphData.process, if_pos valid,
          p79.validatedOutcome, show s.length = 3 ∧ Strings.isDigit s = true from valid,
          if_true] using continued
      · have invalid : ¬ (s.length = 3 ∧ Strings.isDigit s = true) := valid
        simp only [processedOutcome, GraphData.process, if_neg valid,
          p79.validatedOutcome, if_neg invalid]
        rfl

theorem solve_outcome_eq_reference (attempts : List String) :
    solveOutcome attempts = p79.validatedOutcome [] attempts :=
  processed_outcome emptyStorage.graphData [] RepresentsTriples.empty attempts

end solvers.«79».source
