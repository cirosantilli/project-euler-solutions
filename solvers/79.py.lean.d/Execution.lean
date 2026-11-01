import «.formalsw».solvers.«79.py.ast»
import lang.py.FormalSW.DictionaryCalls
import ProjectEulerStatements.Lib.Indegree
import Mathlib.Data.String.Basic
import lang.py.FormalSW.DigitCalls
import lang.py.FormalSW.ReprCalls
import lang.py.FormalSW.SetComprehensionCalls
import Mathlib.Data.List.InsertIdx
import ProjectEulerStatements.Lib.HeapqHeapify
import lang.py.FormalSW.DictionaryConstructorCalls
import ProjectEulerStatements.Lib.HeapqSelection
import ProjectEulerStatements.Lib.HeapqPush
import data.«project-euler-statements».data.py.project_euler_statements.«p79.py»
import lang.py.FormalSW.TupleRelocation
import lang.py.FormalSW.IteratorRules
import lang.py.FormalSW.RuntimeEffects
import lang.py.FormalSW.IteratorExecutionIsolation
import lang.py.FormalSW.IteratorPrefix
import lang.py.FormalSW.IteratorAcquisitionIsolation

/-! Source-bound graph updates retain shared set references and count each edge once. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P79Ast.module
abbrev text (value : String) : Mutable.Value := .scalar (.string value)
abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
abbrev stringSet (values : List String) : List Constant := values.map Constant.string

theorem contains_strings (values : List String) (value : String) :
    ScalarSet.contains (stringSet values) (.string value) = decide (value ∈ values) := by
  apply Bool.eq_iff_iff.mpr
  simp [ScalarSet.contains, stringSet, List.any_eq_true, scalarCompare]

def addEdgeBody : List Stmt :=
  [.ifThen (.membership (.name "v") (.subscript (.name "adj") (.name "u")) true)
    [.expr (.call (.attribute (.subscript (.name "adj") (.name "u")) "add") [.name "v"]),
     .augAssign (.subscript (.name "indeg") (.name "v")) .add (.constant (.int 1))] []]

theorem addEdge_source :
    executableBody ((findFunctionIn program.body "__formalsw_local_build_graph_add_edge").getD
      ([], [])).2 = addEdgeBody := rfl

def edgeHeap (heap : Mutable.Heap) (neighborsAddress degreesAddress : Nat)
    (neighbors : List String) (degrees : List (Constant × Mutable.Value))
    (target : String) (count : Int) : Mutable.Heap :=
  if target ∈ neighbors then heap else
    (heap.set neighborsAddress (.set (stringSet (neighbors ++ [target])))).set degreesAddress
      (.dict (ScalarDict.insert degrees (.string target) (number (count + 1))))

theorem addEdge_block (env : HeapCalls.Env) (heap : Mutable.Heap)
    (source target : String) (adjacencyAddress degreesAddress neighborsAddress : Nat)
    (adjacency degrees : List (Constant × Mutable.Value)) (neighbors : List String) (count : Int)
    (getSource : env "u" = some (text source)) (getTarget : env "v" = some (text target))
    (getAdjacency : env "adj" = some (.ref adjacencyAddress))
    (getDegrees : env "indeg" = some (.ref degreesAddress))
    (adjacencyFound : heap[adjacencyAddress]? = some (.dict adjacency))
    (neighborsFound : heap[neighborsAddress]? = some (.set (stringSet neighbors)))
    (degreesFound : heap[degreesAddress]? = some (.dict degrees))
    (neighborsLookup : ScalarDict.lookup adjacency (.string source) = some (.ref neighborsAddress))
    (countLookup : ScalarDict.lookup degrees (.string target) = some (number count)) :
    BlockContinues program env addEdgeBody heap env
      (edgeHeap heap neighborsAddress degreesAddress neighbors degrees target count) := by
  have readNeighbors : Evaluates program env (.subscript (.name "adj") (.name "u"))
      heap (.ref neighborsAddress) heap := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getAdjacency) (Evaluates.lookup getSource)
    exact Mutable.getItem_dictionary heap adjacencyAddress adjacency adjacencyFound
      (.string source) neighborsLookup
  have tested : Evaluates program env
      (.membership (.name "v") (.subscript (.name "adj") (.name "u")) true) heap
      (.scalar (.bool (decide (target ∉ neighbors)))) heap := by
    apply Evaluates.membership (.cons (Evaluates.lookup getTarget) (.cons readNeighbors .nil))
    simp [HeapCalls.membership, neighborsFound, contains_strings]
  by_cases present : target ∈ neighbors
  · unfold edgeHeap
    rw [if_pos present]
    apply BlockContinues.branch (flag := false)
      (by simpa [present] using tested) rfl
    exact .nil
  · have distinct : neighborsAddress ≠ degreesAddress := by
      intro same
      subst degreesAddress
      rw [neighborsFound] at degreesFound
      cases degreesFound
    let added := heap.set neighborsAddress (.set (stringSet (neighbors ++ [target])))
    have add : BlockContinues program env
        [.expr (.call (.attribute (.subscript (.name "adj") (.name "u")) "add") [.name "v"])]
        heap env added := by
      apply BlockContinues.discard
      have operation := Evaluates.add readNeighbors neighborsFound (Evaluates.lookup getTarget)
        neighborsFound
      simpa [ScalarSet.insert, contains_strings, present, added, stringSet] using operation
    have degreesAfter : added[degreesAddress]? = some (.dict degrees) := by
      simp [added, distinct, degreesFound]
    have increment : BlockContinues program env
        [.augAssign (.subscript (.name "indeg") (.name "v")) .add (.constant (.int 1))]
        added env (added.set degreesAddress
          (.dict (ScalarDict.insert degrees (.string target) (number (count + 1))))) := by
      apply BlockContinues.augItem (updated := number (count + 1)) (afterOperation := added)
        (by intros; intro impossible; cases impossible)
        (Evaluates.lookup getDegrees) (Evaluates.lookup getTarget)
        (Mutable.getItem_dictionary added degreesAddress degrees degreesAfter
          (.string target) countLookup) Evaluates.constant
      · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
      · exact Mutable.setItem_dictionary added degreesAddress degrees degreesAfter
          (.string target) (number (count + 1))
    unfold edgeHeap
    rw [if_neg present]
    apply BlockContinues.branch (flag := true)
      (by simpa [present] using tested) rfl
    exact add.append increment

/-- The lifted source helper skips duplicate edges and increments exactly one destination count. -/
theorem addEdge_result (heap : Mutable.Heap)
    (source target : String) (adjacencyAddress degreesAddress neighborsAddress : Nat)
    (adjacency degrees : List (Constant × Mutable.Value)) (neighbors : List String) (count : Int)
    (adjacencyFound : heap[adjacencyAddress]? = some (.dict adjacency))
    (neighborsFound : heap[neighborsAddress]? = some (.set (stringSet neighbors)))
    (degreesFound : heap[degreesAddress]? = some (.dict degrees))
    (neighborsLookup : ScalarDict.lookup adjacency (.string source) = some (.ref neighborsAddress))
    (countLookup : ScalarDict.lookup degrees (.string target) = some (number count)) :
    ReturnsValue program "__formalsw_local_build_graph_add_edge"
      [text source, text target, .ref adjacencyAddress, .ref degreesAddress] heap (.scalar .none)
      (edgeHeap heap neighborsAddress degreesAddress neighbors degrees target count) := by
  apply BlockReturns.function rfl rfl
  change BlockReturns program _ (addEdgeBody ++ []) heap _ _
  apply (addEdge_block _ heap source target adjacencyAddress degreesAddress neighborsAddress
    adjacency degrees neighbors count rfl rfl rfl rfl adjacencyFound neighborsFound degreesFound
    neighborsLookup countLookup).returns
  exact fun _ _ => .next rfl .refl

end solvers.«79».graph

/-! Source-bound vertex insertion creates one fresh successor set only for a new vertex. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def addNodeBody : List Stmt :=
  [.ifThen (.membership (.name "v") (.name "nodes") true)
    [.expr (.call (.attribute (.name "nodes") "add") [.name "v"]),
     .assign [.subscript (.name "adj") (.name "v")] (.builtinCall .set []),
     .assign [.subscript (.name "indeg") (.name "v")] (.constant (.int 0))] []]

theorem addNode_source :
    executableBody ((findFunctionIn program.body "__formalsw_local_build_graph_add_node").getD
      ([], [])).2 = addNodeBody := rfl

def nodeHeap (heap : Mutable.Heap) (nodesAddress adjacencyAddress degreesAddress : Nat)
    (nodes : List String) (adjacency degrees : List (Constant × Mutable.Value))
    (vertex : String) : Mutable.Heap :=
  if vertex ∈ nodes then heap else
    (((heap.set nodesAddress (.set (stringSet (nodes ++ [vertex])))) ++ [Mutable.Object.set []]).set
      adjacencyAddress
      (.dict (ScalarDict.insert adjacency (.string vertex) (.ref heap.length)))).set
      degreesAddress (.dict (ScalarDict.insert degrees (.string vertex) (number 0)))

theorem addNode_block (env : HeapCalls.Env) (heap : Mutable.Heap)
    (vertex : String) (nodesAddress adjacencyAddress degreesAddress : Nat)
    (nodes : List String) (adjacency degrees : List (Constant × Mutable.Value))
    (getVertex : env "v" = some (text vertex))
    (getNodes : env "nodes" = some (.ref nodesAddress))
    (getAdjacency : env "adj" = some (.ref adjacencyAddress))
    (getDegrees : env "indeg" = some (.ref degreesAddress))
    (nodesFound : heap[nodesAddress]? = some (.set (stringSet nodes)))
    (adjacencyFound : heap[adjacencyAddress]? = some (.dict adjacency))
    (degreesFound : heap[degreesAddress]? = some (.dict degrees))
    (tablesDistinct : adjacencyAddress ≠ degreesAddress) :
    BlockContinues program env addNodeBody heap env
      (nodeHeap heap nodesAddress adjacencyAddress degreesAddress
        nodes adjacency degrees vertex) := by
  have tested : Evaluates program env (.membership (.name "v") (.name "nodes") true) heap
      (.scalar (.bool (decide (vertex ∉ nodes)))) heap := by
    apply Evaluates.membership
      (.cons (Evaluates.lookup getVertex) (.cons (Evaluates.lookup getNodes) .nil))
    simp [HeapCalls.membership, nodesFound, contains_strings]
  by_cases present : vertex ∈ nodes
  · unfold nodeHeap
    rw [if_pos present]
    apply BlockContinues.branch (flag := false) (by simpa [present] using tested) rfl
    exact .nil
  · have nodesAdjacency : nodesAddress ≠ adjacencyAddress := by
      intro same
      subst adjacencyAddress
      rw [nodesFound] at adjacencyFound
      cases adjacencyFound
    have nodesDegrees : nodesAddress ≠ degreesAddress := by
      intro same
      subst degreesAddress
      rw [nodesFound] at degreesFound
      cases degreesFound
    have adjacencyInside := (List.getElem?_eq_some_iff.mp adjacencyFound).1
    have degreesInside := (List.getElem?_eq_some_iff.mp degreesFound).1
    let added := heap.set nodesAddress (.set (stringSet (nodes ++ [vertex])))
    let allocated := added ++ [.set []]
    let linked := allocated.set adjacencyAddress
      (.dict (ScalarDict.insert adjacency (.string vertex) (.ref heap.length)))
    have add : BlockContinues program env
        [.expr (.call (.attribute (.name "nodes") "add") [.name "v"])] heap env added := by
      apply BlockContinues.discard
      have operation := Evaluates.add (module := program) (globals := [])
        (Evaluates.lookup getNodes) nodesFound
        (Evaluates.lookup getVertex) nodesFound
      simpa [ScalarSet.insert, contains_strings, present, added, stringSet] using operation
    have adjacencyAllocated : allocated[adjacencyAddress]? = some (.dict adjacency) := by
      rw [List.getElem?_append_left (by simpa [added] using adjacencyInside)]
      simp [added, nodesAdjacency, adjacencyFound]
    have degreesAllocated : allocated[degreesAddress]? = some (.dict degrees) := by
      rw [List.getElem?_append_left (by simpa [added] using degreesInside)]
      simp [added, nodesDegrees, degreesFound]
    have allocate : Evaluates program env (.builtinCall .set []) added
        (.ref heap.length) allocated := by
      have operation : Evaluates program env (.builtinCall .set []) added
          (.ref added.length) (added ++ [.set []]) := Evaluates.builtin .nil (by rfl)
      simpa only [allocated, added, List.length_set] using operation
    have link : BlockContinues program env
        [.assign [.subscript (.name "adj") (.name "v")] (.builtinCall .set [])]
        added env linked := by
      apply BlockContinues.setItem (by intros; intro impossible; cases impossible) allocate
        (.cons (Evaluates.lookup getAdjacency) (.cons (Evaluates.lookup getVertex) .nil))
      exact Mutable.setItem_dictionary allocated adjacencyAddress adjacency adjacencyAllocated
        (.string vertex) (.ref heap.length)
    have degreesLinked : linked[degreesAddress]? = some (.dict degrees) := by
      simp [linked, tablesDistinct, degreesAllocated]
    have zeroCount : BlockContinues program env
        [.assign [.subscript (.name "indeg") (.name "v")] (.constant (.int 0))]
        linked env (linked.set degreesAddress
          (.dict (ScalarDict.insert degrees (.string vertex) (number 0)))) := by
      apply BlockContinues.setItem (by intros; intro impossible; cases impossible)
        Evaluates.constant
        (.cons (Evaluates.lookup getDegrees) (.cons (Evaluates.lookup getVertex) .nil))
      exact Mutable.setItem_dictionary linked degreesAddress degrees degreesLinked
        (.string vertex) (number 0)
    unfold nodeHeap
    rw [if_neg present]
    apply BlockContinues.branch (flag := true) (by simpa [present] using tested) rfl
    exact add.append (link.append zeroCount)

theorem addNode_result (heap : Mutable.Heap) (vertex : String)
    (nodesAddress adjacencyAddress degreesAddress : Nat) (nodes : List String)
    (adjacency degrees : List (Constant × Mutable.Value))
    (nodesFound : heap[nodesAddress]? = some (.set (stringSet nodes)))
    (adjacencyFound : heap[adjacencyAddress]? = some (.dict adjacency))
    (degreesFound : heap[degreesAddress]? = some (.dict degrees))
    (tablesDistinct : adjacencyAddress ≠ degreesAddress) :
    ReturnsValue program "__formalsw_local_build_graph_add_node"
      [text vertex, .ref nodesAddress, .ref adjacencyAddress, .ref degreesAddress]
      heap (.scalar .none)
      (nodeHeap heap nodesAddress adjacencyAddress degreesAddress
        nodes adjacency degrees vertex) := by
  apply BlockReturns.function rfl rfl
  change BlockReturns program _ (addNodeBody ++ []) heap _ _
  apply (addNode_block _ heap vertex nodesAddress adjacencyAddress degreesAddress
    nodes adjacency degrees rfl rfl rfl rfl nodesFound adjacencyFound degreesFound
    tablesDistinct).returns
  exact fun _ _ => .next rfl .refl

end solvers.«79».graph

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

/-! The graph builder allocates its three initially empty containers after the caller's objects. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev buildCode : List Stmt := executableBody
  ((findFunctionIn program.body "build_graph").getD ([], [])).2

def initialBody : List Stmt :=
  [.annAssign (.name "nodes") (.subscript (.name "set") (.name "str"))
      (some (.builtinCall .set [])),
   .annAssign (.name "adj")
      (.subscript (.name "dict")
        (.tuple [.name "str", .subscript (.name "set") (.name "str")])) (some (.dict [])),
   .annAssign (.name "indeg") (.subscript (.name "dict") (.tuple [.name "str", .name "int"]))
      (some (.dict []))]

theorem initial_source : buildCode.take 3 = initialBody := rfl

def initialRoots (heap : Mutable.Heap) : Roots :=
  ⟨heap.length, heap.length + 1, heap.length + 2⟩

def emptyStorage : Storage := ⟨[], [], [], [], fun _ => 0, fun _ => []⟩

def initialHeap (heap : Mutable.Heap) : Mutable.Heap := heap ++ [.set [], .dict [], .dict []]

def initialEnv (env : HeapCalls.Env) (heap : Mutable.Heap) : HeapCalls.Env :=
  setLocal (setLocal (setLocal env "nodes" (.ref heap.length))
    "adj" (.ref (heap.length + 1))) "indeg" (.ref (heap.length + 2))

theorem initial_represents (heap : Mutable.Heap) :
    Represents (initialRoots heap) emptyStorage (initialHeap heap) := by
  constructor
  · simp [initialRoots, initialHeap, emptyStorage, stringSet]
  · simp [initialRoots, initialHeap, emptyStorage]
  · simp [initialRoots, initialHeap, emptyStorage]
  · simp [initialRoots]
  · exact List.nodup_nil
  · exact List.nodup_nil
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]
  · simp [emptyStorage]

theorem initial_block (env : HeapCalls.Env) (heap : Mutable.Heap) :
    BlockContinues program env initialBody heap (initialEnv env heap) (initialHeap heap) := by
  have nodes : BlockContinues program env [initialBody[0]] heap
      (setLocal env "nodes" (.ref heap.length)) (heap ++ [.set []]) := by
    apply BlockContinues.annAssign
    exact Evaluates.builtin .nil rfl
  have adjacency : BlockContinues program (setLocal env "nodes" (.ref heap.length))
      [initialBody[1]] (heap ++ [.set []])
      (setLocal (setLocal env "nodes" (.ref heap.length)) "adj" (.ref (heap.length + 1)))
      (heap ++ [.set [], .dict []]) := by
    apply BlockContinues.annAssign
    have evaluated : Evaluates program (setLocal env "nodes" (.ref heap.length)) (.dict [])
        (heap ++ [.set []]) (.ref (heap ++ [Mutable.Object.set []]).length)
        ((heap ++ [.set []]) ++ [.dict []]) := Evaluates.dictionary .nil
    simpa only [List.length_append, List.length_cons, List.length_nil, Nat.zero_add,
      List.append_assoc, List.cons_append, List.nil_append] using evaluated
  have degrees : BlockContinues program
      (setLocal (setLocal env "nodes" (.ref heap.length)) "adj" (.ref (heap.length + 1)))
      [initialBody[2]] (heap ++ [.set [], .dict []]) (initialEnv env heap)
      (initialHeap heap) := by
    apply BlockContinues.annAssign
    have evaluated : Evaluates program
        (setLocal (setLocal env "nodes" (.ref heap.length)) "adj" (.ref (heap.length + 1)))
        (.dict []) (heap ++ [.set [], .dict []])
        (.ref (heap ++ [Mutable.Object.set [], .dict []]).length)
        ((heap ++ [.set [], .dict []]) ++ [.dict []]) := Evaluates.dictionary .nil
    simpa only [initialEnv, initialHeap, List.length_append, List.length_cons, List.length_nil,
      Nat.zero_add, List.append_assoc, List.cons_append, List.nil_append] using evaluated
  exact nodes.append (adjacency.append degrees)

theorem initialize_graph (env : HeapCalls.Env) (heap : Mutable.Heap) :
    BlockContinues program env (buildCode.take 3) heap (initialEnv env heap) (initialHeap heap) ∧
      Represents (initialRoots heap) emptyStorage (initialHeap heap) :=
  ⟨initial_block env heap, initial_represents heap⟩

end solvers.«79».graph

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

/-! A validated attempt performs three vertex updates followed by two deduplicated edge updates. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def attemptBody : List Stmt := match (buildCode[3]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

def nodeCall (name : String) : Stmt :=
  .expr (.call (.name "__formalsw_local_build_graph_add_node")
    [.name name, .name "nodes", .name "adj", .name "indeg"])

def edgeCall (source target : String) : Stmt :=
  .expr (.call (.name "__formalsw_local_build_graph_add_edge")
    [.name source, .name target, .name "adj", .name "indeg"])

def updatesBody : List Stmt :=
  [nodeCall "a", nodeCall "b", nodeCall "c", edgeCall "a" "b", edgeCall "b" "c"]

theorem updates_source : attemptBody.drop 2 = updatesBody := rfl

structure Bindings (roots : Roots) (env : HeapCalls.Env) : Prop where
  nodes : env "nodes" = some (.ref roots.nodes)
  adjacency : env "adj" = some (.ref roots.adjacency)
  degrees : env "indeg" = some (.ref roots.degrees)

theorem initial_bindings (env : HeapCalls.Env) (heap : Mutable.Heap) :
    Bindings (initialRoots heap) (initialEnv env heap) := by
  constructor <;> simp [initialRoots, initialEnv, setLocal]

def insertVertex (vertex : String) (vertices : List String) : List String :=
  if vertex ∈ vertices then vertices else vertices ++ [vertex]

theorem Storage.withNode_vertices (state : Storage) (vertex : String) (fresh : Nat) :
    (state.withNode vertex fresh).vertices = insertVertex vertex state.vertices := by
  by_cases present : vertex ∈ state.vertices <;> simp [Storage.withNode, insertVertex, present]

theorem Storage.withNode_edges (state : Storage) (vertex : String) (fresh : Nat) :
    (state.withNode vertex fresh).edges = state.edges := by
  by_cases present : vertex ∈ state.vertices <;> simp [Storage.withNode, present]

theorem Storage.withNode_mem (state : Storage) (vertex other : String) (fresh : Nat) :
    other ∈ (state.withNode vertex fresh).vertices ↔ other ∈ state.vertices ∨ other = vertex := by
  by_cases present : vertex ∈ state.vertices <;> by_cases same : other = vertex <;>
    simp [Storage.withNode, present, same]

theorem node_call (roots : Roots) (state : Storage) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (represented : Represents roots state heap) (bindings : Bindings roots env)
    (name vertex : String) (getVertex : env name = some (text vertex)) :
    ∃ after, BlockContinues program env [nodeCall name] heap env after ∧
      Represents roots (state.withNode vertex heap.length) after ∧
      ∀ caller, CallerPreserved caller roots state heap →
        CallerPreserved caller roots (state.withNode vertex heap.length) after := by
  refine ⟨_, ?_, represented.withNode roots state heap vertex,
    fun caller saved => saved.withNode caller roots state heap vertex⟩
  apply BlockContinues.discard
  exact Evaluates.call
    (.cons (Evaluates.lookup getVertex) (.cons (Evaluates.lookup bindings.nodes)
      (.cons (Evaluates.lookup bindings.adjacency)
        (.cons (Evaluates.lookup bindings.degrees) .nil))))
    (addNode_result heap vertex roots.nodes roots.adjacency roots.degrees
      state.vertices state.adjacency state.degrees represented.nodesFound
      represented.adjacencyFound represented.degreesFound represented.tablesDistinct)

theorem edge_call (roots : Roots) (state : Storage) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (represented : Represents roots state heap) (bindings : Bindings roots env)
    (sourceName targetName source target : String)
    (getSource : env sourceName = some (text source))
    (getTarget : env targetName = some (text target))
    (sourceMember : source ∈ state.vertices) (targetMember : target ∈ state.vertices) :
    ∃ after, BlockContinues program env [edgeCall sourceName targetName] heap env after ∧
      Represents roots (state.withEdge source target) after ∧
      ∀ caller, CallerPreserved caller roots state heap →
        CallerPreserved caller roots (state.withEdge source target) after := by
  refine ⟨_, ?_, represented.withEdge roots state heap source target sourceMember targetMember,
    fun caller saved => saved.withEdge caller roots state heap source target sourceMember⟩
  apply BlockContinues.discard
  exact Evaluates.call
    (.cons (Evaluates.lookup getSource) (.cons (Evaluates.lookup getTarget)
      (.cons (Evaluates.lookup bindings.adjacency)
        (.cons (Evaluates.lookup bindings.degrees) .nil))))
    (addEdge_result heap source target roots.adjacency roots.degrees (state.address source)
      state.adjacency state.degrees (state.neighbors source)
      (ProjectEulerStatements.TopologicalOrder.indegree target state.edges)
      represented.adjacencyFound (represented.neighborsFound source sourceMember)
      represented.degreesFound (represented.neighborsLookup source sourceMember)
      (represented.counts target targetMember))

/-- The actual five calls retain the invariant, including repeated digits and duplicate edges. -/
theorem updates_preserve (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (represented : Represents roots state heap)
    (bindings : Bindings roots env)
    (a b c : String) (getA : env "a" = some (text a)) (getB : env "b" = some (text b))
    (getC : env "c" = some (text c)) :
    ∃ after next, BlockContinues program env (attemptBody.drop 2) heap env after ∧
      Represents roots next after ∧
      next.vertices = insertVertex c (insertVertex b (insertVertex a state.vertices)) ∧
      next.edges = insertEdge b c (insertEdge a b state.edges) ∧
      ∀ caller, CallerPreserved caller roots state heap →
        CallerPreserved caller roots next after := by
  obtain ⟨heapA, stepA, reprA, savedA⟩ :=
    node_call roots state heap env represented bindings "a" a getA
  let stateA := state.withNode a heap.length
  obtain ⟨heapB, stepB, reprB, savedB⟩ := node_call roots stateA heapA env reprA bindings "b" b getB
  let stateB := stateA.withNode b heapA.length
  obtain ⟨heapC, stepC, reprC, savedC⟩ := node_call roots stateB heapB env reprB bindings "c" c getC
  let stateC := stateB.withNode c heapB.length
  have memberA : a ∈ stateC.vertices := by
    simp only [stateC, stateB, stateA, Storage.withNode_mem, or_true, true_or]
  have memberB : b ∈ stateC.vertices := by
    simp only [stateC, stateB, Storage.withNode_mem, or_true, true_or]
  have memberC : c ∈ stateC.vertices := by
    simp only [stateC, Storage.withNode_mem, or_true]
  obtain ⟨heapAB, stepAB, reprAB, savedAB⟩ :=
    edge_call roots stateC heapC env reprC bindings "a" "b" a b getA getB memberA memberB
  obtain ⟨heapBC, stepBC, reprBC, savedBC⟩ := edge_call roots (stateC.withEdge a b) heapAB env
    reprAB bindings "b" "c" b c getB getC memberB memberC
  refine ⟨heapBC, (stateC.withEdge a b).withEdge b c, ?_, reprBC, ?_, ?_, ?_⟩
  · exact stepA.append (stepB.append (stepC.append (stepAB.append stepBC)))
  · simp only [Storage.withEdge, stateC, stateB, stateA, Storage.withNode_vertices]
  · simp only [Storage.withEdge, stateC, stateB, stateA, Storage.withNode_edges]
  · intro caller saved
    exact savedBC caller (savedAB caller (savedC caller (savedB caller (savedA caller saved))))

end solvers.«79».graph

/-! Valid Unicode attempts pass the source guard and unpack before updating the graph. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def invalidTest : Expr := .boolOp .or
  [.compare (.builtinCall .len [.name "s"]) [(.notEq, .constant (.int 3))],
   .unaryOp .not (.call (.attribute (.name "s") "isdigit") [])]

def invalidBody : List Stmt := match (attemptBody[0]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []

def unpackBody : List Stmt :=
  [.assign [.tuple [.name "a", .name "b", .name "c"]]
    (.tuple [.subscript (.name "s") (.constant (.int 0)),
      .subscript (.name "s") (.constant (.int 1)),
      .subscript (.name "s") (.constant (.int 2))])]

theorem attempt_parts : attemptBody =
    [.ifThen invalidTest invalidBody []] ++ unpackBody ++ updatesBody := rfl

theorem valid_guard (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) (length : s.length = 3)
    (digits : Strings.isDigit s = true) :
    BlockContinues program env [.ifThen invalidTest invalidBody []] heap env heap := by
  have sized : Evaluates program env (.builtinCall .len [.name "s"]) heap (number 3) heap := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getS) .nil)
    simp [primitive, Mutable.length, length, number, text]
  have compared : Evaluates program env
      (.compare (.builtinCall .len [.name "s"]) [(.notEq, .constant (.int 3))])
      heap (.scalar (.bool false)) heap := by
    exact Evaluates.compare sized Evaluates.constant rfl
  have tested : Evaluates program env (.call (.attribute (.name "s") "isdigit") [])
      heap (.scalar (.bool true)) heap := by
    simpa only [digits] using Evaluates.is_digit (module := program)
      (Evaluates.lookup (heap := heap) getS)
      (by intro name equal; cases equal; simp [getS])
  have negated : Evaluates program env
      (.unaryOp .not (.call (.attribute (.name "s") "isdigit") []))
      heap (.scalar (.bool false)) heap := Evaluates.unary tested rfl
  have condition : Evaluates program env invalidTest heap (.scalar (.bool false)) heap :=
    Evaluates.boolean_next compared rfl rfl (Evaluates.boolean_last negated rfl)
  exact BlockContinues.branch condition rfl .nil

theorem Represents.append (roots : Roots) (state : Storage) (heap extra : Mutable.Heap)
    (represented : Represents roots state heap) : Represents roots state (heap ++ extra) := by
  have retained (index : Nat) (object : Mutable.Object) (found : heap[index]? = some object) :
      (heap ++ extra)[index]? = some object := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  exact { represented with
    nodesFound := retained _ _ represented.nodesFound
    adjacencyFound := retained _ _ represented.adjacencyFound
    degreesFound := retained _ _ represented.degreesFound
    neighborsFound := fun vertex member => retained _ _ (represented.neighborsFound vertex member) }

abbrev character (value : Char) : String := String.ofList [value]

def unpackEnv (env : HeapCalls.Env) (a b c : Char) : HeapCalls.Env :=
  assignNames env ["a", "b", "c"] [text (character a), text (character b), text (character c)]

def unpackHeap (heap : Mutable.Heap) (a b c : Char) : Mutable.Heap :=
  heap ++ [.tuple [text (character a), text (character b), text (character c)]]

theorem unpack_three (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String) (a b c : Char)
    (getS : env "s" = some (text s)) (characters : s.toList = [a, b, c]) :
    BlockContinues program env unpackBody heap (unpackEnv env a b c) (unpackHeap heap a b c) := by
  have read (index : Int) (value : Char)
      (found : FormalSW.Py.Builtin.Sequence.get? [a, b, c] index = some value) :
      Evaluates program env (.subscript (.name "s") (.constant (.int index)))
        heap (text (character value)) heap := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getS) Evaluates.constant
    simp [Mutable.getItem, text, Strings.get, scalarNumber, characters, found, character]
  have evaluated : Evaluates program env
      (.tuple [.subscript (.name "s") (.constant (.int 0)),
        .subscript (.name "s") (.constant (.int 1)),
        .subscript (.name "s") (.constant (.int 2))]) heap (.ref heap.length)
      (unpackHeap heap a b c) :=
    Evaluates.tuple (.cons (read 0 a rfl) (.cons (read 1 b rfl) (.cons (read 2 c rfl) .nil)))
      (by intro impossible; cases impossible)
  apply BlockContinues.unpack rfl evaluated
  · simp [Mutable.sequence, unpackHeap]
  · rfl

theorem unpack_bindings (roots : Roots) (env : HeapCalls.Env) (a b c : Char)
    (bindings : Bindings roots env) : Bindings roots (unpackEnv env a b c) := by
  constructor
  · simpa [unpackEnv, assignNames, setLocal] using bindings.nodes
  · simpa [unpackEnv, assignNames, setLocal] using bindings.adjacency
  · simpa [unpackEnv, assignNames, setLocal] using bindings.degrees

/-- One complete successful source iteration, without restricting the digit alphabet to ASCII. -/
theorem valid_attempt (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (represented : Represents roots state heap)
    (bindings : Bindings roots env) (s : String) (getS : env "s" = some (text s))
    (length : s.length = 3) (digits : Strings.isDigit s = true) :
    ∃ a b c after next, s.toList = [a, b, c] ∧
      BlockContinues program env attemptBody heap (unpackEnv env a b c) after ∧
      Represents roots next after ∧ Bindings roots (unpackEnv env a b c) ∧
      next.vertices = insertVertex (character c)
        (insertVertex (character b) (insertVertex (character a) state.vertices)) ∧
      next.edges = insertEdge (character b) (character c)
        (insertEdge (character a) (character b) state.edges) ∧
      ∀ caller, CallerPreserved caller roots state heap →
        CallerPreserved caller roots next after := by
  obtain ⟨a, b, c, characters⟩ := List.length_eq_three.mp
    (show s.toList.length = 3 by simpa using length)
  have guard := valid_guard env heap s getS length digits
  have unpacked := unpack_three env heap s a b c getS characters
  have retained : Represents roots state (unpackHeap heap a b c) :=
    represented.append roots state heap _
  have bound := unpack_bindings roots env a b c bindings
  obtain ⟨after, next, updated, result, vertices, edges, saved⟩ :=
    updates_preserve roots state (unpackHeap heap a b c) (unpackEnv env a b c) retained bound
      (character a) (character b) (character c)
      (by simp [unpackEnv, assignNames, setLocal])
      (by simp [unpackEnv, assignNames, setLocal])
      (by simp [unpackEnv, assignNames, setLocal])
  refine ⟨a, b, c, after, next, characters, ?_, result, bound, vertices, edges, ?_⟩
  · rw [attempt_parts]
    exact (guard.append unpacked).append updated
  · intro caller before
    exact saved caller (before.append caller roots state heap _)

end solvers.«79».graph

/-! Invalid attempts stop before unpacking or graph mutation, with the source's exact message. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def invalidMessage : Expr :=
  .binOp (.binOp (.constant (.string "")) .add (.constant (.string "Invalid attempt: ")))
    .add (.builtinCall .repr [.name "s"])

def invalidException (s : String) : ExceptionObject :=
  { typeName := "ValueError"
    args := [.string ("Invalid attempt: " ++ Strings.reprValue (.string s))] }

theorem invalid_source :
    invalidBody = [.raise (some (.call (.name "ValueError") [invalidMessage]))] := rfl

theorem invalid_guard (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) (invalid : s.length ≠ 3 ∨ Strings.isDigit s = false) :
    Evaluates program env invalidTest heap (.scalar (.bool true)) heap := by
  have sized : Evaluates program env (.builtinCall .len [.name "s"])
      heap (number s.length) heap :=
    Evaluates.builtin (.cons (Evaluates.lookup getS) .nil) rfl
  by_cases length : s.length = 3
  · have digits : Strings.isDigit s = false := invalid.resolve_left (by simpa using length)
    have compared : Evaluates program env
        (.compare (.builtinCall .len [.name "s"]) [(.notEq, .constant (.int 3))])
        heap (.scalar (.bool false)) heap := by
      apply Evaluates.compare sized Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, length]
    have tested : Evaluates program env (.call (.attribute (.name "s") "isdigit") [])
        heap (.scalar (.bool false)) heap := by
      simpa only [digits] using Evaluates.is_digit (module := program)
        (Evaluates.lookup (heap := heap) getS)
        (by intro name equal; cases equal; simp [getS])
    have negated : Evaluates program env
        (.unaryOp .not (.call (.attribute (.name "s") "isdigit") []))
        heap (.scalar (.bool true)) heap := Evaluates.unary tested rfl
    exact Evaluates.boolean_next compared rfl rfl (Evaluates.boolean_last negated rfl)
  · have compared : Evaluates program env
        (.compare (.builtinCall .len [.name "s"]) [(.notEq, .constant (.int 3))])
        heap (.scalar (.bool true)) heap := by
      apply Evaluates.compare sized Evaluates.constant
      simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      omega
    exact Evaluates.boolean_stop compared rfl rfl

theorem invalid_message (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) :
    Evaluates program env invalidMessage heap
      (text ("Invalid attempt: " ++ Strings.reprValue (.string s))) heap := by
  have messageStart : Evaluates program env
      (.binOp (.constant (.string "")) .add (.constant (.string "Invalid attempt: ")))
      heap (text "Invalid attempt: ") heap :=
    Evaluates.binary Evaluates.constant Evaluates.constant rfl
  exact Evaluates.binary messageStart (Evaluates.repr (Evaluates.lookup getS)) rfl

theorem invalid_body (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) (following : List Stmt) (anchors : Nat)
    (stack : List HeapCalls.Frame) :
    Finishes (step program anchors) ⟨.block (invalidBody ++ following) env, stack, heap⟩
      (.ok ⟨.error (invalidException s), [], heap, anchors⟩) := by
  rw [invalid_source]
  refine Finishes.next (following :=
    argumentsState (.raiseException "ValueError") env [] [invalidMessage] stack heap) ?_ ?_
  · rfl
  apply (Evaluates.arguments_reach
    (.cons (invalid_message env heap s getS) .nil) (.raiseException "ValueError") []
    anchors stack).finishes
  exact Finishes.done (by
    simp [step, result, run, primitive, done, protectGenerator,
      Calls.generatorException, invalidException, text])

/-- Both guard failures raise before any of this attempt's graph updates. -/
theorem invalid_attempt (env : HeapCalls.Env) (heap : Mutable.Heap) (s : String)
    (getS : env "s" = some (text s)) (invalid : s.length ≠ 3 ∨ Strings.isDigit s = false)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    Finishes (step program anchors) ⟨.block (attemptBody ++ following) env, stack, heap⟩
      (.ok ⟨.error (invalidException s), [], heap, anchors⟩) := by
  rw [attempt_parts]
  refine Finishes.next (following := ⟨.expr invalidTest env,
    .branch invalidBody [] (unpackBody ++ updatesBody ++ following) env :: stack, heap⟩) ?_ ?_
  · rfl
  apply (invalid_guard env heap s getS invalid anchors _).finishes
  refine Finishes.next (following :=
    ⟨.block (invalidBody ++ (unpackBody ++ updatesBody ++ following)) env, stack, heap⟩) ?_ ?_
  · simp [step, run, Mutable.truth, scalarTruth]
  exact invalid_body env heap s getS _ anchors stack

end solvers.«79».graph

/-! The graph-building loop over a sequence stored in the preserved caller objects. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

/-- Reading the retained sequence reference is stable across private graph allocations. -/
structure SequenceInput (caller : Mutable.Heap) (source : SequenceSource)
    (attempts : List String) : Prop where
  read : ∀ heap, caller <+: heap →
    Mutable.sequence (sequenceValue source) heap = some (.ok (attempts.map text), heap)

theorem SequenceInput.list (caller : Mutable.Heap) (address : Nat) (attempts : List String)
    (found : caller[address]? = some (.list (attempts.map text))) :
    SequenceInput caller (.ref address) attempts := by
  constructor
  intro heap saved
  obtain ⟨suffix, rfl⟩ := saved
  have retained : (caller ++ suffix)[address]? = some (.list (attempts.map text)) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  simp [sequenceValue, Mutable.sequence, retained]

theorem SequenceInput.tuple (caller : Mutable.Heap) (address : Nat) (attempts : List String)
    (found : caller[address]? = some (.tuple (attempts.map text))) :
    SequenceInput caller (.ref address) attempts := by
  constructor
  intro heap saved
  obtain ⟨suffix, rfl⟩ := saved
  have retained : (caller ++ suffix)[address]? = some (.tuple (attempts.map text)) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  simp [sequenceValue, Mutable.sequence, retained]

structure GraphData where
  vertices : List String
  edges : List (String × String)
  deriving DecidableEq

def Storage.graphData (state : Storage) : GraphData := ⟨state.vertices, state.edges⟩

def GraphData.extend (data : GraphData) (s : String) : GraphData :=
  match s.toList with
  | [a, b, c] =>
      ⟨insertVertex (character c) (insertVertex (character b)
        (insertVertex (character a) data.vertices)),
       insertEdge (character b) (character c)
        (insertEdge (character a) (character b) data.edges)⟩
  | _ => data

def ValidAttempt (s : String) : Prop := s.length = 3 ∧ Strings.isDigit s = true

theorem Bindings.setAttempt (roots : Roots) (env : HeapCalls.Env)
    (bound : Bindings roots env) (s : String) : Bindings roots (setLocal env "s" (text s)) := by
  constructor
  · simpa [setLocal] using bound.nodes
  · simpa [setLocal] using bound.adjacency
  · simpa [setLocal] using bound.degrees

theorem valid_iteration (caller : Mutable.Heap) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (represented : Represents roots state heap)
    (saved : CallerPreserved caller roots state heap) (bound : Bindings roots env)
    (s : String) (valid : ValidAttempt s) :
    ∃ after next nextEnv,
      BlockContinues program (setLocal env "s" (text s)) attemptBody heap nextEnv after ∧
      Represents roots next after ∧ CallerPreserved caller roots next after ∧
      Bindings roots nextEnv ∧ next.graphData = state.graphData.extend s := by
  obtain ⟨a, b, c, after, next, characters, executed, result, bindings, vertices, edges, framed⟩ :=
    valid_attempt roots state heap (setLocal env "s" (text s)) represented
      (bound.setAttempt roots env s) s (by simp [setLocal]) valid.1 valid.2
  refine ⟨after, next, _, executed, result, framed caller saved, bindings, ?_⟩
  simp only [GraphData.extend, characters, Storage.graphData]
  exact congrArg₂ GraphData.mk vertices edges

def sequenceLoop (source : SequenceSource) (position : Nat) : Stmt :=
  .sequenceStepLoop "s" source position attemptBody []

/-- Every successful iteration consumes one item and leaves the original sequence untouched. -/
theorem valid_sequence (caller : Mutable.Heap) (source : SequenceSource)
    (attempts : List String) (input : SequenceInput caller source attempts)
    (remaining before : List String) (partition : attempts = before ++ remaining)
    (valid : ∀ s ∈ remaining, ValidAttempt s)
    (roots : Roots) (state : Storage) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (represented : Represents roots state heap) (saved : CallerPreserved caller roots state heap)
    (bound : Bindings roots env) :
    ∃ after next nextEnv,
      BlockContinues program env [sequenceLoop source before.length] heap nextEnv after ∧
      Represents roots next after ∧ CallerPreserved caller roots next after ∧
      Bindings roots nextEnv ∧
      next.graphData = remaining.foldl GraphData.extend state.graphData := by
  induction remaining generalizing before state heap env with
  | nil =>
      refine ⟨heap, state, env, ?_, represented, saved, bound, rfl⟩
      apply BlockContinues.sequence_done (input.read heap saved.unchanged) _ .nil
      simp [partition]
  | cons s rest ih =>
      obtain ⟨middle, current, middleEnv, iteration,
        currentRep, currentSaved, currentBound, data⟩ :=
        valid_iteration caller roots state heap env represented saved bound s (valid s (by simp))
      obtain ⟨after, next, nextEnv, continued, result, framed, bindings, contents⟩ :=
        ih (before ++ [s]) (by simpa [List.append_assoc] using partition)
          (fun t member => valid t (by simp [member])) current middle middleEnv
          currentRep currentSaved currentBound
      refine ⟨after, next, nextEnv, ?_, result, framed, bindings, ?_⟩
      · apply BlockContinues.sequence_step (input.read heap saved.unchanged)
          (value := text s)
        · simp [partition]
        · have resume : BlockContinues program middleEnv
              [sequenceLoop source (before.length + 1)] middle nextEnv after := by
            simpa using continued
          exact iteration.append resume
      · simpa only [List.foldl_cons, data] using contents

end solvers.«79».graph

/-! Graph construction consumes the sequence once and stops at the first invalid attempt. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

instance (s : String) : Decidable (ValidAttempt s) := by
  unfold ValidAttempt
  infer_instance

def GraphData.process (data : GraphData) : List String → Except ExceptionObject GraphData
  | [] => .ok data
  | s :: rest =>
      if ValidAttempt s then (data.extend s).process rest else .error (invalidException s)

def SequenceResult (caller : Mutable.Heap) (roots : Roots) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (source : SequenceSource) (position : Nat) :
    Except ExceptionObject GraphData → Prop
  | .ok data => ∃ after next nextEnv,
      BlockContinues program env [sequenceLoop source position] heap nextEnv after ∧
      Represents roots next after ∧ CallerPreserved caller roots next after ∧
      Bindings roots nextEnv ∧ next.graphData = data
  | .error error => ∃ after, caller <+: after ∧ ∀ following anchors stack,
      Finishes (step program anchors)
        ⟨.block ([sequenceLoop source position] ++ following) env, stack, heap⟩
        (.ok ⟨.error error, [], after, anchors⟩)

theorem sequence_enter (caller : Mutable.Heap) (source : SequenceSource)
    (attempts before rest : List String) (s : String)
    (input : SequenceInput caller source attempts) (partition : attempts = before ++ s :: rest)
    (heap : Mutable.Heap) (saved : caller <+: heap) (env : HeapCalls.Env)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) :
    Reaches program anchors
      ⟨.block ([sequenceLoop source before.length] ++ following) env, stack, heap⟩
      ⟨.block (attemptBody ++ ([sequenceLoop source (before.length + 1)] ++ following))
        (setLocal env "s" (text s)), stack, heap⟩ := by
  have item : (attempts.map text)[before.length]? = some (text s) := by simp [partition]
  exact .next (by
    simp [sequenceLoop, step, run, input.read heap saved, item, List.append_assoc]
    rfl) .refl

/-- The loop has the same success or first-error result as structural graph construction. -/
theorem process_sequence (caller : Mutable.Heap) (source : SequenceSource)
    (attempts : List String) (input : SequenceInput caller source attempts)
    (remaining before : List String) (partition : attempts = before ++ remaining)
    (roots : Roots) (state : Storage) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (represented : Represents roots state heap) (saved : CallerPreserved caller roots state heap)
    (bound : Bindings roots env) :
    SequenceResult caller roots heap env source before.length
      (state.graphData.process remaining) := by
  induction remaining generalizing before state heap env with
  | nil =>
      refine ⟨heap, state, env, ?_, represented, saved, bound, rfl⟩
      apply BlockContinues.sequence_done (input.read heap saved.unchanged) _ .nil
      simp [partition]
  | cons s rest ih =>
      rw [GraphData.process]
      by_cases valid : ValidAttempt s
      · rw [if_pos valid]
        obtain ⟨middle, current, middleEnv, iteration,
          currentRep, currentSaved, currentBound, data⟩ :=
          valid_iteration caller roots state heap env represented saved bound s valid
        have recursive := ih (before ++ [s]) (by simpa [List.append_assoc] using partition)
          current middle middleEnv currentRep currentSaved currentBound
        rw [← data]
        cases outcome : current.graphData.process rest with
        | ok result =>
            rw [outcome] at recursive
            obtain ⟨after, next, nextEnv, continued,
              resultRep, framed, bindings, contents⟩ := recursive
            refine ⟨after, next, nextEnv, ?_, resultRep, framed, bindings, contents⟩
            apply BlockContinues.sequence_step (input.read heap saved.unchanged)
              (value := text s)
            · simp [partition]
            · have resume : BlockContinues program middleEnv
                  [sequenceLoop source (before.length + 1)] middle nextEnv after := by
                simpa using continued
              exact iteration.append resume
        | error error =>
            rw [outcome] at recursive
            obtain ⟨after, framed, failed⟩ := recursive
            refine ⟨after, framed, ?_⟩
            intro following anchors stack
            apply (sequence_enter caller source attempts before rest s input partition heap
              saved.unchanged env following anchors stack).finishes
            apply (iteration ([sequenceLoop source (before.length + 1)] ++ following)
              anchors stack).finishes
            simpa using failed following anchors stack
      · rw [if_neg valid]
        refine ⟨heap, saved.unchanged, ?_⟩
        intro following anchors stack
        apply (sequence_enter caller source attempts before rest s input partition heap
          saved.unchanged env following anchors stack).finishes
        apply invalid_attempt (setLocal env "s" (text s)) heap s (by simp [setLocal])
        by_cases length : s.length = 3
        · exact Or.inr (by simpa [ValidAttempt, length] using valid)
        · exact Or.inl length

end solvers.«79».graph

/-! The source graph builder returns its shared containers or the first validation exception. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def inputLoop : Stmt := .forIn (.name "s") (.name "attempts") attemptBody []

def returnGraph : List Stmt := [.return (.tuple [.name "nodes", .name "adj", .name "indeg"])]

def graphValues (roots : Roots) : List Mutable.Value :=
  [.ref roots.nodes, .ref roots.adjacency, .ref roots.degrees]

theorem build_parts : buildCode = initialBody ++ [inputLoop] ++ returnGraph := rfl

theorem return_graph (roots : Roots) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (bound : Bindings roots env) :
    BlockReturns program env returnGraph heap (.ref heap.length)
      (heap ++ [.tuple (graphValues roots)]) := by
  apply BlockReturns.return_value
  exact Evaluates.tuple (.cons (Evaluates.lookup bound.nodes)
    (.cons (Evaluates.lookup bound.adjacency) (.cons (Evaluates.lookup bound.degrees) .nil)))
    (by intro impossible; cases impossible)

theorem input_enter (caller : Mutable.Heap) (value : Mutable.Value) (attempts : List String)
    (input : SequenceInput caller (sequenceSource value) attempts)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (saved : caller <+: heap)
    (lookup : env "attempts" = some value) (following : List Stmt)
    (anchors : Nat) (stack : List HeapCalls.Frame) :
    Reaches program anchors ⟨.block ([inputLoop] ++ following) env, stack, heap⟩
      ⟨.block ([sequenceLoop (sequenceSource value) 0] ++ following) env, stack, heap⟩ := by
  have read : Mutable.sequence value heap = some (.ok (attempts.map text), heap) := by
    simpa only [sequenceValue_source] using input.read heap saved
  apply Reaches.next rfl
  apply (Evaluates.lookup (module := program) lookup anchors _).trans
  exact .next (by simp [step, run, read, sequenceLoop]) .refl

def GraphOutput (caller : Mutable.Heap) (data : GraphData)
    (heap : Mutable.Heap) (address : Nat) : Prop :=
  ∃ state, Represents (initialRoots caller) state heap ∧
    CallerPreserved caller (initialRoots caller) state heap ∧
    heap[address]? = some (.tuple (graphValues (initialRoots caller))) ∧ state.graphData = data

def BuildBodyResult (caller : Mutable.Heap) (env : HeapCalls.Env) :
    Except ExceptionObject GraphData → Prop
  | .ok data => ∃ after address,
      BlockReturns program env buildCode caller (.ref address) after ∧
      GraphOutput caller data after address
  | .error error => ∃ after, caller <+: after ∧ ∀ anchors stack,
      Finishes (step program anchors) ⟨.block buildCode env, stack, caller⟩
        (.ok ⟨.error error, [], after, anchors⟩)

/-- This body theorem applies to every input represented by the explicit sequence contract. -/
theorem build_body_result (caller : Mutable.Heap) (value : Mutable.Value)
    (attempts : List String) (input : SequenceInput caller (sequenceSource value) attempts)
    (env : HeapCalls.Env) (lookup : env "attempts" = some value) :
    BuildBodyResult caller env (emptyStorage.graphData.process attempts) := by
  have initialized := initial_block env caller
  have represented := initial_represents caller
  have saved := initial_caller caller
  have bound := initial_bindings env caller
  have getInput : initialEnv env caller "attempts" = some value := by
    simpa [initialEnv, setLocal] using lookup
  have executed := process_sequence caller (sequenceSource value) attempts input attempts [] rfl
    (initialRoots caller) emptyStorage (initialHeap caller) (initialEnv env caller)
    represented saved bound
  cases outcome : emptyStorage.graphData.process attempts with
  | ok data =>
      rw [outcome] at executed
      obtain ⟨middle, state, finalEnv, loop, result, framed, bindings, contents⟩ := executed
      have read : Mutable.sequence value (initialHeap caller) =
          some (.ok (attempts.map text), initialHeap caller) := by
        simpa only [sequenceValue_source] using input.read (initialHeap caller) saved.unchanged
      have actualLoop : BlockContinues program (initialEnv env caller) [inputLoop]
          (initialHeap caller) finalEnv middle :=
        BlockContinues.for_sequence (by intros; intro impossible; cases impossible)
          (Evaluates.lookup getInput) read loop
      refine ⟨middle ++ [.tuple (graphValues (initialRoots caller))], middle.length, ?_,
        state, result.append _ _ _ _, framed.append _ _ _ _ _, ?_, contents⟩
      · rw [build_parts]
        exact (initialized.append actualLoop).returns (return_graph _ _ _ bindings)
      · simp
  | error error =>
      rw [outcome] at executed
      obtain ⟨after, framed, failed⟩ := executed
      refine ⟨after, framed, ?_⟩
      intro anchors stack
      rw [build_parts]
      apply (initialized ([inputLoop] ++ returnGraph) anchors stack).finishes
      apply (input_enter caller value attempts input (initialEnv env caller) (initialHeap caller)
        saved.unchanged getInput returnGraph anchors stack).finishes
      exact failed returnGraph anchors stack

abbrev buildArgs : List Arg := ((findFunctionIn program.body "build_graph").getD ([], [])).1
abbrev buildBody : List Stmt := ((findFunctionIn program.body "build_graph").getD ([], [])).2
abbrev buildEnv (value : Mutable.Value) : HeapCalls.Env :=
  functionEnv buildArgs buildBody [value] []

def BuildCallResult (caller : Mutable.Heap) (value : Mutable.Value) :
    Except ExceptionObject GraphData → Prop
  | .ok data => ∃ after address,
      ReturnsValue program "build_graph" [value] caller (.ref address) after ∧
      GraphOutput caller data after address
  | .error error => ∃ after, caller <+: after ∧
      (call program "build_graph" [value] caller).Finishes
        ⟨.error error, [], after, caller.length⟩

theorem build_graph_result (caller : Mutable.Heap) (value : Mutable.Value)
    (attempts : List String) (input : SequenceInput caller (sequenceSource value) attempts) :
    BuildCallResult caller value (emptyStorage.graphData.process attempts) := by
  have body := build_body_result caller value attempts input (buildEnv value) rfl
  cases outcome : emptyStorage.graphData.process attempts with
  | ok data =>
      rw [outcome] at body
      obtain ⟨after, address, returned, output⟩ := body
      exact ⟨after, address, BlockReturns.function rfl rfl returned, output⟩
  | error error =>
      rw [outcome] at body
      obtain ⟨after, framed, failed⟩ := body
      refine ⟨after, framed, ?_⟩
      change Finishes (step program caller.length)
        ⟨.operation (.call "build_graph") [value], [], caller⟩ _
      refine Finishes.next (following := ⟨.block buildCode (buildEnv value), [], caller⟩) ?_ ?_
      · rfl
      · exact failed caller.length []

theorem build_graph_list (caller : Mutable.Heap) (address : Nat) (attempts : List String)
    (found : caller[address]? = some (.list (attempts.map text))) :
    BuildCallResult caller (.ref address) (emptyStorage.graphData.process attempts) :=
  build_graph_result caller (.ref address) attempts
    (SequenceInput.list caller address attempts found)

theorem build_graph_tuple (caller : Mutable.Heap) (address : Nat) (attempts : List String)
    (found : caller[address]? = some (.tuple (attempts.map text))) :
    BuildCallResult caller (.ref address) (emptyStorage.graphData.process attempts) :=
  build_graph_result caller (.ref address) attempts
    (SequenceInput.tuple caller address attempts found)

end solvers.«79».graph

/-! The initial priority queue contains exactly the zero-indegree vertices, in any set order. -/
namespace solvers.«79».ready
open FormalSW.PyAst HeapCalls
open solvers.«79».graph ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

def condition : Expr :=
  .compare (.subscript (.name "indeg") (.name "v")) [(.eq, .constant (.int 0))]

def expression : Expr := .listComp (.name "v") [("v", .name "nodes", [condition])]

def body (collector : Nat) : List Stmt :=
  [.ifThen condition [.comprehensionAppend collector (.name "v")] []]

abbrev topoCode : List Stmt := executableBody
  ((findFunctionIn program.body "topo_sort").getD ([], [])).2

theorem source : topoCode[0]? = some
    (.annAssign (.name "heap") (.subscript (.name "list") (.name "str")) (some expression)) := rfl

abbrev values (items : List String) : List Mutable.Value := items.map text
def eligible (state : Storage) (vertex : String) : Bool := decide (indegree vertex state.edges = 0)
def emit (state : Storage) (collected : List String) (vertex : String) : List String :=
  if eligible state vertex then collected ++ [vertex] else collected

theorem collect_one (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (env : HeapCalls.Env)
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (vertex : String) (member : vertex ∈ state.vertices) (collected : List String) :
    BlockContinues program (setLocal env "v" (text vertex)) (body heap.length)
      (heap ++ [.list (values collected)]) (setLocal env "v" (text vertex))
      (heap ++ [.list (values (emit state collected vertex))]) := by
  have degreesInside := (List.getElem?_eq_some_iff.mp represented.degreesFound).1
  have degreesFound : (heap ++ [.list (values collected)])[roots.degrees]? =
      some (.dict state.degrees) := by
    rw [List.getElem?_append_left degreesInside]
    exact represented.degreesFound
  have read : Evaluates program (setLocal env "v" (text vertex))
      (.subscript (.name "indeg") (.name "v")) (heap ++ [.list (values collected)])
      (number (indegree vertex state.edges)) (heap ++ [.list (values collected)]) := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup (by simpa [setLocal] using getDegrees))
      (Evaluates.lookup (value := text vertex) (by simp [setLocal]))
    exact Mutable.getItem_dictionary _ roots.degrees state.degrees degreesFound
      (.string vertex) (represented.counts vertex member)
  have tested : Evaluates program (setLocal env "v" (text vertex)) condition
      (heap ++ [.list (values collected)]) (.scalar (.bool (eligible state vertex)))
      (heap ++ [.list (values collected)]) := by
    apply Evaluates.compare read Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, eligible]
  by_cases ready : eligible state vertex = true
  · simp only [emit, if_pos ready]
    apply BlockContinues.branch (flag := true) (by simpa only [ready] using tested) rfl
    have appended := BlockContinues.comprehension_append (module := program) (globals := [])
      (env := setLocal env "v" (text vertex)) (address := heap.length)
      (value := text vertex) (items := values collected)
      (heap := heap ++ [.list (values collected)])
      (Evaluates.lookup (name := "v") (by simp [setLocal])) (by simp)
    simpa [values] using appended
  · have notReady : eligible state vertex = false := Bool.eq_false_iff.mpr ready
    simp only [emit, if_neg ready]
    exact BlockContinues.branch (flag := false) (by simpa only [notReady] using tested) rfl .nil

theorem collect_loop (choices : ChoiceSchedule) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap)
    (remaining : List String) (inside : ∀ vertex ∈ remaining, vertex ∈ state.vertices)
    (collected : List String) (env : HeapCalls.Env)
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ result finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (.setStepLoop "v" roots.nodes (stringSet state.vertices) (stringSet remaining)
            (body heap.length) [] :: following) env, stack, heap ++ [.list (values collected)]⟩,
          used⟩
        ⟨⟨.block following finalEnv, stack, heap ++ [.list (values result)]⟩, finalUsed⟩ ∧
      result.Perm (collected ++ remaining.filter (eligible state)) := by
  have nodesInside := (List.getElem?_eq_some_iff.mp represented.nodesFound).1
  have nodesFound : (heap ++ [.list (values collected)])[roots.nodes]? =
      some (.set (stringSet state.vertices)) := by
    rw [List.getElem?_append_left nodesInside]
    exact represented.nodesFound
  cases remainingEq : remaining with
  | nil =>
      refine ⟨collected, env, used, .next ?_ .refl, ?_⟩
      · simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, stringSet, nodesFound]
      · simp
  | cons head tail =>
      let index := choices used % (head :: tail).length
      have bound : index < (head :: tail).length := Nat.mod_lt _ (by simp)
      let vertex := (head :: tail)[index]
      let rest := (head :: tail).eraseIdx index
      let boundEnv := setLocal env "v" (text vertex)
      have selected : (stringSet (head :: tail))[choices used % (tail.length + 1)]? =
          some (.string vertex) := by
        simp only [stringSet, List.getElem?_map]
        rw [show choices used % (tail.length + 1) = index from rfl,
          List.getElem?_eq_getElem bound]
        rfl
      have selected' : (Constant.string head :: tail.map Constant.string)[
          choices used % (tail.length + 1)]? = some (.string vertex) := selected
      have erased : (Constant.string head :: tail.map Constant.string).eraseIdx
          (choices used % (tail.length + 1)) = rest.map Constant.string :=
        List.eraseIdx_map Constant.string (head :: tail) index
      have emitted := collect_one roots state heap represented env getDegrees vertex
        (inside vertex (by rw [remainingEq]; exact List.getElem_mem bound)) collected
      obtain ⟨result, finalEnv, finalUsed, completed, contents⟩ :=
        collect_loop choices roots state heap represented rest
          (fun value member => inside value (by
            rw [remainingEq]
            exact List.mem_of_mem_eraseIdx member))
          (emit state collected vertex) boundEnv (by simpa [boundEnv, setLocal] using getDegrees)
          following anchors stack (used + 1)
      refine ⟨result, finalEnv, finalUsed, .next (following :=
        ⟨⟨.block (body heap.length ++ [.setStepLoop "v" roots.nodes (stringSet state.vertices)
            (stringSet rest) (body heap.length) []] ++ following) boundEnv,
          stack, heap ++ [.list (values collected)]⟩, used + 1⟩) ?_ ?_, ?_⟩
      · simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, nodesFound,
          selected', erased, stringSet, boundEnv,
          body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt]
      · exact (SetCalls.Reaches.of_heap (emitted
          (.setStepLoop "v" roots.nodes (stringSet state.vertices) (stringSet rest)
            (body heap.length) [] :: following) anchors stack) choices (used + 1)).trans completed
      · apply contents.trans
        have filtered := (List.getElem_cons_eraseIdx_perm bound).filter (eligible state)
        have appended := filtered.append_left collected
        by_cases ready : eligible state vertex = true
        · simpa [emit, ready, vertex, rest, List.append_assoc] using appended
        · simpa [emit, ready, vertex, rest] using appended
termination_by remaining.length
decreasing_by
  simp only [remainingEq]
  change ((head :: tail).eraseIdx index).length < (head :: tail).length
  rw [List.length_eraseIdx_of_lt bound]
  simp

theorem expression_result (choices : ChoiceSchedule) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap) (env : HeapCalls.Env)
    (getNodes : env "nodes" = some (.ref roots.nodes))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ result finalUsed, SetCalls.Reaches choices program anchors
        ⟨⟨.expr expression env, stack, heap⟩, used⟩
        ⟨⟨.value (.ref heap.length), stack, heap ++ [.list (values result)]⟩, finalUsed⟩ ∧
      result.Perm (state.vertices.filter (eligible state)) := by
  let inner := comprehensionEnv env [("v", .name "nodes", [condition])]
  have bound : inner "indeg" = some (.ref roots.degrees) := by
    simpa [inner, comprehensionEnv] using getDegrees
  obtain ⟨result, finalEnv, finalUsed, loop, contents⟩ :=
    collect_loop choices roots state heap represented state.vertices (fun _ member => member)
      [] inner bound [] anchors (.comprehensionResult heap.length :: stack) used
  have start := SetCalls.list_comp_set_start program env heap "nodes" "v" roots.nodes
    (stringSet state.vertices) getNodes represented.nodesFound (.name "v") [condition] []
    choices anchors stack used
  have finish : HeapCalls.Reaches program anchors
      ⟨.block [] finalEnv, .comprehensionResult heap.length :: stack,
        heap ++ [.list (values result)]⟩
      ⟨.value (.ref heap.length), stack, heap ++ [.list (values result)]⟩ :=
    .next rfl (.next rfl .refl)
  refine ⟨result, finalUsed, start.trans (loop.trans
    (SetCalls.Reaches.of_heap finish choices finalUsed)), ?_⟩
  simpa using contents

/-- The actual first statement allocates the complete ready queue, preserving surrounding locals. -/
theorem initial_result (choices : ChoiceSchedule) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap) (env : HeapCalls.Env)
    (getNodes : env "nodes" = some (.ref roots.nodes))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ result finalUsed, SetCalls.Reaches choices program anchors
        ⟨⟨.block (topoCode.take 1 ++ following) env, stack, heap⟩, used⟩
        ⟨⟨.block following (setLocal env "heap" (.ref heap.length)), stack,
          heap ++ [.list (values result)]⟩, finalUsed⟩ ∧
      result.Perm (state.vertices.filter (eligible state)) := by
  obtain ⟨result, finalUsed, evaluated, contents⟩ :=
    expression_result choices roots state heap represented env getNodes getDegrees anchors
      (.assign "heap" following env :: stack) used
  refine ⟨result, finalUsed, ?_, contents⟩
  have enter : HeapCalls.Reaches program anchors
      ⟨.block (topoCode.take 1 ++ following) env, stack, heap⟩
      ⟨.expr expression env, .assign "heap" following env :: stack, heap⟩ := .next rfl .refl
  have finish : HeapCalls.Reaches program anchors
      ⟨.value (.ref heap.length), .assign "heap" following env :: stack,
        heap ++ [.list (values result)]⟩
      ⟨.block following (setLocal env "heap" (.ref heap.length)), stack,
        heap ++ [.list (values result)]⟩ := .next rfl .refl
  exact (SetCalls.Reaches.of_heap enter choices used).trans
    (evaluated.trans (SetCalls.Reaches.of_heap finish choices finalUsed))

theorem ready_contents (state : Storage) (result : List String)
    (contents : result.Perm (state.vertices.filter (eligible state))) (vertex : String) :
    vertex ∈ result ↔ vertex ∈ state.vertices ∧ Ready vertex state.edges := by
  rw [contents.mem_iff]
  simp [eligible, indegree_zero_iff]

end solvers.«79».ready

/-! Source initialization orders the ready queue, creates the output list, and copies indegrees. -/
namespace solvers.«79».ready
open FormalSW.PyAst HeapCalls
open solvers.«79».graph
open ProjectEulerStatements.MinHeap
set_option Elab.async false

def prepareBody : List Stmt :=
  [.expr (.builtinCall .heapqHeapify [.name "heap"]),
   .annAssign (.name "out") (.subscript (.name "list") (.name "str")) (some (.list [])),
   .assign [.name "indeg2"] (.builtinCall .dict [.name "indeg"])]

theorem prepare_source : (topoCode.drop 1).take 3 = prepareBody := rfl

def preparedEnv (env : HeapCalls.Env) (base : Nat) : HeapCalls.Env :=
  setLocal (setLocal env "out" (.ref (base + 1))) "indeg2" (.ref (base + 2))

def preparedHeap (heap : Mutable.Heap) (state : Storage) (queue : List String) : Mutable.Heap :=
  heap ++ [.list (values queue), .list [], .dict state.degrees]

theorem prepare_queue (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (represented : Represents roots state heap) (queue : List String) (env : HeapCalls.Env)
    (getQueue : env "heap" = some (.ref heap.length))
    (getDegrees : env "indeg" = some (.ref roots.degrees)) :
    ∃ ordered, BlockContinues program env prepareBody (heap ++ [.list (values queue)])
        (preparedEnv env heap.length) (preparedHeap heap state ordered) ∧
      ordered.Perm queue ∧ Ordered ordered := by
  obtain ⟨ordered, operation, contents, orderedHeap⟩ := heapify_ordered text strings_encodeOrder
    (heap ++ [.list (values queue)]) heap.length queue (by simp [values])
  have heapified : Heapq.heapify [.ref heap.length] (heap ++ [.list (values queue)]) =
      some (.ok (.scalar .none), heap ++ [.list (values ordered)]) := by
    simpa using operation
  have first : BlockContinues program env [prepareBody[0]]
      (heap ++ [.list (values queue)]) env (heap ++ [.list (values ordered)]) := by
    apply BlockContinues.discard
    apply Evaluates.builtin (.cons (Evaluates.lookup getQueue) .nil)
    rw [primitive_heapq_heapify]
    exact heapified
  let outputEnv := setLocal env "out" (.ref (heap.length + 1))
  have second : BlockContinues program env [prepareBody[1]]
      (heap ++ [.list (values ordered)]) outputEnv
      (heap ++ [.list (values ordered), .list []]) := by
    apply BlockContinues.annAssign
    have allocate : Evaluates program env (.list []) (heap ++ [.list (values ordered)])
        (.ref (heap ++ [Mutable.Object.list (values ordered)]).length)
        ((heap ++ [.list (values ordered)]) ++ [.list []]) := Evaluates.list .nil
    simpa only [List.length_append, List.length_cons, List.length_nil, Nat.zero_add,
      List.append_assoc, List.cons_append, List.nil_append] using allocate
  have degreeInside := (List.getElem?_eq_some_iff.mp represented.degreesFound).1
  have degreeFound : (heap ++ [.list (values ordered), .list []])[roots.degrees]? =
      some (.dict state.degrees) := by
    rw [List.getElem?_append_left degreeInside]
    exact represented.degreesFound
  have third : BlockContinues program outputEnv [prepareBody[2]]
      (heap ++ [.list (values ordered), .list []]) (preparedEnv env heap.length)
      (preparedHeap heap state ordered) := by
    apply BlockContinues.assign
    have copy : Evaluates program outputEnv (.builtinCall .dict [.name "indeg"])
        (heap ++ [.list (values ordered), .list []])
        (.ref (heap ++ [Mutable.Object.list (values ordered), .list []]).length)
        ((heap ++ [.list (values ordered), .list []]) ++ [.dict state.degrees]) := by
      apply Evaluates.builtin
        (.cons (Evaluates.lookup (by simpa [outputEnv, setLocal] using getDegrees)) .nil)
      exact primitive_dict_copy _ roots.degrees state.degrees degreeFound
    simpa only [preparedHeap, List.length_append, List.length_cons, List.length_nil,
      Nat.zero_add, List.append_assoc, List.cons_append, List.nil_append] using copy
  exact ⟨ordered, first.append (second.append third), contents, orderedHeap⟩

/-- The first four source statements establish an ordered queue and separate mutable workspaces. -/
theorem initialize_topo (choices : ChoiceSchedule) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (represented : Represents roots state heap) (env : HeapCalls.Env)
    (getNodes : env "nodes" = some (.ref roots.nodes))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ queue finalUsed, SetCalls.Reaches choices program anchors
        ⟨⟨.block (topoCode.take 4 ++ following) env, stack, heap⟩, used⟩
        ⟨⟨.block following
            (preparedEnv (setLocal env "heap" (.ref heap.length)) heap.length),
          stack, preparedHeap heap state queue⟩, finalUsed⟩ ∧
      queue.Perm (state.vertices.filter (eligible state)) ∧ Ordered queue := by
  obtain ⟨collected, finalUsed, collect, contents⟩ :=
    initial_result choices roots state heap represented env getNodes getDegrees
      (prepareBody ++ following) anchors stack used
  obtain ⟨queue, prepare, permutation, ordered⟩ := prepare_queue roots state heap represented
    collected (setLocal env "heap" (.ref heap.length)) (by simp [setLocal])
    (by simpa [setLocal] using getDegrees)
  refine ⟨queue, finalUsed, ?_, permutation.trans contents, ordered⟩
  have same : topoCode.take 4 = topoCode.take 1 ++ prepareBody := rfl
  rw [same, List.append_assoc]
  exact collect.trans (SetCalls.Reaches.of_heap (prepare following anchors stack) choices finalUsed)

end solvers.«79».ready

/-! Source loop updates preserve the priority queue and the separate output and count objects. -/
namespace solvers.«79».topo
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».ready
open ProjectEulerStatements.MinHeap
set_option Elab.async false

def workspace (base : Mutable.Heap) (queue output : List String)
    (degrees : List (Constant × Mutable.Value)) : Mutable.Heap :=
  base ++ [.list (values queue), .list (values output), .dict degrees]

def popBody : List Stmt :=
  [.assign [.name "u"] (.builtinCall .heapqPop [.name "heap"]),
   .expr (.call (.attribute (.name "out") "append") [.name "u"])]

def successorBody : List Stmt :=
  [.augAssign (.subscript (.name "indeg2") (.name "v")) .sub (.constant (.int 1)),
   .ifThen (.compare (.subscript (.name "indeg2") (.name "v"))
     [(.eq, .constant (.int 0))])
     [.expr (.builtinCall .heapqPush [.name "heap", .name "v"])] []]

def loopBody : List Stmt :=
  popBody ++ [.forIn (.name "v") (.subscript (.name "adj") (.name "u")) successorBody []]

theorem loop_source : topoCode[4]? = some (.whileDo (.name "heap") loopBody []) := rfl

/-- The actual pop and append statements emit the least eligible vertex. -/
theorem pop_emit (base : Mutable.Heap) (env : HeapCalls.Env)
    (nodes : List String) (edges : List (String × String)) (vertex : String)
    (rest output : List String) (degrees : List (Constant × Mutable.Value))
    (getQueue : env "heap" = some (.ref base.length))
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (contents : (vertex :: rest).Perm
      (ProjectEulerStatements.TopologicalOrder.eligible nodes edges))
    (ordered : Ordered (vertex :: rest)) :
    ∃ queue, BlockContinues program env popBody
        (workspace base (vertex :: rest) output degrees) (setLocal env "u" (text vertex))
        (workspace base queue (output ++ [vertex]) degrees) ∧
      ProjectEulerStatements.TopologicalOrder.choose nodes edges = some vertex ∧
      queue.Perm rest ∧ Ordered queue := by
  obtain ⟨queue, popped, chosen, permutation, orderedQueue⟩ :=
    heappop_greedy_ordered text strings_encodeOrder nodes edges vertex rest
      (workspace base (vertex :: rest) output degrees) base.length contents ordered
      (by simp [workspace])
  have first : BlockContinues program env [popBody[0]]
      (workspace base (vertex :: rest) output degrees) (setLocal env "u" (text vertex))
      (workspace base queue output degrees) := by
    apply BlockContinues.assign
    apply Evaluates.builtin (.cons (Evaluates.lookup getQueue) .nil)
    rw [primitive_heapq_pop]
    simpa [workspace, values] using popped
  have second : BlockContinues program (setLocal env "u" (text vertex)) [popBody[1]]
      (workspace base queue output degrees) (setLocal env "u" (text vertex))
      (workspace base queue (output ++ [vertex]) degrees) := by
    apply BlockContinues.discard
    have appended := Evaluates.append (module := program) (globals := [])
      (env := setLocal env "u" (text vertex)) (base := .name "out")
      (expression := .name "u") (heap := workspace base queue output degrees)
      (current := values output)
      (Evaluates.lookup (by simpa [setLocal] using getOutput))
      (show (workspace base queue output degrees)[base.length + 1]? =
        some (.list (values output)) by simp [workspace])
      (Evaluates.lookup (value := text vertex) (by simp [setLocal]))
      (by simp [workspace])
    simpa [workspace, values] using appended
  exact ⟨queue, first.append second, chosen, permutation, orderedQueue⟩

def decrease (degrees : List (Constant × Mutable.Value)) (vertex : String) (count : Int) :
    List (Constant × Mutable.Value) :=
  ScalarDict.insert degrees (.string vertex) (number (count - 1))

/-- A successor is enqueued exactly at its zero crossing; Python subtraction stays integral. -/
theorem discharge_one (base : Mutable.Heap) (env : HeapCalls.Env)
    (queue output : List String) (degrees : List (Constant × Mutable.Value))
    (vertex : String) (count : Int)
    (getQueue : env "heap" = some (.ref base.length))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (getVertex : env "v" = some (text vertex))
    (lookup : ScalarDict.lookup degrees (.string vertex) = some (number count))
    (ordered : Ordered queue) :
    ∃ nextQueue, BlockContinues program env successorBody (workspace base queue output degrees)
        env (workspace base nextQueue output (decrease degrees vertex count)) ∧
      nextQueue.Perm (queue ++ if count = 1 then [vertex] else []) ∧ Ordered nextQueue := by
  let changed := decrease degrees vertex count
  have found : (workspace base queue output degrees)[base.length + 2]? =
      some (.dict degrees) := by simp [workspace]
  have decrement : BlockContinues program env [successorBody[0]]
      (workspace base queue output degrees) env (workspace base queue output changed) := by
    apply BlockContinues.augItem (updated := number (count - 1))
      (afterOperation := workspace base queue output degrees)
      (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getDegrees) (Evaluates.lookup getVertex)
      (Mutable.getItem_dictionary _ _ degrees found (.string vertex) lookup) Evaluates.constant
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
    · simpa [workspace, changed, decrease] using Mutable.setItem_dictionary
        (workspace base queue output degrees) (base.length + 2) degrees found
        (.string vertex) (number (count - 1))
  have read : Evaluates program env (.subscript (.name "indeg2") (.name "v"))
      (workspace base queue output changed) (number (count - 1))
      (workspace base queue output changed) := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getDegrees) (Evaluates.lookup getVertex)
    exact Mutable.getItem_dictionary _ _ changed (by simp [workspace])
      (.string vertex) (by simp [changed, decrease, ScalarDict.lookup_insert_self])
  have tested : Evaluates program env
      (.compare (.subscript (.name "indeg2") (.name "v")) [(.eq, .constant (.int 0))])
      (workspace base queue output changed) (.scalar (.bool (decide (count = 1))))
      (workspace base queue output changed) := by
    apply Evaluates.compare read Evaluates.constant
    have zero : count - 1 = 0 ↔ count = 1 := by omega
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, zero]
  by_cases crossing : count = 1
  · obtain ⟨nextQueue, pushed, permutation, orderedQueue⟩ :=
      heappush_ordered text strings_encodeOrder (workspace base queue output changed)
        base.length queue vertex ordered (by simp [workspace])
    refine ⟨nextQueue, decrement.append ?_, by simpa [crossing] using permutation, orderedQueue⟩
    apply BlockContinues.branch (flag := true) (by simpa [crossing] using tested) rfl
    apply BlockContinues.discard
    apply Evaluates.builtin
      (.cons (Evaluates.lookup getQueue) (.cons (Evaluates.lookup getVertex) .nil))
    rw [primitive_heapq_push]
    simpa [workspace, values] using pushed
  · refine ⟨queue, decrement.append ?_, by simp [crossing], ordered⟩
    exact BlockContinues.branch (flag := false) (by simpa [crossing] using tested) rfl .nil

end solvers.«79».topo

/-! Every order of successor traversal decrements the same counts and enqueues the same vertices. -/
namespace solvers.«79».topo
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».ready
open ProjectEulerStatements.MinHeap ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

def Counts (degrees : List (Constant × Mutable.Value)) (counts : String → Int)
    (vertices : List String) : Prop :=
  ∀ vertex ∈ vertices, ScalarDict.lookup degrees (.string vertex) = some (number (counts vertex))

theorem Counts.decrease (degrees : List (Constant × Mutable.Value)) (counts : String → Int)
    (vertices : List String) (counted : Counts degrees counts vertices) (vertex : String) :
    Counts (decrease degrees vertex (counts vertex)) (decrement counts vertex) vertices := by
  intro target member
  by_cases same : vertex = target
  · subst target
    simp [topo.decrease, decrement, ScalarDict.lookup_insert_self]
  · simpa [topo.decrease, decrement, same,
      lookup_insert_other_string degrees vertex target _ same] using counted target member

/-- The original adjacency set stays untouched while the queue and copied dictionary change. -/
theorem discharge_loop (choices : ChoiceSchedule) (base : Mutable.Heap)
    (address : Nat) (original remaining vertices : List String)
    (found : base[address]? = some (.set (stringSet original)))
    (inside : ∀ vertex ∈ remaining, vertex ∈ vertices) (unique : remaining.Nodup)
    (queue output : List String) (degrees : List (Constant × Mutable.Value))
    (counts : String → Int) (counted : Counts degrees counts vertices) (ordered : Ordered queue)
    (env : HeapCalls.Env) (getQueue : env "heap" = some (.ref base.length))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ nextQueue nextDegrees finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (.setStepLoop "v" address (stringSet original) (stringSet remaining)
            successorBody [] :: following) env, stack, workspace base queue output degrees⟩,
          used⟩
        ⟨⟨.block following finalEnv, stack, workspace base nextQueue output nextDegrees⟩,
          finalUsed⟩ ∧
      Counts nextDegrees (decrementAll counts remaining) vertices ∧
      nextQueue.Perm (queue ++ remaining.filter (fun vertex => decide (counts vertex = 1))) ∧
      Ordered nextQueue ∧ (∀ name, name ≠ "v" → finalEnv name = env name) := by
  have addressInside := (List.getElem?_eq_some_iff.mp found).1
  have setFound : (workspace base queue output degrees)[address]? =
      some (.set (stringSet original)) := by
    rw [workspace, List.getElem?_append_left addressInside]
    exact found
  cases remainingEq : remaining with
  | nil =>
      refine ⟨queue, degrees, env, used, .next ?_ .refl, counted, ?_, ordered, ?_⟩
      · simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, stringSet, setFound]
      · simp
      · intros; rfl
  | cons head tail =>
      let index := choices used % (head :: tail).length
      have bound : index < (head :: tail).length := Nat.mod_lt _ (by simp)
      let vertex := (head :: tail)[index]
      let rest := (head :: tail).eraseIdx index
      let boundEnv := setLocal env "v" (text vertex)
      have permutation : (vertex :: rest).Perm (head :: tail) :=
        List.getElem_cons_eraseIdx_perm bound
      have selectedUnique : (vertex :: rest).Nodup :=
        permutation.nodup_iff.mpr (by simpa only [remainingEq] using unique)
      have restUnique : rest.Nodup := (List.nodup_cons.mp selectedUnique).2
      have restInside : ∀ value ∈ rest, value ∈ vertices := by
        intro value member
        exact inside value (by rw [remainingEq]; exact List.mem_of_mem_eraseIdx member)
      have vertexInside : vertex ∈ vertices :=
        inside vertex (by rw [remainingEq]; exact List.getElem_mem bound)
      have selected : (stringSet (head :: tail))[choices used % (tail.length + 1)]? =
          some (.string vertex) := by
        simp only [stringSet, List.getElem?_map]
        rw [show choices used % (tail.length + 1) = index from rfl,
          List.getElem?_eq_getElem bound]
        rfl
      have selected' : (Constant.string head :: tail.map Constant.string)[
          choices used % (tail.length + 1)]? = some (.string vertex) := selected
      have erased : (Constant.string head :: tail.map Constant.string).eraseIdx
          (choices used % (tail.length + 1)) = rest.map Constant.string :=
        List.eraseIdx_map Constant.string (head :: tail) index
      obtain ⟨oneQueue, oneStep, oneContents, oneOrdered⟩ :=
        discharge_one base boundEnv queue output degrees vertex (counts vertex)
          (by simpa [boundEnv, setLocal] using getQueue)
          (by simpa [boundEnv, setLocal] using getDegrees)
          (by simp [boundEnv, setLocal]) (counted vertex vertexInside) ordered
      obtain ⟨nextQueue, nextDegrees, finalEnv, finalUsed, completed, finalCounts,
          contents, finalOrdered, preserved⟩ :=
        discharge_loop choices base address original rest vertices found restInside restUnique
          oneQueue output (decrease degrees vertex (counts vertex)) (decrement counts vertex)
          (counted.decrease degrees counts vertices vertex) oneOrdered boundEnv
          (by simpa [boundEnv, setLocal] using getQueue)
          (by simpa [boundEnv, setLocal] using getDegrees) following anchors stack (used + 1)
      refine ⟨nextQueue, nextDegrees, finalEnv, finalUsed, .next (following :=
        ⟨⟨.block (successorBody ++ [.setStepLoop "v" address (stringSet original)
            (stringSet rest) successorBody []] ++ following) boundEnv,
          stack, workspace base queue output degrees⟩, used + 1⟩) ?_ ?_, ?_, ?_,
        finalOrdered, ?_⟩
      · simp [SetCalls.step, HeapCalls.step, SetCalls.iteratorStep, setFound,
          selected', erased, stringSet, boundEnv,
          successorBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt]
      · exact (SetCalls.Reaches.of_heap (oneStep
          (.setStepLoop "v" address (stringSet original) (stringSet rest)
            successorBody [] :: following) anchors stack) choices (used + 1)).trans completed
      · have same := decrementAll_perm counts (vertex :: rest) (head :: tail) permutation
        change decrementAll (decrement counts vertex) rest =
          decrementAll counts (head :: tail) at same
        simpa only [same] using finalCounts
      · have unchanged : rest.filter (fun target => decide (decrement counts vertex target = 1)) =
            rest.filter (fun target => decide (counts target = 1)) := by
          apply List.filter_congr
          intro target member
          have different : vertex ≠ target := by
            intro same
            exact (List.nodup_cons.mp selectedUnique).1 (same ▸ member)
          simp [decrement, different]
        rw [unchanged] at contents
        apply contents.trans
        apply (oneContents.append_right _).trans
        have filtered := permutation.filter (fun target => decide (counts target = 1))
        have appended := filtered.append_left queue
        by_cases crossing : counts vertex = 1 <;>
          simpa [crossing, List.append_assoc] using appended
      · intro name different
        rw [preserved name different]
        simp [boundEnv, setLocal, different]
termination_by remaining.length
decreasing_by
  simp only [remainingEq]
  change ((head :: tail).eraseIdx index).length < (head :: tail).length
  rw [List.length_eraseIdx_of_lt bound]
  simp

end solvers.«79».topo

/-! One complete source iteration refines removal of the least ready graph vertex. -/
namespace solvers.«79».topo
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».ready
open ProjectEulerStatements.MinHeap ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

theorem visit_successors (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (vertex : String) (member : vertex ∈ state.vertices)
    (queue output : List String) (degrees : List (Constant × Mutable.Value))
    (counts : String → Int) (counted : Counts degrees counts state.vertices)
    (ordered : Ordered queue) (env : HeapCalls.Env)
    (getQueue : env "heap" = some (.ref base.length))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (getAdjacency : env "adj" = some (.ref roots.adjacency))
    (getVertex : env "u" = some (text vertex))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ nextQueue nextDegrees finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (.forIn (.name "v") (.subscript (.name "adj") (.name "u"))
            successorBody [] :: following) env, stack, workspace base queue output degrees⟩,
          used⟩
        ⟨⟨.block following finalEnv, stack, workspace base nextQueue output nextDegrees⟩,
          finalUsed⟩ ∧
      Counts nextDegrees (decrementAll counts (state.neighbors vertex)) state.vertices ∧
      nextQueue.Perm
        (queue ++ (state.neighbors vertex).filter (fun target => decide (counts target = 1))) ∧
      Ordered nextQueue ∧ (∀ name, name ≠ "v" → finalEnv name = env name) := by
  have neighbors := represented.contents vertex member
  have inside : ∀ target ∈ state.neighbors vertex, target ∈ state.vertices := by
    intro target neighbor
    exact (represented.endpoints (vertex, target)
      ((mem_successors vertex target state.edges).mp (neighbors.mem_iff.mp neighbor))).2
  have unique := neighbors.nodup_iff.mpr
    (successors_nodup vertex state.edges represented.edgesUnique)
  have adjacencyFound : (workspace base queue output degrees)[roots.adjacency]? =
      some (.dict state.adjacency) := by
    rw [workspace, List.getElem?_append_left
      (List.getElem?_eq_some_iff.mp represented.adjacencyFound).1]
    exact represented.adjacencyFound
  have setFound : (workspace base queue output degrees)[state.address vertex]? =
      some (.set (stringSet (state.neighbors vertex))) := by
    rw [workspace, List.getElem?_append_left
      (List.getElem?_eq_some_iff.mp (represented.neighborsFound vertex member)).1]
    exact represented.neighborsFound vertex member
  have read : Evaluates program env (.subscript (.name "adj") (.name "u"))
      (workspace base queue output degrees) (.ref (state.address vertex))
      (workspace base queue output degrees) := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getAdjacency) (Evaluates.lookup getVertex)
    exact Mutable.getItem_dictionary _ roots.adjacency state.adjacency adjacencyFound
      (.string vertex) (represented.neighborsLookup vertex member)
  obtain ⟨nextQueue, nextDegrees, finalEnv, finalUsed, completed, countedAfter,
      contents, orderedAfter, preserved⟩ :=
    discharge_loop choices base (state.address vertex) (state.neighbors vertex)
      (state.neighbors vertex) state.vertices (represented.neighborsFound vertex member)
      inside unique queue output degrees counts counted ordered env getQueue getDegrees
      following anchors stack used
  refine ⟨nextQueue, nextDegrees, finalEnv, finalUsed, .next rfl ?_,
    countedAfter, contents, orderedAfter, preserved⟩
  apply (SetCalls.Reaches.of_heap (read anchors
    (.sequence "v" successorBody [] following env :: stack)) choices used).trans
  exact .next (by
    simp [SetCalls.step, HeapCalls.step, run, Mutable.sequence, setFound,
      Mutable.unsupported, SetCalls.iteratorStep]) completed

/-- Queue contents and every stored indegree become those of the residual graph. -/
theorem iteration (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (nodes : List String) (edges : List (String × String))
    (nodesUnique : nodes.Nodup) (edgesUnique : edges.Nodup)
    (endpoints : ∀ edge ∈ edges, edge.1 ∈ nodes ∧ edge.2 ∈ nodes)
    (vertices : ∀ vertex ∈ nodes, vertex ∈ state.vertices)
    (vertex : String) (rest output : List String) (degrees : List (Constant × Mutable.Value))
    (queueContents : (vertex :: rest).Perm
      (ProjectEulerStatements.TopologicalOrder.eligible nodes edges))
    (ordered : Ordered (vertex :: rest))
    (adjacency : (state.neighbors vertex).Perm (successors vertex edges))
    (counted : Counts degrees (fun target => (indegree target edges : Int)) state.vertices)
    (env : HeapCalls.Env) (getQueue : env "heap" = some (.ref base.length))
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (getAdjacency : env "adj" = some (.ref roots.adjacency))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ nextQueue nextDegrees finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (loopBody ++ following) env, stack,
          workspace base (vertex :: rest) output degrees⟩, used⟩
        ⟨⟨.block following finalEnv, stack,
          workspace base nextQueue (output ++ [vertex]) nextDegrees⟩, finalUsed⟩ ∧
      choose nodes edges = some vertex ∧
      Counts nextDegrees (fun target => (indegree target (removeNode vertex edges) : Int))
        state.vertices ∧
      nextQueue.Perm (ProjectEulerStatements.TopologicalOrder.eligible
        (nodes.erase vertex) (removeNode vertex edges)) ∧ Ordered nextQueue ∧
      (∀ name, name ≠ "u" → name ≠ "v" → finalEnv name = env name) := by
  obtain ⟨poppedQueue, popped, chosen, permutation, poppedOrdered⟩ :=
    pop_emit base env nodes edges vertex rest output degrees
      getQueue getOutput queueContents ordered
  have chosenSpec := choose_spec nodes edges vertex chosen
  obtain ⟨nextQueue, nextDegrees, finalEnv, finalUsed, visited, countsAfter,
      contents, orderedAfter, preserved⟩ :=
    visit_successors choices base roots state represented vertex (vertices vertex chosenSpec.1)
      poppedQueue (output ++ [vertex]) degrees (fun target => (indegree target edges : Int))
      counted poppedOrdered (setLocal env "u" (text vertex))
      (by simpa [setLocal] using getQueue) (by simpa [setLocal] using getDegrees)
      (by simpa [setLocal] using getAdjacency) (by simp [setLocal])
      following anchors stack used
  have dischargeContents : nextQueue.Perm (poppedQueue ++
      (discharge (fun target => (indegree target edges : Int)) (state.neighbors vertex)).2) := by
    rw [discharge_new _ _ (adjacency.nodup_iff.mpr (successors_nodup vertex edges edgesUnique))]
    exact contents
  have residual := discharge_queue vertex nodes edges poppedQueue (state.neighbors vertex)
    nextQueue nodesUnique edgesUnique endpoints ((permutation.cons vertex).trans queueContents)
    adjacency dischargeContents
  refine ⟨nextQueue, nextDegrees, finalEnv, finalUsed, ?_, chosen, ?_, residual.2,
    orderedAfter, ?_⟩
  · exact (SetCalls.Reaches.of_heap (popped
      (.forIn (.name "v") (.subscript (.name "adj") (.name "u")) successorBody [] :: following)
      anchors stack) choices used).trans visited
  · rw [discharge_counts] at residual
    simpa only [residual.1] using countsAfter
  · intro name notU notV
    rw [preserved name notV]
    simp [setLocal, notU]

end solvers.«79».topo

/-! The source while loop terminates by removing one remaining vertex at each successful pop. -/
namespace solvers.«79».topo
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».ready
open ProjectEulerStatements.MinHeap ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

def loop : Stmt := .whileDo (.name "heap") loopBody []

/-- Removing a ready vertex leaves the outgoing neighbors of every other vertex unchanged. -/
theorem successors_remove_ready (removed source : String) (edges : List (String × String))
    (ready : Ready removed edges) (different : source ≠ removed) :
    successors source (removeNode removed edges) = successors source edges := by
  unfold successors removeNode
  rw [List.filter_filter]
  congr 1
  apply List.filter_congr
  intro edge member
  have notRemoved := ready edge member
  by_cases matching : edge.1 = source <;> simp [matching, different, notRemoved]

theorem enter_loop (choices : ChoiceSchedule) (base : Mutable.Heap) (env : HeapCalls.Env)
    (vertex : String) (rest output : List String) (degrees : List (Constant × Mutable.Value))
    (getQueue : env "heap" = some (.ref base.length))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.block (loop :: following) env, stack, workspace base (vertex :: rest) output degrees⟩,
        used⟩
      ⟨⟨.block (loopBody ++ loop :: following) env, stack,
        workspace base (vertex :: rest) output degrees⟩, used⟩ := by
  apply SetCalls.Reaches.of_heap
  apply HeapCalls.Reaches.next rfl
  apply ((Evaluates.lookup (module := program) (heap := workspace base (vertex :: rest)
    output degrees) getQueue) anchors _).trans
  exact .next (by
    simp [HeapCalls.step, run, Mutable.truth, workspace, values, loop,
      loopBody, popBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt]) .refl

/-- The completed output agrees with the greedy model, including partial output before a cycle. -/
theorem run_loop (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (nodes : List String) (edges : List (String × String))
    (nodesUnique : nodes.Nodup) (edgesUnique : edges.Nodup)
    (endpoints : ∀ edge ∈ edges, edge.1 ∈ nodes ∧ edge.2 ∈ nodes)
    (vertices : ∀ vertex ∈ nodes, vertex ∈ state.vertices)
    (queue output : List String) (degrees : List (Constant × Mutable.Value))
    (queueContents : queue.Perm (ProjectEulerStatements.TopologicalOrder.eligible nodes edges))
    (ordered : Ordered queue)
    (adjacency : ∀ vertex ∈ nodes, (state.neighbors vertex).Perm (successors vertex edges))
    (counted : Counts degrees (fun target => (indegree target edges : Int)) state.vertices)
    (env : HeapCalls.Env) (getQueue : env "heap" = some (.ref base.length))
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (getDegrees : env "indeg2" = some (.ref (base.length + 2)))
    (getAdjacency : env "adj" = some (.ref roots.adjacency))
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (used : Nat) :
    ∃ emitted nextDegrees finalEnv finalUsed,
      SetCalls.Reaches choices program anchors
        ⟨⟨.block (loop :: following) env, stack, workspace base queue output degrees⟩, used⟩
        ⟨⟨.block following finalEnv, stack, workspace base [] (output ++ emitted) nextDegrees⟩,
          finalUsed⟩ ∧
      emitted.length ≤ nodes.length ∧
      order nodes edges = (if emitted.length = nodes.length then some emitted else none) ∧
      (∀ name, name ≠ "u" → name ≠ "v" → finalEnv name = env name) := by
  cases queue with
  | nil =>
      have noEligible : ProjectEulerStatements.TopologicalOrder.eligible nodes edges = [] :=
        List.length_eq_zero_iff.mp queueContents.length_eq.symm
      have noChoice : choose nodes edges = none := by
        change (ProjectEulerStatements.TopologicalOrder.eligible nodes edges).min? = none
        rw [noEligible]
        rfl
      refine ⟨[], degrees, env, used, ?_, by simp, ?_, ?_⟩
      · apply SetCalls.Reaches.of_heap
        simpa only [List.append_nil] using
          (BlockContinues.while_loop (body := loopBody) (otherwise := [])
            (Evaluates.lookup (module := program) (heap := workspace base [] output degrees)
              getQueue)
            (show Mutable.truth (.ref base.length) (workspace base [] output degrees) =
              some (.ok false, workspace base [] output degrees) by
                simp [Mutable.truth, workspace, values]) BlockContinues.nil
            following anchors stack)
      · by_cases empty : nodes = []
        · simp [order, empty]
        · have positive : nodes.length ≠ 0 := by simpa using empty
          rw [order, if_neg empty, noChoice]
          simp [Ne.symm positive]
      · intros; rfl
  | cons vertex rest =>
      have chosen := head_eq_choose nodes edges vertex rest queueContents ordered
      have selected := choose_spec nodes edges vertex chosen
      have member := selected.1
      have shorter : (nodes.erase vertex).length < nodes.length := by
        rw [List.length_erase_of_mem member]
        have := List.length_pos_of_mem member
        omega
      have nextEndpoints : ∀ edge ∈ removeNode vertex edges,
          edge.1 ∈ nodes.erase vertex ∧ edge.2 ∈ nodes.erase vertex := by
        intro edge retained
        have filtered := List.mem_filter.mp retained
        have different := of_decide_eq_true filtered.2
        have previous := endpoints edge filtered.1
        exact ⟨nodesUnique.mem_erase_iff.mpr ⟨different.1, previous.1⟩,
          nodesUnique.mem_erase_iff.mpr ⟨different.2, previous.2⟩⟩
      have nextAdjacency : ∀ source ∈ nodes.erase vertex,
          (state.neighbors source).Perm (successors source (removeNode vertex edges)) := by
        intro source retained
        have original := nodesUnique.mem_erase_iff.mp retained
        rw [successors_remove_ready vertex source edges selected.2.1 original.1]
        exact adjacency source original.2
      obtain ⟨nextQueue, nextDegrees, nextEnv, nextUsed, stepped, _, nextCounts,
          nextContents, nextOrdered, bindings⟩ :=
        iteration choices base roots state represented nodes edges nodesUnique edgesUnique
          endpoints vertices vertex rest output degrees queueContents ordered
          (adjacency vertex member) counted env getQueue getOutput getDegrees getAdjacency
          (loop :: following) anchors stack used
      obtain ⟨emitted, finalDegrees, finalEnv, finalUsed, completed, lengthBound,
          result, preserved⟩ :=
        run_loop choices base roots state represented (nodes.erase vertex) (removeNode vertex edges)
          (nodesUnique.erase vertex) (edgesUnique.filter _)
          nextEndpoints (fun source retained => vertices source (List.mem_of_mem_erase retained))
          nextQueue (output ++ [vertex]) nextDegrees nextContents nextOrdered nextAdjacency
          nextCounts nextEnv
          (by rw [bindings "heap" (by decide) (by decide)]; exact getQueue)
          (by rw [bindings "out" (by decide) (by decide)]; exact getOutput)
          (by rw [bindings "indeg2" (by decide) (by decide)]; exact getDegrees)
          (by rw [bindings "adj" (by decide) (by decide)]; exact getAdjacency)
          following anchors stack nextUsed
      refine ⟨vertex :: emitted, finalDegrees, finalEnv, finalUsed, ?_, ?_, ?_, ?_⟩
      · have entered := enter_loop choices base env vertex rest output degrees getQueue
          following anchors stack used
        simpa only [List.append_assoc, List.singleton_append] using
          entered.trans (stepped.trans completed)
      · simp only [List.length_cons]
        omega
      · have nonempty : nodes ≠ [] := List.ne_nil_of_mem member
        have size : nodes.length = (nodes.erase vertex).length + 1 := by
          rw [List.length_erase_of_mem member]
          have := List.length_pos_of_mem member
          omega
        rw [order, if_neg nonempty, chosen]
        dsimp only
        rw [result]
        by_cases complete : emitted.length = (nodes.erase vertex).length
        · simp [complete, size]
        · have incomplete : emitted.length + 1 ≠ nodes.length := by omega
          simp [complete, incomplete]
      · intro name notU notV
        exact (preserved name notU notV).trans (bindings name notU notV)
termination_by nodes.length

end solvers.«79».topo

/-! The complete topological-sort body returns the greedy passcode or the exact cycle exception. -/
namespace solvers.«79».topo
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».ready
open ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

def cycleMessage : String := "Constraints contain a cycle; no valid passcode exists."
def cycleException : ExceptionObject :=
  { typeName := "ValueError", args := [.string cycleMessage] }

def resultOutcome : Option (List String) → Except ExceptionObject Mutable.Value
  | none => .error cycleException
  | some output => .ok (text (String.intercalate "" output))

def finishTest : Expr :=
  .compare (.builtinCall .len [.name "out"]) [(.notEq, .builtinCall .len [.name "nodes"])]

def cycleBody : List Stmt :=
  [.raise (some (.call (.name "ValueError") [.constant (.string cycleMessage)]))]

def finishCode : List Stmt :=
  [.ifThen finishTest cycleBody [],
   .return (.call (.attribute (.constant (.string "")) "join") [.name "out"])]

theorem body_source : topoCode = topoCode.take 4 ++ loop :: finishCode := rfl

theorem finish_context (base : Mutable.Heap) (nodes output : List String) (address : Nat)
    (found : base[address]? = some (.set (stringSet nodes)))
    (degrees : List (Constant × Mutable.Value)) (env : HeapCalls.Env)
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (getNodes : env "nodes" = some (.ref address)) (anchors : Nat) (stack : List Frame) :
    match resultOutcome (if output.length = nodes.length then some output else none) with
    | .ok value => Reaches program anchors
        ⟨.block finishCode env, stack, workspace base [] output degrees⟩
        ⟨.value value, stack, workspace base [] output degrees⟩
    | .error error => Finishes (HeapCalls.step program anchors)
        ⟨.block finishCode env, stack, workspace base [] output degrees⟩
        (.ok ⟨.error error, [], workspace base [] output degrees, anchors⟩) := by
  have nodesFound : (workspace base [] output degrees)[address]? =
      some (.set (stringSet nodes)) := by
    rw [workspace, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  have outputLength : Evaluates program env (.builtinCall .len [.name "out"])
      (workspace base [] output degrees) (number output.length)
      (workspace base [] output degrees) := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getOutput) .nil)
    simp [primitive, Mutable.length, workspace, values]
  have nodesLength : Evaluates program env (.builtinCall .len [.name "nodes"])
      (workspace base [] output degrees) (number nodes.length)
      (workspace base [] output degrees) := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getNodes) .nil)
    simp [primitive, Mutable.length, nodesFound, stringSet]
  have tested : Evaluates program env finishTest (workspace base [] output degrees)
      (.scalar (.bool (decide (output.length ≠ nodes.length))))
      (workspace base [] output degrees) := by
    apply Evaluates.compare outputLength nodesLength
    simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    omega
  by_cases complete : output.length = nodes.length
  · simp only [if_pos complete, resultOutcome]
    refine Reaches.next rfl ((tested anchors _).trans ?_)
    refine Reaches.next (following :=
      ⟨.block [finishCode[1]] env, stack, workspace base [] output degrees⟩)
      (by simp [HeapCalls.step, run, Mutable.truth, scalarTruth, finishCode, complete]) ?_
    have joined : BlockReturns program env [finishCode[1]]
        (workspace base [] output degrees) (text (String.intercalate "" output))
        (workspace base [] output degrees) := by
      apply BlockReturns.return_value
      apply Evaluates.join Evaluates.constant (.cons (Evaluates.lookup getOutput) .nil)
      · exact primitive_join_list _ (base.length + 1) "" output (by simp [workspace, values])
      · intros; contradiction
    exact joined anchors stack
  · simp only [if_neg complete, resultOutcome]
    refine Finishes.next rfl ((tested anchors _).finishes ?_)
    refine Finishes.next (following :=
      ⟨.block (cycleBody ++ [finishCode[1]]) env, stack, workspace base [] output degrees⟩)
      (by simp [HeapCalls.step, run, Mutable.truth, scalarTruth, finishCode, complete]) ?_
    refine Finishes.next (following := argumentsState (.raiseException "ValueError") env []
      [.constant (.string cycleMessage)] stack (workspace base [] output degrees)) rfl ?_
    apply (Evaluates.arguments_reach
      (.cons (Evaluates.constant (module := program) (env := env)) .nil)
      (.raiseException "ValueError") [] anchors stack).finishes
    exact .done (by
      simp [HeapCalls.step, result, run, primitive, done, cycleException, protectGenerator,
        Calls.generatorException])

theorem finish (base : Mutable.Heap) (nodes output : List String) (address : Nat)
    (found : base[address]? = some (.set (stringSet nodes)))
    (degrees : List (Constant × Mutable.Value)) (env : HeapCalls.Env)
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (getNodes : env "nodes" = some (.ref address)) (anchors : Nat) :
    Finishes (HeapCalls.step program anchors)
      ⟨.block finishCode env, [], workspace base [] output degrees⟩
      (.ok ⟨resultOutcome (if output.length = nodes.length then some output else none), [],
        workspace base [] output degrees, anchors⟩) := by
  have result := finish_context base nodes output address found degrees env getOutput getNodes
    anchors []
  cases outcome : resultOutcome (if output.length = nodes.length then some output else none) with
  | ok value =>
      exact (show Reaches _ _ _ _ from by simpa only [outcome] using result).finishes (.done rfl)
  | error error => simpa only [outcome] using result

/-- Successful calls resume their caller with the updated traversal counter. -/
def ContextResult (choices : ChoiceSchedule) (anchors : Nat) (work : Work) (stack : List Frame)
    (heap : Mutable.Heap) (used : Nat) (outcome : Except ExceptionObject Mutable.Value)
    (after : Mutable.Heap) : Prop :=
  match outcome with
  | .ok value => ∃ finalUsed, SetCalls.Reaches choices program anchors
      ⟨⟨work, stack, heap⟩, used⟩ ⟨⟨.value value, stack, after⟩, finalUsed⟩
  | .error error => Finishes (SetCalls.step program anchors choices)
      ⟨⟨work, stack, heap⟩, used⟩ (.ok ⟨.error error, [], after, anchors⟩)

theorem ContextResult.finishes (choices : ChoiceSchedule) (anchors : Nat) (work : Work)
    (heap : Mutable.Heap) (used : Nat) (outcome : Except ExceptionObject Mutable.Value)
    (after : Mutable.Heap)
    (result : ContextResult choices anchors work [] heap used outcome after) :
    Finishes (SetCalls.step program anchors choices) ⟨⟨work, [], heap⟩, used⟩
      (.ok ⟨outcome, [], after, anchors⟩) := by
  cases outcome with
  | ok value =>
      obtain ⟨finalUsed, reached⟩ := result
      exact reached.finishes (.done rfl)
  | error error => exact result

theorem ContextResult.before (choices : ChoiceSchedule) (anchors : Nat)
    (work next : Work) (stack : List Frame) (heap middle after : Mutable.Heap)
    (used nextUsed : Nat) (outcome : Except ExceptionObject Mutable.Value)
    (reached : SetCalls.Reaches choices program anchors
      ⟨⟨work, stack, heap⟩, used⟩ ⟨⟨next, stack, middle⟩, nextUsed⟩)
    (result : ContextResult choices anchors next stack middle nextUsed outcome after) :
    ContextResult choices anchors work stack heap used outcome after := by
  cases outcome with
  | ok value =>
      obtain ⟨finalUsed, continued⟩ := result
      exact ⟨finalUsed, reached.trans continued⟩
  | error error => exact reached.finishes result

/-- All traversal schedules produce the abstract ordering result and preserve the input graph. -/
theorem body_context (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (env : HeapCalls.Env) (getNodes : env "nodes" = some (.ref roots.nodes))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (getAdjacency : env "adj" = some (.ref roots.adjacency)) (anchors used : Nat)
    (stack : List Frame) :
    ∃ after, ContextResult choices anchors (.block topoCode env) stack base used
      (resultOutcome (order state.vertices state.edges)) after ∧
      base <+: after := by
  obtain ⟨queue, preparedUsed, initialized, contents, ordered⟩ :=
    initialize_topo choices roots state base represented env getNodes getDegrees
      (loop :: finishCode) anchors stack used
  let prepared := preparedEnv (setLocal env "heap" (.ref base.length)) base.length
  have queueContents : queue.Perm
      (ProjectEulerStatements.TopologicalOrder.eligible state.vertices state.edges) := by
    change queue.Perm (state.vertices.filter (fun vertex =>
      decide (indegree vertex state.edges = 0))) at contents
    simpa only [indegree_zero_iff, ProjectEulerStatements.TopologicalOrder.eligible] using contents
  obtain ⟨emitted, degrees, finalEnv, finalUsed, executed, _, result, preserved⟩ :=
    run_loop choices base roots state represented state.vertices state.edges
      represented.verticesUnique represented.edgesUnique represented.endpoints
      (fun _ member => member)
      queue [] state.degrees queueContents ordered represented.contents represented.counts
      prepared (by simp [prepared, preparedEnv, setLocal])
      (by simp [prepared, preparedEnv, setLocal]) (by simp [prepared, preparedEnv, setLocal])
      (by simpa [prepared, preparedEnv, setLocal] using getAdjacency)
      finishCode anchors stack preparedUsed
  have outputBinding : finalEnv "out" = some (.ref (base.length + 1)) := by
    rw [preserved "out" (by decide) (by decide)]
    simp [prepared, preparedEnv, setLocal]
  have nodesBinding : finalEnv "nodes" = some (.ref roots.nodes) := by
    rw [preserved "nodes" (by decide) (by decide)]
    simpa [prepared, preparedEnv, setLocal] using getNodes
  have finished := finish_context base state.vertices emitted roots.nodes represented.nodesFound
    degrees finalEnv outputBinding nodesBinding anchors stack
  refine ⟨workspace base [] emitted degrees, ?_, ?_⟩
  · rw [body_source]
    rw [← result] at finished
    cases outcome : order state.vertices state.edges with
    | none =>
        apply initialized.finishes
        apply executed.finishes
        exact SetCalls.finishes_of_heap
          (by simpa only [outcome, resultOutcome] using finished) choices finalUsed
    | some output =>
        refine ⟨finalUsed, initialized.trans (executed.trans ?_)⟩
        apply SetCalls.Reaches.of_heap
        simpa only [outcome, resultOutcome] using finished
  · exact ⟨_, rfl⟩

theorem body_result (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (env : HeapCalls.Env) (getNodes : env "nodes" = some (.ref roots.nodes))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (getAdjacency : env "adj" = some (.ref roots.adjacency)) (anchors used : Nat) :
    ∃ after, Finishes (SetCalls.step program anchors choices)
      ⟨⟨.block topoCode env, [], base⟩, used⟩
      (.ok ⟨resultOutcome (order state.vertices state.edges), [], after, anchors⟩) ∧
      base <+: after := by
  obtain ⟨after, result, saved⟩ := body_context choices base roots state represented env
    getNodes getDegrees getAdjacency anchors used []
  exact ⟨after, ContextResult.finishes _ _ _ _ _ _ _ result, saved⟩

theorem call_context (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (anchors used : Nat) (stack : List Frame) :
    ∃ after, ContextResult choices anchors
      (.operation (.call "topo_sort") [.ref roots.nodes, .ref roots.adjacency, .ref roots.degrees])
      stack base used (resultOutcome (order state.vertices state.edges)) after ∧
      base <+: after := by
  let source := (findFunctionIn program.body "topo_sort").getD ([], [])
  let env := functionEnv source.1 source.2
    [.ref roots.nodes, .ref roots.adjacency, .ref roots.degrees] []
  obtain ⟨after, result, saved⟩ := body_context choices base roots state represented env
    rfl rfl rfl anchors used stack
  refine ⟨after, ?_, saved⟩
  cases outcome : order state.vertices state.edges with
  | none =>
      change Finishes _ _ _
      exact .next rfl (by simpa only [outcome, ContextResult, resultOutcome] using result)
  | some output =>
      obtain ⟨finalUsed, reached⟩ :=
        (show ∃ _, _ from by simpa only [outcome, ContextResult, resultOutcome] using result)
      exact ⟨finalUsed, .next rfl reached⟩

/-- The actual Python function call has the same result for every set traversal schedule. -/
theorem call_result (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base) :
    ∃ after, ((SetCalls.call program "topo_sort"
        [.ref roots.nodes, .ref roots.adjacency, .ref roots.degrees] base).run choices).Finishes
      ⟨resultOutcome (order state.vertices state.edges), [], after, base.length⟩ ∧
      base <+: after := by
  let source := (findFunctionIn program.body "topo_sort").getD ([], [])
  let env := functionEnv source.1 source.2
    [.ref roots.nodes, .ref roots.adjacency, .ref roots.degrees] []
  obtain ⟨after, finished, preserved⟩ :=
    body_result choices base roots state represented env rfl rfl rfl base.length 0
  exact ⟨after, .next rfl finished, preserved⟩

end solvers.«79».topo

/-! The actual solver composes graph construction and topological sorting for sequence inputs. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».topo
open ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

def solveOutcome (attempts : List String) : Except ExceptionObject Mutable.Value :=
  match emptyStorage.graphData.process attempts with
  | .error error => .error error
  | .ok data => resultOutcome (order data.vertices data.edges)

def returnTopo : List Stmt :=
  [.return (.call (.name "topo_sort") [.name "nodes", .name "adj", .name "indeg"])]

def unpackGraph : List Stmt :=
  [.assign [.tuple [.name "nodes", .name "adj", .name "indeg"]]
    (.call (.name "build_graph") [.name "attempts"])]

def solveCode : List Stmt := executableBody
  ((findFunctionIn program.body "solve").getD ([], [])).2

theorem solve_source : solveCode = unpackGraph ++ returnTopo := rfl

theorem return_topo (choices : ChoiceSchedule) (heap : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state heap)
    (env : HeapCalls.Env) (getNodes : env "nodes" = some (.ref roots.nodes))
    (getAdjacency : env "adj" = some (.ref roots.adjacency))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (anchors used : Nat) (stack : List Frame) :
    ∃ after, ContextResult choices anchors (.block returnTopo env) stack heap used
      (resultOutcome (order state.vertices state.edges)) after ∧ heap <+: after := by
  obtain ⟨after, called, saved⟩ :=
    call_context choices heap roots state represented anchors used stack
  refine ⟨after, ?_, saved⟩
  apply ContextResult.before _ _ _ _ _ _ _ _ _ _ _ _ called
  apply SetCalls.Reaches.of_heap
  refine Reaches.next rfl (Reaches.next rfl ?_)
  exact Evaluates.arguments_reach
    (.cons (Evaluates.lookup getNodes)
      (.cons (Evaluates.lookup getAdjacency) (.cons (Evaluates.lookup getDegrees) .nil)))
    (.call "topo_sort") [] anchors stack

theorem solve_body_result (choices : ChoiceSchedule) (base : Mutable.Heap) (value : Mutable.Value)
    (attempts : List String) (input : SequenceInput base (sequenceSource value) attempts)
    (env : HeapCalls.Env) (getInput : env "attempts" = some value)
    (anchors used : Nat) (stack : List Frame) :
    ∃ after, ContextResult choices anchors (.block solveCode env) stack base used
      (solveOutcome attempts) after ∧ base <+: after := by
  have built := build_body_result base value attempts input (buildEnv value) rfl
  cases processed : emptyStorage.graphData.process attempts with
  | ok data =>
      rw [processed] at built
      obtain ⟨middle, address, body, state, represented, framed, found, contents⟩ := built
      have called : ReturnsValue program "build_graph" [value] base (.ref address) middle :=
        BlockReturns.function rfl rfl body
      let bound := assignNames env ["nodes", "adj", "indeg"] (graphValues (initialRoots base))
      have unpacked : BlockContinues program env unpackGraph base bound middle := by
        apply BlockContinues.unpack rfl
          (Evaluates.call (.cons (Evaluates.lookup getInput) .nil) called)
        · simp [Mutable.sequence, found]
        · rfl
      obtain ⟨after, returned, saved⟩ := return_topo choices middle (initialRoots base) state
        represented bound (by simp [bound, assignNames, graphValues, setLocal])
        (by simp [bound, assignNames, graphValues, setLocal])
        (by simp [bound, assignNames, graphValues, setLocal]) anchors used stack
      refine ⟨after, ?_, framed.unchanged.trans saved⟩
      simp only [solveOutcome, processed]
      rw [← contents]
      rw [solve_source]
      exact ContextResult.before _ _ _ _ _ _ _ _ _ _ _
        (SetCalls.Reaches.of_heap (unpacked returnTopo anchors stack) choices used) returned
  | error error =>
      rw [processed] at built
      obtain ⟨after, saved, failed⟩ := built
      refine ⟨after, ?_, saved⟩
      simp only [solveOutcome, processed, ContextResult]
      apply SetCalls.finishes_of_heap
      rw [solve_source]
      refine Finishes.next rfl (Finishes.next rfl ?_)
      apply (Evaluates.arguments_reach (.cons (Evaluates.lookup getInput) .nil)
        (.call "build_graph") [] anchors _).finishes
      exact .next rfl (failed anchors _)

abbrev solveArgs : List Arg := ((findFunctionIn program.body "solve").getD ([], [])).1
abbrev solveBody : List Stmt := ((findFunctionIn program.body "solve").getD ([], [])).2

theorem solve_sequence_finishes (choices : ChoiceSchedule) (base : Mutable.Heap)
    (value : Mutable.Value) (attempts : List String)
    (input : SequenceInput base (sequenceSource value) attempts) :
    ∃ after, ((SetCalls.call program "solve" [value] base).run choices).Finishes
      ⟨solveOutcome attempts, [], after, base.length⟩ ∧ base <+: after := by
  obtain ⟨after, result, saved⟩ := solve_body_result choices base value attempts input
    (functionEnv solveArgs solveBody [value] []) rfl base.length 0 []
  exact ⟨after, .next rfl (ContextResult.finishes _ _ _ _ _ _ _ result), saved⟩

theorem solve_list_finishes (choices : ChoiceSchedule) (base : Mutable.Heap)
    (address : Nat) (attempts : List String)
    (found : base[address]? = some (.list (attempts.map text))) :
    ∃ after, ((SetCalls.call program "solve" [.ref address] base).run choices).Finishes
      ⟨solveOutcome attempts, [], after, base.length⟩ ∧ base <+: after :=
  solve_sequence_finishes choices base (.ref address) attempts
    (SequenceInput.list base address attempts found)

theorem solve_tuple_finishes (choices : ChoiceSchedule) (base : Mutable.Heap)
    (address : Nat) (attempts : List String)
    (found : base[address]? = some (.tuple (attempts.map text))) :
    ∃ after, ((SetCalls.call program "solve" [.ref address] base).run choices).Finishes
      ⟨solveOutcome attempts, [], after, base.length⟩ ∧ base <+: after :=
  solve_sequence_finishes choices base (.ref address) attempts
    (SequenceInput.tuple base address attempts found)

end solvers.«79».source

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

/-! Actual graph-builder iterations preserve disjoint objects even after iterator allocation. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

theorem node_call_cells (keep : Nat → Prop) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (represented : Represents roots state heap)
    (bindings : Bindings roots env) (saved : ProtectedCells keep roots state heap)
    (name vertex : String) (getVertex : env name = some (text vertex)) :
    ∃ after, BlockContinues program env [nodeCall name] heap env after ∧
      Represents roots (state.withNode vertex heap.length) after ∧
      ProtectedCells keep roots (state.withNode vertex heap.length) after ∧
      PreservesCells keep heap after ∧
      Mutable.Relocation.emptyTupleAddress after = Mutable.Relocation.emptyTupleAddress heap := by
  refine ⟨_, ?_, represented.withNode roots state heap vertex,
    saved.withNode vertex, saved.node_preserves vertex,
    represented.node_preserves_emptyTuple roots state heap vertex⟩
  apply BlockContinues.discard
  exact Evaluates.call
    (.cons (Evaluates.lookup getVertex) (.cons (Evaluates.lookup bindings.nodes)
      (.cons (Evaluates.lookup bindings.adjacency)
        (.cons (Evaluates.lookup bindings.degrees) .nil))))
    (addNode_result heap vertex roots.nodes roots.adjacency roots.degrees
      state.vertices state.adjacency state.degrees represented.nodesFound
      represented.adjacencyFound represented.degreesFound represented.tablesDistinct)

theorem edge_call_cells (keep : Nat → Prop) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (represented : Represents roots state heap)
    (bindings : Bindings roots env) (saved : ProtectedCells keep roots state heap)
    (sourceName targetName source target : String)
    (getSource : env sourceName = some (text source))
    (getTarget : env targetName = some (text target))
    (sourceMember : source ∈ state.vertices) (targetMember : target ∈ state.vertices) :
    ∃ after, BlockContinues program env [edgeCall sourceName targetName] heap env after ∧
      Represents roots (state.withEdge source target) after ∧
      ProtectedCells keep roots (state.withEdge source target) after ∧
      PreservesCells keep heap after ∧
      Mutable.Relocation.emptyTupleAddress after = Mutable.Relocation.emptyTupleAddress heap := by
  refine ⟨_, ?_, represented.withEdge roots state heap source target sourceMember targetMember,
    saved.withEdge source target sourceMember, saved.edge_preserves source target sourceMember,
    represented.edge_preserves_emptyTuple roots state heap source target sourceMember⟩
  apply BlockContinues.discard
  exact Evaluates.call
    (.cons (Evaluates.lookup getSource) (.cons (Evaluates.lookup getTarget)
      (.cons (Evaluates.lookup bindings.adjacency)
        (.cons (Evaluates.lookup bindings.degrees) .nil))))
    (addEdge_result heap source target roots.adjacency roots.degrees (state.address source)
      state.adjacency state.degrees (state.neighbors source)
      (ProjectEulerStatements.TopologicalOrder.indegree target state.edges)
      represented.adjacencyFound (represented.neighborsFound source sourceMember)
      represented.degreesFound (represented.neighborsLookup source sourceMember)
      (represented.counts target targetMember))

theorem updates_preserve_cells (keep : Nat → Prop) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (represented : Represents roots state heap)
    (bindings : Bindings roots env) (saved : ProtectedCells keep roots state heap)
    (a b c : String) (getA : env "a" = some (text a)) (getB : env "b" = some (text b))
    (getC : env "c" = some (text c)) :
    ∃ after next, BlockContinues program env updatesBody heap env after ∧
      Represents roots next after ∧ ProtectedCells keep roots next after ∧
      PreservesCells keep heap after ∧
      Mutable.Relocation.emptyTupleAddress after = Mutable.Relocation.emptyTupleAddress heap ∧
      next.vertices = insertVertex c (insertVertex b (insertVertex a state.vertices)) ∧
      next.edges = insertEdge b c (insertEdge a b state.edges) := by
  obtain ⟨heapA, stepA, reprA, savedA, readA, internedA⟩ :=
    node_call_cells keep roots state heap env represented bindings saved "a" a getA
  let stateA := state.withNode a heap.length
  obtain ⟨heapB, stepB, reprB, savedB, readB, internedB⟩ :=
    node_call_cells keep roots stateA heapA env reprA bindings savedA "b" b getB
  let stateB := stateA.withNode b heapA.length
  obtain ⟨heapC, stepC, reprC, savedC, readC, internedC⟩ :=
    node_call_cells keep roots stateB heapB env reprB bindings savedB "c" c getC
  let stateC := stateB.withNode c heapB.length
  have memberA : a ∈ stateC.vertices := by
    simp only [stateC, stateB, stateA, Storage.withNode_mem, or_true, true_or]
  have memberB : b ∈ stateC.vertices := by
    simp only [stateC, stateB, Storage.withNode_mem, or_true, true_or]
  have memberC : c ∈ stateC.vertices := by
    simp only [stateC, Storage.withNode_mem, or_true]
  obtain ⟨heapAB, stepAB, reprAB, savedAB, readAB, internedAB⟩ :=
    edge_call_cells keep roots stateC heapC env reprC bindings savedC
      "a" "b" a b getA getB memberA memberB
  obtain ⟨heapBC, stepBC, reprBC, savedBC, readBC, internedBC⟩ :=
    edge_call_cells keep roots (stateC.withEdge a b) heapAB env reprAB bindings savedAB
      "b" "c" b c getB getC memberB memberC
  refine ⟨heapBC, (stateC.withEdge a b).withEdge b c, ?_, reprBC, savedBC,
    readA.trans (readB.trans (readC.trans (readAB.trans readBC))),
    internedBC.trans (internedAB.trans (internedC.trans (internedB.trans internedA))), ?_, ?_⟩
  · exact stepA.append (stepB.append (stepC.append (stepAB.append stepBC)))
  · simp only [Storage.withEdge, stateC, stateB, stateA, Storage.withNode_vertices]
  · simp only [Storage.withEdge, stateC, stateB, stateA, Storage.withNode_edges]

/-- Every selected cell survives the complete successful source iteration, including unpacking. -/
theorem valid_iteration_cells (keep : Nat → Prop) (roots : Roots) (state : Storage)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (represented : Represents roots state heap)
    (saved : ProtectedCells keep roots state heap) (bindings : Bindings roots env)
    (s : String) (valid : ValidAttempt s) :
    ∃ after next nextEnv,
      BlockContinues program (setLocal env "s" (text s)) attemptBody heap nextEnv after ∧
      Represents roots next after ∧ ProtectedCells keep roots next after ∧
      PreservesCells keep heap after ∧ Bindings roots nextEnv ∧
      Mutable.Relocation.emptyTupleAddress after = Mutable.Relocation.emptyTupleAddress heap ∧
      next.graphData = state.graphData.extend s := by
  obtain ⟨a, b, c, characters⟩ := List.length_eq_three.mp
    (show s.toList.length = 3 by simpa using valid.1)
  let start := setLocal env "s" (text s)
  have getS : start "s" = some (text s) := by simp [start, setLocal]
  have bound : Bindings roots start := bindings.setAttempt roots env s
  have guard := valid_guard start heap s getS valid.1 valid.2
  have unpacked := unpack_three start heap s a b c getS characters
  have retained := represented.append roots state heap
    [.tuple [text (character a), text (character b), text (character c)]]
  have unpackBound := unpack_bindings roots start a b c bound
  obtain ⟨after, next, updated, result, framed, read, interned, vertices, edges⟩ :=
    updates_preserve_cells keep roots state (unpackHeap heap a b c) (unpackEnv start a b c)
      retained unpackBound (saved.append _) (character a) (character b) (character c)
      (by simp [unpackEnv, assignNames, setLocal])
      (by simp [unpackEnv, assignNames, setLocal])
      (by simp [unpackEnv, assignNames, setLocal])
  refine ⟨after, next, unpackEnv start a b c, ?_, result, framed,
    (PreservesCells.append keep heap _).trans read, unpackBound, ?_, ?_⟩
  · rw [attempt_parts]
    exact (guard.append unpacked).append updated
  · exact interned.trans
      (Mutable.Relocation.emptyTupleAddress_append_nonempty heap _ (by simp))
  · simp only [GraphData.extend, characters, Storage.graphData]
    exact congrArg₂ GraphData.mk vertices edges

end solvers.«79».graph

/-! The graph-builder's source loop composes actual protocol execution with graph updates.
The iterator may mutate and allocate; graph updates preserve its disjoint resulting cells. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev inputIteratorLoop (env : HeapCalls.Env) (following : List Stmt)
    (stack : List HeapCalls.Frame) : IteratorCalls.Loop :=
  ⟨"s", attemptBody, [], following, env, stack⟩

theorem iterator_valid_iteration (context : Module) (extension : Extends context program)
    (choices : ChoiceSchedule) (anchors used : Nat) (keep : Nat → Prop)
    (roots : Roots) (state : Storage)
    (before yielded : Mutable.Heap) (iterator : Mutable.Value) (operation : Operation)
    (env : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before = some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.ok (text s)) yielded)
    (represented : Represents roots state yielded) (saved : ProtectedCells keep roots state yielded)
    (bindings : Bindings roots env) (following : List Stmt)
    (stack : List HeapCalls.Frame) (resumes : List IteratorCalls.Resume) :
    ∃ after next nextEnv finalUsed,
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
          stack, before⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) nextEnv,
          stack, after⟩, finalUsed⟩ resumes) ∧
      Represents roots next after ∧ ProtectedCells keep roots next after ∧
      PreservesCells keep yielded after ∧ Bindings roots nextEnv ∧
      Mutable.Relocation.emptyTupleAddress after = Mutable.Relocation.emptyTupleAddress yielded ∧
      next.graphData = state.graphData.extend s := by
  obtain ⟨after, next, nextEnv, body, result, framed, retained, nextBindings, interned, data⟩ :=
    valid_iteration_cells keep roots state yielded env represented saved bindings s valid
  obtain ⟨finalUsed, continued⟩ := IteratorCalls.advance_continue
    (loop := inputIteratorLoop env following stack) (anchors := anchors) (used := used)
    (resumes := resumes) isObject bound executed rfl (body.extend extension)
  exact ⟨after, next, nextEnv, finalUsed, continued, result, framed, retained,
    nextBindings, interned, data⟩

/-- One source iteration preserves the relocated graph left by the iterator method. -/
theorem iterator_valid_iteration_layout (context : Module) (extension : Extends context program)
    (choices : ChoiceSchedule) (anchors used : Nat) (keep : Nat → Prop)
    (roots : Roots) (state : Storage)
    (before yielded : Mutable.Heap) (iterator : Mutable.Value) (operation : Operation)
    (env : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before = some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.ok (text s)) yielded)
    (represented : Represents roots state yielded) (saved : ProtectedCells keep roots state yielded)
    (logical : Mutable.Heap) (rename : Nat → Nat)
    (layout : Mutable.Relocation.HeapLayout logical yielded rename)
    (selected : ∀ index < logical.length, keep (rename index))
    (bindings : Bindings roots env) (following : List Stmt)
    (stack : List HeapCalls.Frame) (resumes : List IteratorCalls.Resume) :
    ∃ after next nextEnv finalUsed,
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
          stack, before⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) nextEnv,
          stack, after⟩, finalUsed⟩ resumes) ∧
      Represents roots next after ∧ ProtectedCells keep roots next after ∧
      PreservesCells keep yielded after ∧ Bindings roots nextEnv ∧
      Mutable.Relocation.HeapLayout logical after rename ∧
      Mutable.Relocation.emptyTupleAddress after = Mutable.Relocation.emptyTupleAddress yielded ∧
      next.graphData = state.graphData.extend s := by
  obtain ⟨after, next, nextEnv, finalUsed, run, representedAfter, savedAfter,
      retained, nextBindings, interned, data⟩ :=
    iterator_valid_iteration context extension choices anchors used keep roots state
      before yielded iterator operation env s valid isObject bound executed represented saved
      bindings following stack resumes
  exact ⟨after, next, nextEnv, finalUsed, run, representedAfter, savedAfter, retained,
    nextBindings, retained.layout layout selected, interned, data⟩

/-- One source iteration retains exact relocated objects for the next method call. -/
theorem iterator_valid_iteration_runtime (context : Module) (extension : Extends context program)
    (choices : ChoiceSchedule) (anchors used : Nat) (keep : Nat → Prop)
    (roots : Roots) (state : Storage)
    (before yielded : Mutable.Heap) (iterator : Mutable.Value) (operation : Operation)
    (env : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before = some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.ok (text s)) yielded)
    (represented : Represents roots state yielded) (saved : ProtectedCells keep roots state yielded)
    (logical : Mutable.Heap) (rename : Nat → Nat)
    (layout : Mutable.Relocation.RuntimeLayout logical yielded rename)
    (selected : ∀ index < logical.length, keep (rename index))
    (bindings : Bindings roots env) (following : List Stmt)
    (stack : List HeapCalls.Frame) (resumes : List IteratorCalls.Resume) :
    ∃ after next nextEnv finalUsed,
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
          stack, before⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) nextEnv,
          stack, after⟩, finalUsed⟩ resumes) ∧
      Represents roots next after ∧ ProtectedCells keep roots next after ∧
      PreservesCells keep yielded after ∧ Bindings roots nextEnv ∧
      Mutable.Relocation.RuntimeLayout logical after rename ∧
      Mutable.Relocation.emptyTupleAddress after = Mutable.Relocation.emptyTupleAddress yielded ∧
      next.graphData = state.graphData.extend s := by
  obtain ⟨after, next, nextEnv, finalUsed, run, representedAfter, savedAfter,
      retained, nextBindings, interned, data⟩ :=
    iterator_valid_iteration context extension choices anchors used keep roots state
      before yielded iterator operation env s valid isObject bound executed represented saved
      bindings following stack resumes
  exact ⟨after, next, nextEnv, finalUsed, run, representedAfter, savedAfter, retained,
    nextBindings, retained.runtimeLayout layout selected, interned, data⟩

/-- The next iterator call retains valid references and the same empty-tuple identity. -/
theorem iterator_valid_iteration_execution (context : Module) (extension : Extends context program)
    (choices : ChoiceSchedule) (anchors used : Nat) (keep : Nat → Prop)
    (roots : Roots) (state : Storage)
    (before yielded : Mutable.Heap) (iterator : Mutable.Value) (operation : Operation)
    (env : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before = some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.ok (text s)) yielded)
    (represented : Represents roots state yielded) (saved : ProtectedCells keep roots state yielded)
    (logical : Mutable.Heap) (rename : Nat → Nat)
    (layout : Mutable.Relocation.ExecutionLayout logical yielded rename)
    (selected : ∀ index < logical.length, keep (rename index))
    (bindings : Bindings roots env) (following : List Stmt)
    (stack : List HeapCalls.Frame) (resumes : List IteratorCalls.Resume) :
    ∃ after next nextEnv finalUsed,
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
          stack, before⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) nextEnv,
          stack, after⟩, finalUsed⟩ resumes) ∧
      Represents roots next after ∧ ProtectedCells keep roots next after ∧
      PreservesCells keep yielded after ∧ Bindings roots nextEnv ∧
      Mutable.Relocation.ExecutionLayout logical after rename ∧
      Mutable.Relocation.emptyTupleAddress after = Mutable.Relocation.emptyTupleAddress yielded ∧
      next.graphData = state.graphData.extend s := by
  obtain ⟨after, next, nextEnv, finalUsed, run, representedAfter, savedAfter,
      retained, nextBindings, related, interned, data⟩ :=
    iterator_valid_iteration_runtime context extension choices anchors used keep roots state
      before yielded iterator operation env s valid isObject bound executed represented saved
      logical rename layout.runtime selected bindings following stack resumes
  exact ⟨after, next, nextEnv, finalUsed, run, representedAfter, savedAfter, retained,
    nextBindings, layout.preserves related interned, interned, data⟩

/-- An invalid yielded string stops immediately after validation, retaining next's writes. -/
theorem iterator_invalid_attempt (context : Module) (extension : Extends context program)
    (choices : ChoiceSchedule) (anchors used : Nat) (before yielded : Mutable.Heap)
    (iterator : Mutable.Value)
    (operation : Operation) (env : HeapCalls.Env) (s : String) (invalid : ¬ ValidAttempt s)
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before = some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.ok (text s)) yielded)
    (following : List Stmt) (stack : List HeapCalls.Frame)
    (resumes : List IteratorCalls.Resume) :
    ∃ finalUsed, IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
        stack, before⟩, used⟩ resumes)
      (.completed (.ok ⟨.error (invalidException s), [], yielded, anchors⟩) finalUsed resumes) := by
  obtain ⟨middleUsed, advanced⟩ := IteratorCalls.advance_returns
    (loop := inputIteratorLoop env following stack) (anchors := anchors) (used := used)
    (resumes := resumes) isObject bound executed
  have rejected := invalid_attempt (setLocal env "s" (text s)) yielded s (by simp [setLocal])
    (show s.length ≠ 3 ∨ Strings.isDigit s = false from by
      by_cases length : s.length = 3
      · exact Or.inr (by simpa [ValidAttempt, length] using invalid)
      · exact Or.inl length)
    (sequenceLoop (sequenceSource iterator) 0 :: following) anchors stack
  obtain ⟨finalUsed, failed⟩ := IteratorCalls.completion_of_set
    (SetCalls.finishes_of_heap (finishes_extend rejected extension) choices middleUsed)
    (by intro completion same; cases same; simp [invalidException]) resumes
  refine ⟨finalUsed, advanced.trans ?_⟩
  simpa only [IteratorCalls.loopCode, inputIteratorLoop, Calls.loopContinuation,
    show Calls.loopControl attemptBody = false from rfl, Bool.false_eq_true, ↓reduceIte,
    List.append_assoc, List.singleton_append] using failed


/-- The source for statement evaluates and acquires its input exactly once. -/
theorem iterator_enter (context : Module) (choices : ChoiceSchedule) (anchors used : Nat)
    (before after : Mutable.Heap) (receiver iterator : Mutable.Value)
    (operation advance : Operation) (env : HeapCalls.Env)
    (getInput : env "attempts" = some receiver)
    (isObject : IteratorCalls.isInstance before receiver = true)
    (bound : ProtocolCalls.iterMethod context receiver before = some (.ok operation, before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.ok iterator) after)
    (valid : ProtocolCalls.nextMethod context iterator after = some (.ok (some advance), after))
    (following : List Stmt) (stack : List HeapCalls.Frame)
    (resumes : List IteratorCalls.Resume) :
    ∃ finalUsed, IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (.forIn (.name "s") (.name "attempts") attemptBody [] ::
        following) env, stack, before⟩, used⟩ resumes)
      (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
        stack, after⟩, finalUsed⟩ resumes) :=
  IteratorCalls.for_instance (loop := inputIteratorLoop env following stack)
    (by intro expressions impossible; cases impossible)
    (Evaluates.lookup getInput) isObject bound executed valid

/-- Non-exhaustion failures escape before validation or updates to the graph or checked list. -/
theorem iterator_failed (context : Module) (choices : ChoiceSchedule) (anchors used : Nat)
    (before after : Mutable.Heap) (iterator : Mutable.Value) (operation : Operation)
    (env : HeapCalls.Env) (error : ExceptionObject)
    (notExhausted : error.typeName ≠ "StopIteration")
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before =
      some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation
      before (.error error) after)
    (following : List Stmt) (stack : List HeapCalls.Frame)
    (resumes : List IteratorCalls.Resume) :
    ∃ finalUsed, IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
        stack, before⟩, used⟩ resumes)
      (.completed (.ok ⟨.error error, [], after, anchors⟩) finalUsed resumes) :=
  IteratorCalls.advance_failed (loop := inputIteratorLoop env following stack)
    isObject bound executed notExhausted

/-- Exhaustion finishes the consuming loop with the actual final iterator state. -/
theorem iterator_exhausted (context : Module) (choices : ChoiceSchedule) (anchors used : Nat)
    (before after : Mutable.Heap) (iterator : Mutable.Value) (operation : Operation)
    (env : HeapCalls.Env) (args : List Constant)
    (isObject : IteratorCalls.isInstance before iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator before = some (.ok (some operation), before))
    (executed : IteratorCalls.OperationCompletes context choices used operation before
      (.error ⟨"StopIteration", args⟩) after)
    (following : List Stmt) (stack : List HeapCalls.Frame)
    (resumes : List IteratorCalls.Resume) :
    ∃ finalUsed, IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: following) env,
        stack, before⟩, used⟩ resumes)
      (.running ⟨⟨.block following env, stack, after⟩, finalUsed⟩ resumes) :=
  IteratorCalls.advance_exhausted (loop := inputIteratorLoop env following stack)
    isObject bound executed

end solvers.«79».graph

/-! Logical iterator execution preserves the private graph before the source body updates it. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls Mutable.Relocation
set_option Elab.async false

def callerImage (logical : Mutable.Heap) (rename : Nat → Nat) (address : Nat) : Prop :=
  ∃ index < logical.length, rename index = address

/-- The same transported method trace protects all graph containers, including neighbor sets. -/
theorem iterator_valid_from_logical (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule) (anchors used finalUsed : Nat)
    (logical physical yielded : Mutable.Heap) (rename : Nat → Nat)
    (iterator : Mutable.Value) (operation : Operation) (roots : Roots) (state : Storage)
    (env : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (layout : ExecutionLayout logical physical rename)
    (inside : iterator.edge.Within logical.length)
    (isObject : IteratorCalls.isInstance logical iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator logical = some (.ok (some operation), logical))
    (executed : IteratorCalls.Reaches context logical.length choices []
      (.running ⟨⟨.operation operation [], [], logical⟩, used⟩ [])
      (.completed (.ok ⟨.ok (text s), [], yielded, logical.length⟩) finalUsed []))
    (noGrowth : yielded.length = logical.length)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical)
    (bindings : Bindings roots env)
    (following : List Stmt) (stack : List HeapCalls.Frame) (resumes : List IteratorCalls.Resume) :
    ∃ after next nextEnv nextMap,
      (∀ index < logical.length, nextMap index = rename index) ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          nextEnv, stack, after⟩, finalUsed⟩ resumes) ∧
      Represents roots next after ∧ ProtectedCells (callerImage logical rename) roots next after ∧
      Bindings roots nextEnv ∧ ExecutionLayout yielded after nextMap ∧
      next.graphData = state.graphData.extend s := by
  obtain ⟨other, nextMap, extended, kept, called⟩ := IteratorCalls.Relocation.advance_completion_isolated
    context choices logical physical rename logical.length anchors used finalUsed iterator operation (.ok (text s))
    yielded [] source layout inside (by simp [HeapCalls.Relocation.GlobalsWithin]) isObject bound executed noGrowth
  have outside (address : Nat) (separate : ¬ callerImage logical rename address) :
      ∀ index < logical.length, rename index ≠ address := by
    intro index bounded same
    exact separate ⟨index, bounded, same⟩
  have afterRep : Represents roots state other :=
    { represented with
      nodesFound := (kept roots.nodes _ ⟨layout, represented.nodesFound, outside _ saved.nodes⟩).saved
      adjacencyFound := (kept roots.adjacency _ ⟨layout, represented.adjacencyFound, outside _ saved.adjacency⟩).saved
      degreesFound := (kept roots.degrees _ ⟨layout, represented.degreesFound, outside _ saved.degrees⟩).saved
      neighborsFound := fun vertex member => (kept (state.address vertex) _
        ⟨layout, represented.neighborsFound vertex member, outside _ (saved.neighbors vertex member)⟩).saved }
  have afterSaved : ProtectedCells (callerImage logical rename) roots state other :=
    { saved with inside := fun address selected =>
      Nat.lt_of_lt_of_le (saved.inside address selected) extended.physicalGrows }
  have selected : ∀ index < yielded.length, callerImage logical rename (nextMap index) := by
    intro index bounded
    have old : index < logical.length := by omega
    exact ⟨index, old, (extended.agrees index old).symm⟩
  obtain ⟨after, next, nextEnv, body, result, framed, retained, nextBindings, interned, data⟩ :=
    valid_iteration_cells (callerImage logical rename) roots state other env afterRep afterSaved bindings s valid
  have completion : IteratorCalls.Reaches context anchors choices []
      (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following) env,
        stack, physical⟩, used⟩ resumes)
      (.completed (.ok ⟨.ok (text s), [], other, anchors⟩) finalUsed
        (.advance (inputIteratorLoop env following stack) (relocateValue rename iterator) :: resumes)) := by
    simpa only [HeapCalls.Relocation.relocateGlobals, List.map_nil, Except.map, text, relocateValue] using
      called (inputIteratorLoop env following stack) resumes
  refine ⟨after, next, nextEnv, nextMap, extended.agrees, ?_, result, framed, nextBindings,
    extended.layout.preserves (retained.runtimeLayout extended.layout.runtime selected) interned, data⟩
  apply (IteratorCalls.advanced completion).trans
  have resumed := IteratorCalls.Reaches.of_heap (choices := choices) (used := finalUsed)
    ((body.extend extension) (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
      anchors stack) resumes
  simpa only [IteratorCalls.loopCode, inputIteratorLoop, Calls.loopContinuation,
    show Calls.loopControl attemptBody = false from rfl, Bool.false_eq_true, ↓reduceIte,
    List.append_assoc, List.singleton_append] using resumed

end solvers.«79».graph

/-! Graph construction stops at the first invalid yield without requiring a terminating iterator. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
set_option Elab.async false

theorem process_iterator_prefix (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical finalLogical : Mutable.Heap) (used finalUsed : Nat)
    (attempts : List String)
    (trace : YieldPrefix context choices iterator text logical used attempts finalLogical finalUsed)
    (noGrowth : finalLogical.length = logical.length) (valid : ∀ s ∈ attempts, ValidAttempt s)
    (physical : Mutable.Heap) (rename : Nat → Nat) (roots : Roots) (state : Storage)
    (env : HeapCalls.Env) (layout : ExecutionLayout logical physical rename)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical) (bindings : Bindings roots env)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after next finalEnv,
      Represents roots next after ∧ ProtectedCells (callerImage logical rename) roots next after ∧
      Bindings roots finalEnv ∧ ExecutionLayout finalLogical after rename ∧
      next.graphData = attempts.foldl GraphData.extend state.graphData ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          finalEnv, stack, after⟩, finalUsed⟩ resumes) := by
  induction trace generalizing physical state env with
  | nil => exact ⟨physical, state, env, represented, saved, bindings, layout, rfl, .refl⟩
  | @cons before middle after used nextUsed finalUsed s rest ran later ih =>
      obtain ⟨middleClosed, firstGrows⟩ := ran.effects source layout.closed
      have laterGrows := (later.effects source middleClosed).2
      have firstSize : middle.length = before.length := by omega
      have laterSize : after.length = middle.length := by omega
      obtain ⟨operation, inside, isObject, bound, executed⟩ := ran
      obtain ⟨middlePhysical, current, currentEnv, nextMap, agrees, iteration,
        currentRep, currentSaved, currentBindings, currentLayout, data⟩ :=
        iterator_valid_from_logical context extension source choices anchors used nextUsed before physical middle
          rename iterator operation roots state env s (valid s (by simp)) layout inside isObject bound executed
          firstSize represented saved bindings following stack resumes
      have fixedMap : ExecutionLayout middle middlePhysical rename :=
        currentLayout.congr (fun index bounded => (agrees index (by omega)).symm)
      have images : callerImage middle rename = callerImage before rename := by
        funext address
        simp only [callerImage, firstSize]
      have fixedSaved : ProtectedCells (callerImage middle rename) roots current middlePhysical := by
        rw [images]
        exact currentSaved
      obtain ⟨result, next, finalEnv, finalRep, finalSaved, finalBindings, finalLayout, finalData, exited⟩ :=
        ih laterSize (fun value member => valid value (List.mem_cons_of_mem _ member))
          middlePhysical current currentEnv fixedMap currentRep fixedSaved currentBindings
      refine ⟨result, next, finalEnv, finalRep, ?_, finalBindings, finalLayout, ?_, iteration.trans exited⟩
      · rwa [images] at finalSaved
      · simpa only [List.foldl_cons, data] using finalData


theorem iterator_invalid_from_logical (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical yielded physical : Mutable.Heap) (used finalUsed anchors : Nat)
    (s : String) (invalid : ¬ ValidAttempt s)
    (ran : NextRun context choices iterator logical used (.ok (text s)) yielded finalUsed)
    (noGrowth : yielded.length = logical.length) (rename : Nat → Nat)
    (layout : ExecutionLayout logical physical rename)
    (env : HeapCalls.Env) (following : List Stmt) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after, ExecutionLayout yielded after rename ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.completed (.ok ⟨.error (invalidException s), [], after, anchors⟩) finalUsed resumes) := by
  obtain ⟨after, afterLayout, _, advanced⟩ := ran.scalar_isolated source physical rename layout noGrowth anchors
  have rejected := invalid_attempt (setLocal env "s" (text s)) after s (by simp [setLocal])
    (show s.length ≠ 3 ∨ Strings.isDigit s = false from by
      by_cases length : s.length = 3
      · exact Or.inr (by simpa [ValidAttempt, length] using invalid)
      · exact Or.inl length)
    (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following) anchors stack
  have failed := completion_of_heap (choices := choices) (used := finalUsed) (finishes_extend rejected extension)
    (by intro completion same; cases same; simp [invalidException]) resumes
  refine ⟨after, afterLayout, (advanced (inputIteratorLoop env following stack) resumes).trans ?_⟩
  simpa only [loopCode, inputIteratorLoop, Calls.loopContinuation,
    show Calls.loopControl attemptBody = false from rfl, Bool.false_eq_true, ↓reduceIte,
    List.append_assoc, List.singleton_append] using failed

theorem reject_after_iterator_prefix (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical middle yielded : Mutable.Heap) (used nextUsed finalUsed : Nat)
    (attempts : List String) (s : String)
    (trace : YieldPrefix context choices iterator text logical used attempts middle nextUsed)
    (ran : NextRun context choices iterator middle nextUsed (.ok (text s)) yielded finalUsed)
    (noGrowth : yielded.length = logical.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (invalid : ¬ ValidAttempt s)
    (physical : Mutable.Heap) (rename : Nat → Nat) (roots : Roots) (state : Storage)
    (env : HeapCalls.Env) (layout : ExecutionLayout logical physical rename)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical) (bindings : Bindings roots env)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after, ExecutionLayout yielded after rename ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (.completed (.ok ⟨.error (invalidException s), [], after, anchors⟩) finalUsed resumes) := by
  obtain ⟨middleClosed, prefixGrows⟩ := trace.effects source layout.closed
  have lastGrows := (ran.effects source middleClosed).2
  obtain ⟨middlePhysical, _, nextEnv, _, _, _, nextLayout, _, accepted⟩ :=
    process_iterator_prefix context extension source choices iterator logical middle used nextUsed attempts
      trace (by omega) valid physical rename roots state env layout represented saved bindings following anchors stack resumes
  obtain ⟨after, finalLayout, failed⟩ := iterator_invalid_from_logical context extension source choices
    iterator middle yielded middlePhysical nextUsed finalUsed anchors s invalid ran (by omega) rename nextLayout
      nextEnv following stack resumes
  exact ⟨after, finalLayout, accepted.trans failed⟩

end solvers.«79».graph

/-! Graph construction composes a complete finite iterator traversal and preserves its final caller state. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
set_option Elab.async false

theorem iterator_exit_from_logical (context : Module) (source : IteratorCalls.Relocation.ModuleSource context)
    (choices : ChoiceSchedule) (iterator : Mutable.Value) (logical finalLogical physical : Mutable.Heap)
    (used finalUsed anchors : Nat) (error : ExceptionObject)
    (ran : NextRun context choices iterator logical used (.error error) finalLogical finalUsed)
    (noGrowth : finalLogical.length = logical.length) (rename : Nat → Nat)
    (layout : ExecutionLayout logical physical rename) (roots : Roots) (state : Storage)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical)
    (env : HeapCalls.Env) (following : List Stmt) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after, Represents roots state after ∧ ProtectedCells (callerImage logical rename) roots state after ∧
      ExecutionLayout finalLogical after rename ∧
      ExitResult context choices anchors
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (inputIteratorLoop env following stack) resumes after finalUsed (exitOutcome error) := by
  obtain ⟨after, afterLayout, kept, exited⟩ := ran.exit_isolated source physical rename layout noGrowth anchors
  have outside (address : Nat) (separate : ¬ callerImage logical rename address) :
      ∀ index < logical.length, rename index ≠ address := by
    intro index bounded equal
    exact separate ⟨index, bounded, equal⟩
  have afterRep : Represents roots state after :=
    { represented with
      nodesFound := (kept roots.nodes _ ⟨layout, represented.nodesFound, outside _ saved.nodes⟩).saved
      adjacencyFound := (kept roots.adjacency _ ⟨layout, represented.adjacencyFound, outside _ saved.adjacency⟩).saved
      degreesFound := (kept roots.degrees _ ⟨layout, represented.degreesFound, outside _ saved.degrees⟩).saved
      neighborsFound := fun vertex member => (kept (state.address vertex) _
        ⟨layout, represented.neighborsFound vertex member, outside _ (saved.neighbors vertex member)⟩).saved }
  have afterSaved : ProtectedCells (callerImage logical rename) roots state after := by
    refine { saved with inside := ?_ }
    intro address selected
    obtain ⟨index, bounded, rfl⟩ := selected
    exact afterLayout.runtime.bound index (by omega)
  exact ⟨after, afterRep, afterSaved, afterLayout, exited (inputIteratorLoop env following stack) resumes⟩

theorem process_iterator_traversal (context : Module) (extension : Extends context program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (iterator : Mutable.Value) (logical finalLogical : Mutable.Heap) (used finalUsed : Nat)
    (attempts : List String) (outcome : Except ExceptionObject Unit)
    (trace : Traversal context choices iterator text logical used attempts outcome finalLogical finalUsed)
    (noGrowth : finalLogical.length = logical.length) (valid : ∀ s ∈ attempts, ValidAttempt s)
    (physical : Mutable.Heap) (rename : Nat → Nat) (roots : Roots) (state : Storage)
    (env : HeapCalls.Env) (layout : ExecutionLayout logical physical rename)
    (represented : Represents roots state physical)
    (saved : ProtectedCells (callerImage logical rename) roots state physical) (bindings : Bindings roots env)
    (following : List Stmt) (anchors : Nat) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after next finalEnv,
      Represents roots next after ∧ ProtectedCells (callerImage logical rename) roots next after ∧
      Bindings roots finalEnv ∧ ExecutionLayout finalLogical after rename ∧
      next.graphData = attempts.foldl GraphData.extend state.graphData ∧
      ExitResult context choices anchors
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue rename iterator)) 0 :: following)
          env, stack, physical⟩, used⟩ resumes)
        (inputIteratorLoop finalEnv following stack) resumes after finalUsed outcome := by
  obtain ⟨middle, nextUsed, error, first, last, result⟩ := trace.decompose
  obtain ⟨middleClosed, prefixGrows⟩ := first.effects source layout.closed
  have lastGrows := (last.effects source middleClosed).2
  have sameSize : middle.length = logical.length := by omega
  have images : callerImage middle rename = callerImage logical rename := by
    funext address
    simp only [callerImage, sameSize]
  obtain ⟨middlePhysical, next, nextEnv, nextRep, nextSaved, nextBindings, nextLayout, data, accepted⟩ :=
    process_iterator_prefix context extension source choices iterator logical middle used nextUsed attempts
      first sameSize valid physical rename roots state env layout represented saved bindings following anchors stack resumes
  obtain ⟨after, finalRep, finalSaved, finalLayout, exited⟩ := iterator_exit_from_logical context source choices
    iterator middle finalLogical middlePhysical nextUsed finalUsed anchors error last (by omega) rename nextLayout
      roots next nextRep (by rwa [images]) nextEnv following stack resumes
  refine ⟨after, next, nextEnv, finalRep, ?_, nextBindings, finalLayout, data, ?_⟩
  · rwa [images] at finalSaved
  · rw [← result]
    exact exited.before accepted

end solvers.«79».graph

/-! The solver call initializes its graph and acquires the input before consuming actual iterator prefixes. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation IteratorCalls
open solvers.«79».graph
set_option Elab.async false

def iteratorSolveEnv (receiver : Mutable.Value) : HeapCalls.Env := functionEnv solveArgs solveBody [receiver] []
def iteratorBuildStack (receiver : Mutable.Value) (stack : List HeapCalls.Frame) : List HeapCalls.Frame :=
  .unpack ["nodes", "adj", "indeg"] returnTopo (iteratorSolveEnv receiver) :: stack

theorem solve_iterator_initialize (base : Mutable.Heap) (receiver : Mutable.Value) (anchors : Nat)
    (stack : List HeapCalls.Frame) :
    HeapCalls.Reaches graph.program anchors ⟨.operation (.call "solve") [receiver], stack, base⟩
      ⟨.block (inputLoop :: returnGraph) (initialEnv (buildEnv receiver) base),
        iteratorBuildStack receiver stack, initialHeap base⟩ := by
  apply HeapCalls.Reaches.next (following := ⟨.block solveCode (iteratorSolveEnv receiver), stack, base⟩) rfl
  rw [solve_source]
  refine .next rfl (.next rfl ?_)
  apply (Evaluates.arguments_reach (.cons (Evaluates.lookup
    (show iteratorSolveEnv receiver "attempts" = some receiver from rfl)) .nil)
      (.call "build_graph") [] anchors (iteratorBuildStack receiver stack)).trans
  apply HeapCalls.Reaches.next (following :=
    ⟨.block buildCode (buildEnv receiver), iteratorBuildStack receiver stack, base⟩) rfl
  simpa only [build_parts, List.singleton_append] using
    initial_block (buildEnv receiver) base (inputLoop :: returnGraph) anchors (iteratorBuildStack receiver stack)

theorem initial_execution_layout (base : Mutable.Heap) (closed : Closed base) :
    ExecutionLayout base (initialHeap base) id := by
  have nodes := (ExecutionLayout.refl base closed).append_private_nonempty (.set []) (by simp)
  have adjacency := nodes.append_private_nonempty (.dict []) (by simp)
  have degrees := adjacency.append_private_nonempty (.dict []) (by simp)
  simpa only [initialHeap, List.append_assoc, List.cons_append, List.nil_append] using degrees

theorem initial_protected_cells (base : Mutable.Heap) :
    ProtectedCells (callerImage base id) (initialRoots base) emptyStorage (initialHeap base) := by
  constructor
  · intro address selected
    obtain ⟨index, bounded, same⟩ := selected
    simp only [id_eq] at same
    simp only [initialHeap, List.length_append, List.length_cons, List.length_nil]
    omega
  · simp [callerImage, initialRoots]
  · simp [callerImage, initialRoots]
  · simp [callerImage, initialRoots]
  · simp [emptyStorage]

theorem solve_iterator_entry (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired : Mutable.Heap) (used finalUsed anchors : Nat)
    (closed : Closed base) (ran : AcquireRun context choices receiver base used iterator acquired finalUsed)
    (noGrowth : acquired.length = base.length) (stack : List HeapCalls.Frame) (resumes : List Resume) :
    ∃ after, ExecutionLayout acquired after id ∧ Represents (initialRoots base) emptyStorage after ∧
      ProtectedCells (callerImage acquired id) (initialRoots base) emptyStorage after ∧
      IteratorCalls.Reaches context anchors choices []
        (.running ⟨⟨.operation (.call "solve") [receiver], stack, base⟩, used⟩ resumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource iterator) 0 :: returnGraph)
          (initialEnv (buildEnv receiver) base), iteratorBuildStack receiver stack, after⟩, finalUsed⟩ resumes) := by
  have layout := initial_execution_layout base closed
  have represented := initial_represents base
  have saved := initial_protected_cells base
  obtain ⟨after, afterLayout, kept, entered⟩ := ran.for_name_isolated source (initialHeap base) id layout noGrowth anchors
  have outside (address : Nat) (separate : ¬ callerImage base id address) :
      ∀ index < base.length, id index ≠ address := by
    intro index bounded equal
    exact separate ⟨index, bounded, equal⟩
  have afterRep : Represents (initialRoots base) emptyStorage after :=
    { represented with
      nodesFound := (kept (initialRoots base).nodes _ ⟨layout, represented.nodesFound, outside _ saved.nodes⟩).saved
      adjacencyFound := (kept (initialRoots base).adjacency _ ⟨layout, represented.adjacencyFound, outside _ saved.adjacency⟩).saved
      degreesFound := (kept (initialRoots base).degrees _ ⟨layout, represented.degreesFound, outside _ saved.degrees⟩).saved
      neighborsFound := by simp [emptyStorage] }
  have afterSaved : ProtectedCells (callerImage acquired id) (initialRoots base) emptyStorage after := by
    have images : callerImage acquired id = callerImage base id := by
      funext address
      simp only [callerImage, noGrowth]
    rw [images]
    refine { saved with inside := ?_ }
    intro address selected
    obtain ⟨index, bounded, rfl⟩ := selected
    exact afterLayout.runtime.bound index (by omega)
  refine ⟨after, afterLayout, afterRep, afterSaved, ?_⟩
  apply (IteratorCalls.Reaches.of_heap ((solve_iterator_initialize base receiver anchors stack).extend extension) resumes).trans
  simpa only [relocateValue_id, loopCode, inputIteratorLoop, inputLoop] using entered
    (inputIteratorLoop (initialEnv (buildEnv receiver) base) returnGraph (iteratorBuildStack receiver stack)) resumes "attempts"
    (by simpa only [relocateValue_id] using
      (show initialEnv (buildEnv receiver) base "attempts" = some receiver from rfl))

theorem solve_iterator_rejection (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired middle yielded : Mutable.Heap)
    (acquiredUsed nextUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String) (s : String)
    (trace : YieldPrefix context choices iterator graph.text acquired acquiredUsed attempts middle nextUsed)
    (ran : NextRun context choices iterator middle nextUsed (.ok (graph.text s)) yielded finalUsed)
    (noGrowth : yielded.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value)
    (invalid : ¬ ValidAttempt s) :
    ∃ after, ExecutionLayout yielded after id ∧
      ((IteratorCalls.call context "solve" [receiver] base).run choices).Finishes
        ⟨.error (invalidException s), [], after, base.length⟩ := by
  obtain ⟨acquiredClosed, acquisitionGrows, _⟩ := acquisition.effects source closed
  obtain ⟨middleClosed, prefixGrows⟩ := trace.effects source acquiredClosed
  have lastGrows := (ran.effects source middleClosed).2
  obtain ⟨initialized, layout, represented, saved, entered⟩ := solve_iterator_entry context extension source choices
    receiver iterator base acquired 0 acquiredUsed base.length closed acquisition (by omega) [] []
  obtain ⟨after, finalLayout, failed⟩ := graph.reject_after_iterator_prefix context extension source choices iterator
    acquired middle yielded acquiredUsed nextUsed finalUsed attempts s trace ran (by omega) valid invalid
      initialized id (initialRoots base) emptyStorage (initialEnv (buildEnv receiver) base) layout represented saved
      (initial_bindings (buildEnv receiver) base) returnGraph base.length (iteratorBuildStack receiver []) []
  refine ⟨after, finalLayout, ?_⟩
  exact (entered.trans (by simpa only [relocateValue_id] using failed)).finishes (.done rfl)

theorem solve_iterator_failure (context : Module) (extension : Extends context graph.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (receiver iterator : Mutable.Value) (base acquired yielded : Mutable.Heap)
    (acquiredUsed finalUsed : Nat) (closed : Closed base)
    (acquisition : AcquireRun context choices receiver base 0 iterator acquired acquiredUsed)
    (attempts : List String) (error : ExceptionObject)
    (trace : Traversal context choices iterator graph.text acquired acquiredUsed attempts (.error error) yielded finalUsed)
    (noGrowth : yielded.length = base.length) (valid : ∀ value ∈ attempts, ValidAttempt value) :
    ∃ after, ExecutionLayout yielded after id ∧
      ((IteratorCalls.call context "solve" [receiver] base).run choices).Finishes
        ⟨.error error, [], after, base.length⟩ := by
  obtain ⟨acquiredClosed, acquisitionGrows, _⟩ := acquisition.effects source closed
  have traversalGrows := (trace.effects source acquiredClosed).2
  obtain ⟨initialized, layout, represented, saved, entered⟩ := solve_iterator_entry context extension source choices
    receiver iterator base acquired 0 acquiredUsed base.length closed acquisition (by omega) [] []
  obtain ⟨after, _, _, _, _, _, finalLayout, _, failed⟩ := process_iterator_traversal context extension source choices iterator
    acquired yielded acquiredUsed finalUsed attempts (.error error) trace (by omega) valid
      initialized id (initialRoots base) emptyStorage (initialEnv (buildEnv receiver) base) layout represented saved
      (initial_bindings (buildEnv receiver) base) returnGraph base.length (iteratorBuildStack receiver []) []
  refine ⟨after, finalLayout, ?_⟩
  exact (entered.trans (by simpa only [ExitResult, relocateValue_id] using failed)).finishes (.done rfl)

end solvers.«79».source

/-! Both consumers advance from one actual caller-method trace and retain their ordering invariant. -/
namespace solvers.«79».source
open FormalSW.PyAst HeapCalls Mutable.Relocation
open solvers.«79».graph
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

def ConsumersAgree (state : Storage) (checked : List String) : Prop :=
  ∀ rest, processedOutcome state.graphData rest = p79.validatedOutcome checked rest

theorem ConsumersAgree.empty : ConsumersAgree emptyStorage [] :=
  processed_outcome emptyStorage.graphData [] RepresentsTriples.empty

theorem ConsumersAgree.extend (state next : Storage) (checked : List String) (s : String)
    (agree : ConsumersAgree state checked) (valid : ValidAttempt s)
    (data : next.graphData = state.graphData.extend s) : ConsumersAgree next (checked ++ [s]) := by
  intro rest
  have matched := agree (s :: rest)
  have accepted : s.length = 3 ∧ Strings.isDigit s = true := valid
  simpa only [processedOutcome, GraphData.process, if_pos valid,
    p79.validatedOutcome, if_pos accepted, data] using matched

/-- The consumers execute their own source bodies; the method trace fixes their shared choice counter. -/
theorem paired_valid_iterator_step (context : Module)
    (solverExtension : Extends context graph.program) (referenceExtension : Extends context p79.program)
    (source : IteratorCalls.Relocation.ModuleSource context) (choices : ChoiceSchedule)
    (leftAnchors rightAnchors used finalUsed : Nat) (logical yielded left right : Mutable.Heap)
    (leftMap rightMap : Nat → Nat) (iterator : Mutable.Value) (operation : Operation)
    (roots : Roots) (state : Storage) (checked : List String) (checkedAddress : Nat)
    (leftEnv rightEnv : HeapCalls.Env) (s : String) (valid : ValidAttempt s)
    (leftLayout : ExecutionLayout logical left leftMap)
    (rightLayout : PrivateLayout logical right rightMap checkedAddress (.list (checked.map p79.text)))
    (inside : iterator.edge.Within logical.length)
    (isObject : IteratorCalls.isInstance logical iterator = true)
    (bound : ProtocolCalls.nextMethod context iterator logical = some (.ok (some operation), logical))
    (executed : IteratorCalls.Reaches context logical.length choices []
      (.running ⟨⟨.operation operation [], [], logical⟩, used⟩ [])
      (.completed (.ok ⟨.ok (graph.text s), [], yielded, logical.length⟩) finalUsed []))
    (noGrowth : yielded.length = logical.length)
    (represented : Represents roots state left)
    (saved : ProtectedCells (callerImage logical leftMap) roots state left)
    (bindings : Bindings roots leftEnv) (getChecked : rightEnv "checked" = some (.ref checkedAddress))
    (agree : ConsumersAgree state checked)
    (leftFollowing rightFollowing : List Stmt) (leftStack rightStack : List HeapCalls.Frame)
    (leftResumes rightResumes : List IteratorCalls.Resume) :
    ∃ (leftAfter rightYielded : Mutable.Heap) (next : Storage) (nextEnv : HeapCalls.Env)
      (nextLeftMap nextRightMap : Nat → Nat),
      IteratorCalls.Reaches context leftAnchors choices []
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue leftMap iterator)) 0 :: leftFollowing)
          leftEnv, leftStack, left⟩, used⟩ leftResumes)
        (.running ⟨⟨.block (sequenceLoop (sequenceSource (relocateValue leftMap iterator)) 0 :: leftFollowing)
          nextEnv, leftStack, leftAfter⟩, finalUsed⟩ leftResumes) ∧
      IteratorCalls.Reaches context rightAnchors choices []
        (.running ⟨⟨.block (p79.validationLoop (sequenceSource (relocateValue rightMap iterator)) 0 :: rightFollowing)
          rightEnv, rightStack, right⟩, used⟩ rightResumes)
        (.running ⟨⟨.block (p79.validationLoop (sequenceSource (relocateValue rightMap iterator)) 0 :: rightFollowing)
          (setLocal rightEnv "attempt" (p79.text s)), rightStack,
          rightYielded.set checkedAddress (.list ((checked ++ [s]).map p79.text))⟩, finalUsed⟩ rightResumes) ∧
      (∀ index < logical.length, nextLeftMap index = leftMap index) ∧
      (∀ index < logical.length, nextRightMap index = rightMap index) ∧
      Represents roots next leftAfter ∧ ProtectedCells (callerImage logical leftMap) roots next leftAfter ∧
      Bindings roots nextEnv ∧ ExecutionLayout yielded leftAfter nextLeftMap ∧
      PrivateLayout yielded (rightYielded.set checkedAddress (.list ((checked ++ [s]).map p79.text)))
        nextRightMap checkedAddress (.list ((checked ++ [s]).map p79.text)) ∧
      ConsumersAgree next (checked ++ [s]) := by
  obtain ⟨leftAfter, next, nextEnv, nextLeftMap, leftAgrees, leftRun,
    nextRep, nextSaved, nextBindings, nextLayout, data⟩ :=
    iterator_valid_from_logical context solverExtension source choices leftAnchors used finalUsed
      logical left yielded leftMap iterator operation roots state leftEnv s valid leftLayout inside
      isObject bound executed noGrowth represented saved bindings leftFollowing leftStack leftResumes
  obtain ⟨rightYielded, nextRightMap, rightAgrees, rightRun, nextPrivate⟩ :=
    p79.iterator_accepted_from_logical context referenceExtension source choices rightAnchors used finalUsed
      logical right yielded rightMap iterator operation checkedAddress checked rightEnv s valid.1 valid.2
      rightLayout inside isObject bound executed noGrowth getChecked rightFollowing rightStack rightResumes
  exact ⟨leftAfter, rightYielded, next, nextEnv, nextLeftMap, nextRightMap,
    leftRun, rightRun, leftAgrees, rightAgrees, nextRep, nextSaved, nextBindings, nextLayout, nextPrivate,
    agree.extend state next checked s valid data⟩

end solvers.«79».source
