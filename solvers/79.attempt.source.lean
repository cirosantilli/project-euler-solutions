import solvers.«79.initialize.source»

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
      Represents roots (state.withNode vertex heap.length) after := by
  obtain ⟨after, result, retained⟩ := addNode_preserves roots state heap represented vertex
  refine ⟨after, ?_, retained⟩
  apply BlockContinues.discard
  exact Evaluates.call
    (.cons (Evaluates.lookup getVertex) (.cons (Evaluates.lookup bindings.nodes)
      (.cons (Evaluates.lookup bindings.adjacency)
        (.cons (Evaluates.lookup bindings.degrees) .nil))))
    result

theorem edge_call (roots : Roots) (state : Storage) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (represented : Represents roots state heap) (bindings : Bindings roots env)
    (sourceName targetName source target : String)
    (getSource : env sourceName = some (text source))
    (getTarget : env targetName = some (text target))
    (sourceMember : source ∈ state.vertices) (targetMember : target ∈ state.vertices) :
    ∃ after, BlockContinues program env [edgeCall sourceName targetName] heap env after ∧
      Represents roots (state.withEdge source target) after := by
  obtain ⟨after, result, retained⟩ :=
    addEdge_preserves roots state heap represented source target sourceMember targetMember
  refine ⟨after, ?_, retained⟩
  apply BlockContinues.discard
  exact Evaluates.call
    (.cons (Evaluates.lookup getSource) (.cons (Evaluates.lookup getTarget)
      (.cons (Evaluates.lookup bindings.adjacency)
        (.cons (Evaluates.lookup bindings.degrees) .nil))))
    result

/-- The actual five calls retain the invariant, including repeated digits and duplicate edges. -/
theorem updates_preserve (roots : Roots) (state : Storage) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (represented : Represents roots state heap)
    (bindings : Bindings roots env)
    (a b c : String) (getA : env "a" = some (text a)) (getB : env "b" = some (text b))
    (getC : env "c" = some (text c)) :
    ∃ after next, BlockContinues program env (attemptBody.drop 2) heap env after ∧
      Represents roots next after ∧
      next.vertices = insertVertex c (insertVertex b (insertVertex a state.vertices)) ∧
      next.edges = insertEdge b c (insertEdge a b state.edges) := by
  obtain ⟨heapA, stepA, reprA⟩ := node_call roots state heap env represented bindings "a" a getA
  let stateA := state.withNode a heap.length
  obtain ⟨heapB, stepB, reprB⟩ := node_call roots stateA heapA env reprA bindings "b" b getB
  let stateB := stateA.withNode b heapA.length
  obtain ⟨heapC, stepC, reprC⟩ := node_call roots stateB heapB env reprB bindings "c" c getC
  let stateC := stateB.withNode c heapB.length
  have memberA : a ∈ stateC.vertices := by
    simp only [stateC, stateB, stateA, Storage.withNode_mem, or_true, true_or]
  have memberB : b ∈ stateC.vertices := by
    simp only [stateC, stateB, Storage.withNode_mem, or_true, true_or]
  have memberC : c ∈ stateC.vertices := by
    simp only [stateC, Storage.withNode_mem, or_true]
  obtain ⟨heapAB, stepAB, reprAB⟩ :=
    edge_call roots stateC heapC env reprC bindings "a" "b" a b getA getB memberA memberB
  obtain ⟨heapBC, stepBC, reprBC⟩ := edge_call roots (stateC.withEdge a b) heapAB env
    reprAB bindings "b" "c" b c getB getC memberB memberC
  refine ⟨heapBC, (stateC.withEdge a b).withEdge b c, ?_, reprBC, ?_, ?_⟩
  · exact stepA.append (stepB.append (stepC.append (stepAB.append stepBC)))
  · simp only [Storage.withEdge, stateC, stateB, stateA, Storage.withNode_vertices]
  · simp only [Storage.withEdge, stateC, stateB, stateA, Storage.withNode_edges]

end solvers.«79».graph
