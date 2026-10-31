import solvers.«79.iterator.frame.source»
import solvers.«79.sequence.source»

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
