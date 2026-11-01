import solvers.«79.graph.source»

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
