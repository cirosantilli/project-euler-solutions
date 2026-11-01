import solvers.«79.topo.loop.source»

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

theorem finish (base : Mutable.Heap) (nodes output : List String) (address : Nat)
    (found : base[address]? = some (.set (stringSet nodes)))
    (degrees : List (Constant × Mutable.Value)) (env : HeapCalls.Env)
    (getOutput : env "out" = some (.ref (base.length + 1)))
    (getNodes : env "nodes" = some (.ref address)) (anchors : Nat) :
    Finishes (HeapCalls.step program anchors)
      ⟨.block finishCode env, [], workspace base [] output degrees⟩
      (.ok ⟨resultOutcome (if output.length = nodes.length then some output else none), [],
        workspace base [] output degrees, anchors⟩) := by
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
  refine Finishes.next rfl ((tested anchors _).finishes ?_)
  by_cases complete : output.length = nodes.length
  · simp only [complete, ne_eq, not_true_eq_false, decide_false, if_true, resultOutcome]
    refine Finishes.next (following :=
      ⟨.block [finishCode[1]] env, [], workspace base [] output degrees⟩)
      (by simp [HeapCalls.step, run, Mutable.truth, scalarTruth, finishCode]) ?_
    have joined : BlockReturns program env [finishCode[1]]
        (workspace base [] output degrees) (text (String.intercalate "" output))
        (workspace base [] output degrees) := by
      apply BlockReturns.return_value
      apply Evaluates.join Evaluates.constant (.cons (Evaluates.lookup getOutput) .nil)
      · exact primitive_join_list _ (base.length + 1) "" output (by simp [workspace, values])
      · intros; contradiction
    exact (joined anchors []).finishes (.done rfl)
  · simp only [complete, ne_eq, not_false_eq_true, decide_true, if_false, resultOutcome]
    refine Finishes.next (following :=
      ⟨.block (cycleBody ++ [finishCode[1]]) env, [], workspace base [] output degrees⟩)
      (by simp [HeapCalls.step, run, Mutable.truth, scalarTruth, finishCode]) ?_
    refine Finishes.next (following := argumentsState (.raiseException "ValueError") env []
      [.constant (.string cycleMessage)] [] (workspace base [] output degrees)) rfl ?_
    apply (Evaluates.arguments_reach
      (.cons (Evaluates.constant (module := program) (env := env)) .nil)
      (.raiseException "ValueError") [] anchors []).finishes
    exact .done (by
      simp [HeapCalls.step, result, run, primitive, done, cycleException, protectGenerator,
        Calls.generatorException])

/-- All traversal schedules produce the abstract ordering result and preserve the input graph. -/
theorem body_result (choices : ChoiceSchedule) (base : Mutable.Heap)
    (roots : Roots) (state : Storage) (represented : Represents roots state base)
    (env : HeapCalls.Env) (getNodes : env "nodes" = some (.ref roots.nodes))
    (getDegrees : env "indeg" = some (.ref roots.degrees))
    (getAdjacency : env "adj" = some (.ref roots.adjacency)) (anchors used : Nat) :
    ∃ after, Finishes (SetCalls.step program anchors choices)
      ⟨⟨.block topoCode env, [], base⟩, used⟩
      (.ok ⟨resultOutcome (order state.vertices state.edges), [], after, anchors⟩) ∧
      base <+: after := by
  obtain ⟨queue, preparedUsed, initialized, contents, ordered⟩ :=
    initialize_topo choices roots state base represented env getNodes getDegrees
      (loop :: finishCode) anchors [] used
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
      finishCode anchors [] preparedUsed
  have outputBinding : finalEnv "out" = some (.ref (base.length + 1)) := by
    rw [preserved "out" (by decide) (by decide)]
    simp [prepared, preparedEnv, setLocal]
  have nodesBinding : finalEnv "nodes" = some (.ref roots.nodes) := by
    rw [preserved "nodes" (by decide) (by decide)]
    simpa [prepared, preparedEnv, setLocal] using getNodes
  have finished := finish base state.vertices emitted roots.nodes represented.nodesFound
    degrees finalEnv outputBinding nodesBinding anchors
  refine ⟨workspace base [] emitted degrees, ?_, ?_⟩
  · rw [body_source]
    apply initialized.finishes
    apply executed.finishes
    rw [result]
    exact SetCalls.finishes_of_heap finished choices finalUsed
  · exact ⟨_, rfl⟩

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
