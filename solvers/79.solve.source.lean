import solvers.«79.build.source»
import solvers.«79.topo.result.source»

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
