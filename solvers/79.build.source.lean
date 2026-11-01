import solvers.«79.process.source»

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
