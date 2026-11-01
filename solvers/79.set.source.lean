import solvers.«79.equivalence.source»
import lang.py.FormalSW.SetChoiceCalls

/-! The graph-building source loop follows the actual finite-set choice schedule. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def setInputLoop (address : Nat) (original remaining : List String) : Stmt :=
  .setStepLoop "s" address (original.map Constant.string) (remaining.map Constant.string) attemptBody []

def SetProcessResult (choices : ChoiceSchedule) (used : Nat) (caller : Mutable.Heap)
    (address : Nat) (original remaining : List String) (roots : Roots) (heap : Mutable.Heap)
    (env : HeapCalls.Env) : Except ExceptionObject GraphData → Prop
  | .ok data => ∃ after next nextEnv,
      (∀ following anchors stack, SetCalls.Reaches choices program anchors
        ⟨⟨.block (setInputLoop address original remaining :: following) env, stack, heap⟩, used⟩
        ⟨⟨.block following nextEnv, stack, after⟩, used + remaining.length⟩) ∧
      Represents roots next after ∧ CallerPreserved caller roots next after ∧
      Bindings roots nextEnv ∧ next.graphData = data
  | .error error => ∃ after, caller <+: after ∧ ∀ following anchors stack,
      Finishes (SetCalls.step program anchors choices)
        ⟨⟨.block (setInputLoop address original remaining :: following) env, stack, heap⟩, used⟩
        (.ok ⟨.error error, [], after, anchors⟩)

theorem process_set (choices : ChoiceSchedule) (used : Nat) (caller : Mutable.Heap)
    (address : Nat) (original remaining : List String)
    (found : caller[address]? = some (.set (original.map Constant.string)))
    (roots : Roots) (state : Storage) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (represented : Represents roots state heap) (saved : CallerPreserved caller roots state heap)
    (bound : Bindings roots env) :
    SetProcessResult choices used caller address original remaining roots heap env
      (state.graphData.process (SetCalls.selectionOrder choices used remaining)) := by
  have retained : heap[address]? = some (.set (original.map Constant.string)) := by
    obtain ⟨suffix, same⟩ := saved.unchanged
    rw [← same, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
    exact found
  cases remaining with
  | nil =>
      simp only [SetCalls.selectionOrder, GraphData.process]
      refine ⟨heap, state, env, ?_, represented, saved, bound, rfl⟩
      intro following anchors stack
      exact .next (by simp [setInputLoop, SetCalls.step, step, SetCalls.iteratorStep, retained]) .refl
  | cons head tail =>
      let index := choices used % (head :: tail).length
      have bounded : index < (head :: tail).length := Nat.mod_lt _ (by simp)
      let s := (head :: tail)[index]
      let rest := (head :: tail).eraseIdx index
      have selected : ((head :: tail).map Constant.string)[index]? = some (.string s) := by
        rw [List.getElem?_map, List.getElem?_eq_getElem bounded]
        rfl
      have control : Calls.loopControl attemptBody = false := rfl
      have entered (following : List Stmt) (anchors : Nat) (stack : List Frame) :
          SetCalls.step program anchors choices
            ⟨⟨.block (setInputLoop address original (head :: tail) :: following) env, stack, heap⟩, used⟩ =
          .next ⟨⟨.block (attemptBody ++ (setInputLoop address original rest :: following))
            (setLocal env "s" (text s)), stack, heap⟩, used + 1⟩ := by
        simpa only [setInputLoop, Calls.loopContinuation, control, Bool.false_eq_true,
          ↓reduceIte, List.eraseIdx_map, List.singleton_append, rest, text] using
          SetCalls.select_item_mod program choices anchors used address index heap
            (original.map Constant.string) ((head :: tail).map Constant.string) (.string s)
            env "s" attemptBody [] following stack retained selected (by simp [index])
      rw [SetCalls.selectionOrder]
      change SetProcessResult _ _ _ _ _ _ _ _ _
        (state.graphData.process (s :: SetCalls.selectionOrder choices (used + 1) rest))
      rw [GraphData.process]
      by_cases valid : ValidAttempt s
      · rw [if_pos valid]
        obtain ⟨middle, current, middleEnv, iteration, currentRep, currentSaved, currentBound, data⟩ :=
          valid_iteration caller roots state heap env represented saved bound s valid
        have recursive := process_set choices (used + 1) caller address original rest found roots
          current middle middleEnv currentRep currentSaved currentBound
        rw [← data]
        cases outcome : current.graphData.process (SetCalls.selectionOrder choices (used + 1) rest) with
        | ok result =>
            rw [outcome] at recursive
            obtain ⟨after, next, nextEnv, continued, resultRep, framed, bindings, contents⟩ := recursive
            refine ⟨after, next, nextEnv, ?_, resultRep, framed, bindings, contents⟩
            intro following anchors stack
            apply SetCalls.Reaches.next (entered following anchors stack)
            apply (SetCalls.Reaches.of_heap
              (iteration (setInputLoop address original rest :: following) anchors stack)
              choices (used + 1)).trans
            have length : rest.length = tail.length := by
              simp [rest, List.length_eraseIdx_of_lt bounded]
            simpa only [length, List.length_cons, Nat.add_assoc, Nat.add_comm 1] using
              continued following anchors stack
        | error error =>
            rw [outcome] at recursive
            obtain ⟨after, framed, failed⟩ := recursive
            refine ⟨after, framed, ?_⟩
            intro following anchors stack
            apply Finishes.next (entered following anchors stack)
            exact (SetCalls.Reaches.of_heap
              (iteration (setInputLoop address original rest :: following) anchors stack)
              choices (used + 1)).finishes (failed following anchors stack)
      · rw [if_neg valid]
        refine ⟨heap, saved.unchanged, ?_⟩
        intro following anchors stack
        apply Finishes.next (entered following anchors stack)
        apply SetCalls.finishes_of_heap (invalid_attempt (setLocal env "s" (text s)) heap s
          (by simp [setLocal]) ?_ _ anchors stack)
        by_cases length : s.length = 3
        · exact Or.inr (by simpa [ValidAttempt, length] using valid)
        · exact Or.inl length
termination_by remaining.length
decreasing_by
  rw [List.length_eraseIdx_of_lt bounded]
  simp

end solvers.«79».graph

namespace solvers.«79».source
open FormalSW.PyAst HeapCalls
open solvers.«79».graph solvers.«79».topo
open ProjectEulerStatements.TopologicalOrder
set_option Elab.async false

/-- The complete solver consumes the input set, then sorts the constructed graph under the remaining schedule. -/
theorem solve_set_finishes (choices : ChoiceSchedule) (base : Mutable.Heap)
    (address : Nat) (attempts : List String)
    (found : base[address]? = some (.set (attempts.map Constant.string))) :
    ∃ after, ((SetCalls.call program "solve" [.ref address] base).run choices).Finishes
      ⟨solveOutcome (SetCalls.selectionOrder choices 0 attempts), [], after, base.length⟩ ∧
      base <+: after := by
  let env := functionEnv solveArgs solveBody [.ref address] []
  let stack : List Frame := [.unpack ["nodes", "adj", "indeg"] returnTopo env]
  let initialized := initialEnv (buildEnv (.ref address)) base
  have retained : (initialHeap base)[address]? = some (.set (attempts.map Constant.string)) := by
    simpa only [initialHeap, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1] using found
  have entered : SetCalls.Reaches choices program base.length
      ⟨⟨.operation (.call "solve") [.ref address], [], base⟩, 0⟩
      ⟨⟨.block (setInputLoop address attempts attempts :: returnGraph) initialized,
        stack, initialHeap base⟩, 0⟩ := by
    apply (SetCalls.Reaches.of_heap
      (show HeapCalls.Reaches program base.length
        ⟨.operation (.call "solve") [.ref address], [], base⟩ ⟨.block solveCode env, [], base⟩
        from .next rfl .refl) choices 0).trans
    rw [solve_source]
    apply SetCalls.Reaches.next rfl
    apply SetCalls.Reaches.next rfl
    apply (SetCalls.Reaches.of_heap (Evaluates.arguments_reach
      (.cons (Evaluates.lookup (by rfl)) .nil) (.call "build_graph") [] base.length _) choices 0).trans
    apply SetCalls.Reaches.next (following := ⟨⟨.block buildCode (buildEnv (.ref address)), stack, base⟩, 0⟩) rfl
    rw [build_parts]
    apply (SetCalls.Reaches.of_heap
      (initial_block (buildEnv (.ref address)) base _ base.length stack) choices 0).trans
    exact SetCalls.for_set_name program choices base.length 0 address (initialHeap base)
      (attempts.map Constant.string) initialized "s" "attempts" attemptBody [] returnGraph stack (by rfl) retained
  have built := process_set choices 0 base address attempts attempts found (initialRoots base)
    emptyStorage (initialHeap base) initialized (initial_represents base) (initial_caller base)
    (initial_bindings (buildEnv (.ref address)) base)
  cases processed : emptyStorage.graphData.process (SetCalls.selectionOrder choices 0 attempts) with
  | error error =>
      rw [processed] at built
      obtain ⟨after, saved, failed⟩ := built
      refine ⟨after, ?_, saved⟩
      change Finishes (SetCalls.step program base.length choices)
        ⟨⟨.operation (.call "solve") [.ref address], [], base⟩, 0⟩ _
      simpa only [solveOutcome, processed] using entered.finishes (failed returnGraph base.length stack)
  | ok data =>
      rw [processed] at built
      obtain ⟨middle, state, finalEnv, loop, represented, saved, bindings, contents⟩ := built
      let heap := middle ++ [.tuple (graphValues (initialRoots base))]
      let bound := assignNames env ["nodes", "adj", "indeg"] (graphValues (initialRoots base))
      have returned : SetCalls.Reaches choices program base.length
          ⟨⟨.block returnGraph finalEnv, stack, middle⟩, attempts.length⟩
          ⟨⟨.block returnTopo bound, [], heap⟩, attempts.length⟩ := by
        apply (SetCalls.Reaches.of_heap (return_graph (initialRoots base) finalEnv middle bindings
          base.length stack) choices attempts.length).trans
        apply SetCalls.Reaches.of_heap
        exact .next (by simp [step, run, stack, Mutable.sequence, heap, bound, graphValues]) .refl
      obtain ⟨after, sorted, sortedSaved⟩ := return_topo choices heap (initialRoots base) state
        (represented.append _ _ _ _) bound (by simp [bound, assignNames, graphValues, setLocal])
        (by simp [bound, assignNames, graphValues, setLocal])
        (by simp [bound, assignNames, graphValues, setLocal]) base.length attempts.length []
      refine ⟨after, ?_, saved.unchanged.trans ((List.prefix_append _ _).trans sortedSaved)⟩
      change Finishes (SetCalls.step program base.length choices)
        ⟨⟨.operation (.call "solve") [.ref address], [], base⟩, 0⟩ _
      apply entered.finishes
      apply (loop returnGraph base.length stack).finishes
      simp only [Nat.zero_add]
      apply returned.finishes
      have completed := ContextResult.finishes _ _ _ _ _ _ _ sorted
      simpa only [solveOutcome, processed, ← contents] using completed

end solvers.«79».source
