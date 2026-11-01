import solvers.«79.solve.source»
import data.«project-euler-statements».data.py.project_euler_statements.«p79.choice.source»
import lang.py.FormalSW.SetChoiceCalls
import lang.py.FormalSW.IteratorLifting

/-! Independent set choices can select different validation errors in the actual P79 calls. -/
namespace solvers.«79».choice
open FormalSW.PyAst HeapCalls
open data.«project-euler-statements».data.py.project_euler_statements
set_option Elab.async false

/-- Graph initialization happens before the invalid attempt, and its allocations remain in the error completion. -/
theorem build_set_failure (heap : Mutable.Heap) (address index : Nat) (items : List Constant)
    (s : String) (choices : ChoiceSchedule) (anchors used : Nat) (stack : List HeapCalls.Frame)
    (found : heap[address]? = some (.set items)) (selected : items[index]? = some (.string s))
    (invalid : s.length ≠ 3 ∨ Strings.isDigit s = false) (choice : choices used = index) :
    Finishes (SetCalls.step graph.program anchors choices)
      ⟨⟨.block graph.buildCode (graph.buildEnv (.ref address)), stack, heap⟩, used⟩
      (.ok ⟨.error (graph.invalidException s), [], graph.initialHeap heap, anchors⟩) := by
  let env := graph.initialEnv (graph.buildEnv (.ref address)) heap
  have saved : (graph.initialHeap heap)[address]? = some (.set items) := by
    simpa only [graph.initialHeap, List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1] using found
  rw [graph.build_parts]
  apply (SetCalls.Reaches.of_heap (graph.initial_block (graph.buildEnv (.ref address)) heap _ anchors stack) choices used).finishes
  apply (SetCalls.for_set_name graph.program choices anchors used address (graph.initialHeap heap) items env "s" "attempts"
    graph.attemptBody [] graph.returnGraph stack (by rfl) saved).finishes
  apply Finishes.next (SetCalls.select_item graph.program choices anchors used address index (graph.initialHeap heap)
    items items (.string s) env "s" graph.attemptBody [] graph.returnGraph stack saved selected choice)
  exact SetCalls.finishes_of_heap (graph.invalid_attempt (setLocal env "s" (.scalar (.string s))) (graph.initialHeap heap)
    s (by simp [setLocal, graph.text]) invalid _ anchors stack) choices (used + 1)

theorem solve_set_failure (heap : Mutable.Heap) (address index : Nat) (items : List Constant)
    (s : String) (choices : ChoiceSchedule)
    (found : heap[address]? = some (.set items)) (selected : items[index]? = some (.string s))
    (invalid : s.length ≠ 3 ∨ Strings.isDigit s = false) (choice : choices 0 = index) :
    ((SetCalls.call graph.program "solve" [.ref address] heap).run choices).Finishes
      ⟨.error (graph.invalidException s), [], graph.initialHeap heap, heap.length⟩ := by
  let env := functionEnv source.solveArgs source.solveBody [.ref address] []
  change Finishes (SetCalls.step graph.program heap.length choices)
    ⟨⟨.operation (.call "solve") [.ref address], [], heap⟩, 0⟩ _
  apply SetCalls.Reaches.finishes (SetCalls.Reaches.of_heap
    (show HeapCalls.Reaches graph.program heap.length
      ⟨.operation (.call "solve") [.ref address], [], heap⟩ ⟨.block source.solveCode env, [], heap⟩ from .next rfl .refl)
    choices 0)
  rw [source.solve_source]
  refine Finishes.next (step := SetCalls.step graph.program heap.length choices) rfl ?_
  refine Finishes.next (step := SetCalls.step graph.program heap.length choices) rfl ?_
  apply (SetCalls.Reaches.of_heap (Evaluates.arguments_reach (.cons (Evaluates.lookup (by rfl)) .nil)
    (.call "build_graph") [] heap.length _) choices 0).finishes
  exact .next rfl (build_set_failure heap address index items s choices heap.length 0 _ found selected invalid choice)


/-- The same set input can produce incompatible exact errors under independent schedules. -/
theorem not_guaranteed_equiv (heap : Mutable.Heap) (address leftIndex rightIndex : Nat)
    (items : List Constant) (left right : String)
    (found : heap[address]? = some (.set items))
    (leftSelected : items[leftIndex]? = some (.string left))
    (rightSelected : items[rightIndex]? = some (.string right))
    (leftInvalid : left.length ≠ 3 ∨ Strings.isDigit left = false)
    (rightInvalid : right.length ≠ 3 ∨ Strings.isDigit right = false)
    (different : graph.invalidException left ≠ p79.invalidException right) :
    ¬ (IteratorCalls.call graph.program "solve" [.ref address] heap).Equiv
      (IteratorCalls.call p79.program "naive" [.ref address] heap) := by
  have first : ((IteratorCalls.call graph.program "solve" [.ref address] heap).run (fun _ => leftIndex)).Finishes
      ⟨.error (graph.invalidException left), [], graph.initialHeap heap, heap.length⟩ :=
    IteratorCalls.finishes_of_set (solve_set_failure heap address leftIndex items left (fun _ => leftIndex)
      found leftSelected leftInvalid rfl) (by intro completion equal; cases equal; simp [graph.invalidException])
  have second : ((IteratorCalls.call p79.program "naive" [.ref address] heap).run (fun _ => rightIndex)).Finishes
      ⟨.error (p79.invalidException right), [], heap ++ [.list []], heap.length⟩ :=
    IteratorCalls.finishes_of_set (p79.naive_set_failure heap address rightIndex items right (fun _ => rightIndex)
      found rightSelected rightInvalid rfl) (by intro completion equal; cases equal; simp [p79.invalidException])
  apply Mutable.CallFamily.not_equiv_of_finishes first second
  intro equivalent
  rcases equivalent with same | ⟨_, _, same, _⟩
  · exact different (Except.error.inj (congrArg Mutable.Completion.outcome same))
  · exact different same

def exampleHeap : Mutable.Heap := [.set [.string "", .string "x"]]

theorem example_valid : exampleHeap.Valid := by
  simp [exampleHeap, Mutable.Heap.Valid, ScalarSet.Valid, scalarCompare, evalCmpOp]

theorem example_not_guaranteed_equiv :
    ¬ (IteratorCalls.call graph.program "solve" [.ref 0] exampleHeap).Equiv
      (IteratorCalls.call p79.program "naive" [.ref 0] exampleHeap) := by
  apply not_guaranteed_equiv exampleHeap 0 0 1 [.string "", .string "x"] "" "x" rfl rfl rfl
    (Or.inl (by simp)) (Or.inl (by change (1 : Nat) ≠ 3; omega))
  intro same
  have args := congrArg ExceptionObject.args same
  change [Constant.string "Invalid attempt: ''"] = [Constant.string "Invalid attempt: 'x'"] at args
  simp at args

end solvers.«79».choice
