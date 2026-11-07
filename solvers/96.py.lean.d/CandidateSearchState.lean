import solvers.«96.py.lean.d».CandidateClone
import solvers.«96.py.lean.d».CandidateChoiceLoop

namespace solvers.«96».solve_cands.search
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.Sudoku reduce_cands
set_option Elab.async false
noncomputable section

structure Answer where
  address : Nat
  assignment : Assignment

def value : Option Answer → Mutable.Value
  | none => .scalar .none
  | some answer => .ref answer.address

def Correct (candidates : Candidates) (heap : Mutable.Heap) : Option Answer → Prop
  | none => ¬ ∃ assignment, Legal assignment ∧ Admits candidates assignment
  | some answer => Legal answer.assignment ∧ Admits candidates answer.assignment ∧
      View answer.address heap (fun cell => {answer.assignment cell})

/-- A complete source call may update its input array, preserving every other existing object. -/
def Executes (choices : ChoiceSchedule) (globals : Mutable.Globals) (address : Nat)
    (candidates : Candidates) (heap : Mutable.Heap) : Prop :=
  ∀ used anchors stack, ∃ finalHeap finalUsed answer, heap.length ≤ finalHeap.length ∧
    (∀ index < heap.length, address ≠ index → finalHeap[index]? = heap[index]?) ∧
    Correct candidates finalHeap answer ∧
    SetCalls.Reaches choices program anchors
      ⟨⟨.operation (.call "solve_cands") [.ref address], stack, heap⟩, used⟩
      ⟨⟨.value (value answer), stack, finalHeap⟩, finalUsed⟩ globals

abbrev statement : Stmt := .whileDo (.name "mask") bits.body []
abbrev test : Expr := .unaryOp .not (.builtinCall .identity [.name "res", .constant .none])
abbrev accept : Stmt := .ifThen test [.return (.name "res")] []

theorem source_loop : solved.definition.2.drop 4 = [statement, .return (.constant .none)] := rfl
theorem source_call : bits.body.drop 3 =
    [.assign [.name "res"] (.call (.name "solve_cands") [.name "nxt"]), accept, bits.advance] := rfl

theorem peer_frame (peers : PeerTable) (before after : Mutable.Heap) (view : peers.View before)
    (frame : ∀ index < before.length, after[index]? = before[index]?) : peers.View after :=
  ⟨(frame peers.root (List.getElem?_eq_some_iff.mp view.root).1).trans view.root,
   fun cell => (frame (peers.address cell) (List.getElem?_eq_some_iff.mp (view.sets cell)).1).trans (view.sets cell)⟩

theorem unit_frame (units : UnitTable) (before after : Mutable.Heap) (view : units.View before)
    (frame : ∀ index < before.length, after[index]? = before[index]?) : units.View after :=
  ⟨(frame units.root (List.getElem?_eq_some_iff.mp view.root).1).trans view.root,
   fun unit member => (frame unit.address (List.getElem?_eq_some_iff.mp (view.units unit member)).1).trans (view.units unit member)⟩

structure Locals (address : Nat) (cell : Cell) (digits : Finset Digit) (env : HeapCalls.Env) : Prop where
  candidates : env "cands" = some (.ref address)
  index : env "idx" = some (number (Encoding.index cell).val)
  mask : env "mask" = some (number (Masks.encode digits))

theorem Locals.set_other {address : Nat} {cell : Cell} {digits : Finset Digit} {env : HeapCalls.Env}
    (locals : Locals address cell digits env) (name : String) (newValue : Mutable.Value)
    (notCandidates : "cands" ≠ name) (notIndex : "idx" ≠ name) (notMask : "mask" ≠ name) :
    Locals address cell digits (setLocal env name newValue) := by
  constructor
  · simpa [setLocal, notCandidates] using locals.candidates
  · simpa [setLocal, notIndex] using locals.index
  · simpa [setLocal, notMask] using locals.mask

theorem mask_truth (digits : Finset Digit) (heap : Mutable.Heap) :
    Mutable.truth (number (Masks.encode digits)) heap = some (.ok (decide (digits ≠ ∅)), heap) := by
  by_cases empty : digits = ∅
  · subst digits
    simp [Mutable.truth, scalarTruth]
  · simp [Mutable.truth, scalarTruth, empty]
    exact (Masks.zero_iff digits).not.mpr empty

theorem enter (choices : ChoiceSchedule) (globals : Mutable.Globals) (address : Nat) (cell : Cell)
    (digits : Finset Digit) (nonempty : digits ≠ ∅) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (locals : Locals address cell digits env) (following : List Stmt) (used anchors : Nat) (stack : List HeapCalls.Frame) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.block (statement :: following) env, stack, heap⟩, used⟩
      ⟨⟨.block (bits.body ++ statement :: following) env, stack, heap⟩, used⟩ globals := by
  apply SetCalls.Reaches.of_heap
  apply HeapCalls.Reaches.next (following := ⟨.expr (.name "mask") env,
    .branch (bits.body ++ [Calls.loopContinuation bits.body statement]) [] following env :: stack, heap⟩) rfl
  apply (Evaluates.lookup (globals := globals) (module := program) (heap := heap) locals.mask anchors _).trans
  exact .next (by simp [step, run, mask_truth, nonempty, Calls.loopContinuation,
    show Calls.loopControl bits.body = false from rfl]) .refl

theorem finish (choices : ChoiceSchedule) (globals : Mutable.Globals) (address : Nat) (cell : Cell)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (locals : Locals address cell ∅ env)
    (used anchors : Nat) (stack : List HeapCalls.Frame) :
    SetCalls.Reaches choices program anchors
      ⟨⟨.block [statement, .return (.constant .none)] env, stack, heap⟩, used⟩
      ⟨⟨.value (.scalar .none), stack, heap⟩, used⟩ globals := by
  apply SetCalls.Reaches.of_heap
  apply HeapCalls.Reaches.next (following := ⟨.expr (.name "mask") env,
    .branch (bits.body ++ [Calls.loopContinuation bits.body statement]) [] [.return (.constant .none)] env :: stack, heap⟩) rfl
  apply (Evaluates.lookup (globals := globals) (module := program) (heap := heap) locals.mask anchors _).trans
  apply HeapCalls.Reaches.next (following := ⟨.block [.return (.constant .none)] env, stack, heap⟩)
    (by simp [step, run, Mutable.truth, scalarTruth])
  exact BlockReturns.return_value (module := program) (env := env) (globals := globals) (heap := heap)
    (rest := []) Evaluates.constant anchors stack

theorem returned_test (answer : Option Answer) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (getResult : env "res" = some (value answer)) :
    Evaluates program env test heap (.scalar (.bool answer.isSome)) heap globals := by
  have identityTest : Evaluates program env (.builtinCall .identity [.name "res", .constant .none]) heap
      (.scalar (.bool (!answer.isSome))) heap globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getResult) (.cons Evaluates.constant .nil))
    cases answer <;> rfl
  apply Evaluates.unary identityTest
  cases answer <;> rfl

theorem accept_solution (answer : Answer) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (getResult : env "res" = some (.ref answer.address)) (following : List Stmt) :
    BlockReturns program env (accept :: following) heap (.ref answer.address) heap globals := by
  apply BlockReturns.branch (flag := true) (returned_test (some answer) env heap globals getResult) rfl
  exact BlockReturns.return_value (Evaluates.lookup getResult)

theorem reject_branch (env : HeapCalls.Env) (heap : Mutable.Heap)
    (globals : Mutable.Globals) (getResult : env "res" = some (.scalar .none)) :
    BlockContinues program env [accept] heap env heap globals :=
  BlockContinues.branch (flag := false) (returned_test none env heap globals getResult) rfl .nil

end
end solvers.«96».solve_cands.search
