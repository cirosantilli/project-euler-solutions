import solvers.«92.py.lean.d».Execution
import data.«project-euler-statements».data.py.project_euler_statements.«p92.py»

namespace solvers.«92».next_term
open FormalSW.PyAst
set_option Elab.async false

private abbrev inputCondition (n : Int) :=
  Calls.Condition program (fun name => if name = "n" then some (.int n) else none)
    Solvers.P92Ast.requires_next_term_0

theorem «Returns» (n : Int) (valid : inputCondition n) :
    (HeapCalls.scalarFunction program "next_term" [.int n]).Returns := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare valid rfl
  lift n to Nat using nonnegative
  obtain ⟨_, returned⟩ := result n []
  exact returned.scalar_returns (values := [.int n])

theorem «Equiv(reference_next_term)» (n : Int) (valid : inputCondition n) :
    (HeapCalls.scalarFunction program "next_term" [.int n]).Equiv
      (HeapCalls.scalarFunction P92Ast.module "next_term" [.int n]) := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare valid rfl
  lift n to Nat using nonnegative
  obtain ⟨_, optimized⟩ := result n []
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p92.next_term.result n []
  have left := optimized.scalar_finishes (values := [.int n])
  have right := reference.scalar_finishes (values := [.int n])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ n : Int, inputCondition n → (HeapCalls.scalarFunction program "next_term" [.int n]).Supported) ∧
    (∀ n : Int, inputCondition n → (HeapCalls.scalarFunction P92Ast.module "next_term" [.int n]).Supported) := by
  constructor
  · intro n valid
    exact («Returns» n valid).supported
  · intro n valid
    have nonnegative : 0 ≤ n := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare valid rfl
    lift n to Nat using nonnegative
    exact ((data.«project-euler-statements».data.py.project_euler_statements.p92.next_term.result n []).scalar_returns
      (values := [.int n])).supported

end solvers.«92».next_term

namespace solvers.«92»
open FormalSW.PyAst ProjectEulerSolutions.DigitSquareChains

theorem terminal_source (start cache : Nat) (entries : store.Entries)
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    store.Found heap cache entries → Memo.Valid (store.lookup entries) →
      terminal.Verified start cache entries heap globals :=
  terminal.verified start cache entries heap globals

theorem distribution_source (bound : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    distribution.Verified bound heap globals :=
  distribution.verified bound heap globals

end solvers.«92»

namespace solvers.«92».solve
open FormalSW.PyAst ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

/-- The actual optimized source returns the count for every signed input. -/
theorem finishes (limit : Int) :
    Finishes (HeapMemoCalls.scalarFunction program "solve" [.int limit]).step
      (HeapMemoCalls.scalarFunction program "solve" [.int limit]).initial (.ok (.int (count limit))) := by
  have run := (result limit [] []).consequence
    (post := fun value _ => value = .scalar (.int (count limit)) ∧ True)
    (by intro value after same; exact ⟨same, trivial⟩)
  obtain ⟨_, _, finished⟩ := run.scalar_finishes (values := [.int limit])
  exact finished

theorem «Returns» (limit_exclusive : Int) :
    (HeapMemoCalls.scalarFunction program "solve" [.int limit_exclusive]).Returns :=
  ⟨_, finishes limit_exclusive⟩

theorem «Equiv(naive)» (limit_exclusive : Int) :
    (HeapMemoCalls.scalarFunction program "solve" [.int limit_exclusive]).Equiv
      (HeapCalls.scalarFunction P92Ast.module "naive" [.int limit_exclusive]) := by
  obtain ⟨_, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p92.naive.result limit_exclusive []
  have left := finishes limit_exclusive
  have right := reference.scalar_finishes (values := [.int limit_exclusive])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique left]
    exact right
  · intro finished
    rw [finished.unique right]
    exact left

theorem «Supported» :
    (∀ limit : Int, (HeapMemoCalls.scalarFunction program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, (HeapCalls.scalarFunction P92Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit
    exact («Returns» limit).supported
  · intro limit
    exact (data.«project-euler-statements».data.py.project_euler_statements.p92.naive.«Returns» limit).supported

end solvers.«92».solve
