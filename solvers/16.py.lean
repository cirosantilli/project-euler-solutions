import «.formalsw».solvers.«16.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p16.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P16

namespace solvers.«16»
open FormalSW.PyAst
private abbrev program := Solvers.P16Ast.module

private theorem parse_digit (c : Char) (digit : c.isDigit) :
    Calls.integerConversion [.string (String.singleton c)] =
      some (.ok (.int ((c.toNat - '0'.toNat : Nat) : Int))) := by
  change Calls.integerConversion [.string (String.ofList [c])] = _
  rw [Calls.integerConversion, FormalSW.Py.Builtin.Int.parseText_digits [c] (by simp)
    (by intro d member; simp only [List.mem_singleton] at member; subst d; exact digit)]
  simp [Calls.parsedInteger, Nat.ofDigitChars]

private theorem fold_digits_cast (characters : List Char) (initial : Nat) :
    characters.foldl (fun (total : Int) c => total + ((c.toNat - '0'.toNat : Nat) : Int)) initial =
      ((characters.foldl (fun (total : Nat) (c : Char) => (total + (c.toNat - '0'.toNat) : Nat)) initial : Nat) : Int) := by
  induction characters generalizing initial with
  | nil => rfl
  | cons head tail ih =>
      simpa only [List.foldl_cons, Int.natCast_add] using ih (initial + (head.toNat - '0'.toNat))

set_option maxHeartbeats 1000000 in
/-- Parsing and summing the decimal characters computes the mathematical digit sum. -/
theorem digit_sum.result (n : Nat) (heap : Mutable.Heap) :
    HeapCalls.ReturnsValue program "digit_sum" [.scalar (.int n)] heap
      (.scalar (.int (ProjectEulerSolutions.P16.digitSum n))) heap := by
  let characters := (toString n).toList
  have total : characters.attach.foldl
      (fun (total : Int) c => total + ((c.val.toNat - '0'.toNat : Nat) : Int)) 0 =
      (ProjectEulerSolutions.P16.digitSum n : Int) := by
    rw [List.foldl_attach (f := fun (total : Int) (c : Char) => total + ((c.toNat - '0'.toNat : Nat) : Int))]
    exact fold_digits_cast characters 0
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.return_value
  rw [← total]
  apply HeapCalls.Evaluates.sum_generator_sequence
    (items := characters.map (fun c => .scalar (.string (String.singleton c)))) (by intros; simp)
    (HeapCalls.Evaluates.builtin (name := .str) (values := [.scalar (.int n)])
      (value := .scalar (.string (toString n))) (.cons (HeapCalls.Evaluates.lookup rfl) .nil) rfl)
    (by simp [Mutable.sequence, characters]) HeapCalls.Evaluates.constant rfl
  apply HeapCalls.sum_map_loop (values := characters.attach)
    (encode := fun c => .scalar (.string (String.singleton c.val)))
    (number := fun c => ((c.val.toNat - '0'.toNat : Nat) : Int)) characters.attach (by decide)
  · simp [HeapCalls.sequenceValue, HeapCalls.sequenceSource, Mutable.sequence, characters]
  · intro env c lookup
    apply HeapCalls.Evaluates.builtin (.cons (HeapCalls.Evaluates.lookup lookup) .nil)
    change HeapCalls.scalarResult (Calls.integerConversion [.string (String.singleton c.val)]) heap = _
    have member : c.val ∈ Nat.toDigits 10 n := by simpa [characters] using c.property
    rw [parse_digit c.val (Nat.isDigit_of_mem_toDigits (by decide) (by decide) member)]
    rfl
  · rfl
  · simp [HeapCalls.setLocal]

theorem solve.result (n : Nat) (heap : Mutable.Heap) :
    HeapCalls.ReturnsValue program "solve" [.scalar (.int n)] heap
      (.scalar (.int (ProjectEulerSolutions.P16.solve n))) heap := by
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.return_value
  apply HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.binary (finalHeap := heap)
    (value := .scalar (.int ((2 ^ n : Nat) : Int))) HeapCalls.Evaluates.constant
    (HeapCalls.Evaluates.lookup rfl)
    (by simp [Mutable.binary, scalarBinary, show ¬(n : Int) < 0 by omega])) .nil)
  exact digit_sum.result (2 ^ n) heap

theorem solve.«Equiv(naive)» (n : Int) (condition : Calls.Condition program
    (fun name => if name = "n" then some (.int n) else none) Solvers.P16Ast.requires_solve_0) :
    (HeapCalls.scalarFunction program "solve" [.int n]).Equiv
      (HeapCalls.scalarFunction P16Ast.module "naive" [.int n]) := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
  lift n to Nat using nonnegative
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p16.naive.result n []
  rw [ProjectEulerSolutions.P16.equiv] at reference
  have optimized := HeapCalls.ReturnsValue.scalar_finishes (values := [.int n]) (solve.result n [])
  have referenceRun := HeapCalls.ReturnsValue.scalar_finishes (values := [.int n]) reference
  intro outcome
  constructor
  · intro run
    rw [run.unique optimized]
    exact referenceRun
  · intro run
    rw [run.unique referenceRun]
    exact optimized

theorem solve.«Supported» :
    (∀ (n : Int), Calls.Condition program (fun name => if name = "n" then some (.int n) else none)
      Solvers.P16Ast.requires_solve_0 → (HeapCalls.scalarFunction program "solve" [.int n]).Supported) ∧
    (∀ (n : Int), Calls.Condition program (fun name => if name = "n" then some (.int n) else none)
      Solvers.P16Ast.requires_solve_0 → (HeapCalls.scalarFunction P16Ast.module "naive" [.int n]).Supported) := by
  constructor
  · intro n condition
    have nonnegative : 0 ≤ n := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
    lift n to Nat using nonnegative
    exact (solve.result n []).scalar_returns.supported
  · intro n condition
    have nonnegative : 0 ≤ n := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
    lift n to Nat using nonnegative
    exact (data.«project-euler-statements».data.py.project_euler_statements.p16.naive.result n []).scalar_returns.supported

end solvers.«16»
