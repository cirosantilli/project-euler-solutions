import «.formalsw».solvers.«20.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p20.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P20

namespace solvers.«20»
open FormalSW.PyAst
private abbrev program := Solvers.P20Ast.module

private def finalExpr : Expr :=
  match executableBody ((findFunctionIn program.body "factorial_digit_sum").getD ([], [])).2 with
  | [_, _, .return expression] => expression
  | _ => .constant .none

private def body : List Stmt :=
  match executableBody ((findFunctionIn program.body "factorial_digit_sum").getD ([], [])).2 with
  | [_, .forIn _ _ statements _, _] => statements
  | _ => []

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
private theorem sumDigits.result (n : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (getFact : env "fact" = some (.scalar (.int n))) :
    HeapCalls.Evaluates program env finalExpr heap
      (.scalar (.int (ProjectEulerSolutions.P20.digitSum n))) heap := by
  let characters := (toString n).toList
  have total : characters.attach.foldl
      (fun (total : Int) c => total + ((c.val.toNat - '0'.toNat : Nat) : Int)) 0 =
      (ProjectEulerSolutions.P20.digitSum n : Int) := by
    rw [List.foldl_attach (f := fun (total : Int) (c : Char) => total + ((c.toNat - '0'.toNat : Nat) : Int))]
    exact fold_digits_cast characters 0
  rw [← total]
  apply HeapCalls.Evaluates.sum_generator_sequence
    (items := characters.map (fun c => .scalar (.string (String.singleton c)))) (by intros; simp)
    (HeapCalls.Evaluates.builtin (name := .str) (values := [.scalar (.int n)])
      (value := .scalar (.string (toString n))) (.cons (HeapCalls.Evaluates.lookup getFact) .nil) rfl)
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


private theorem loop_result (n k acc : Nat) (env : Mutable.Env) (heap : Mutable.Heap)
    (getFact : env "fact" = some (.scalar (.int acc))) :
    HeapCalls.BlockReturns program env
      [.rangeStepLoop "k" k (n + 1) 1 body [], .return finalExpr] heap
      (.scalar (.int (ProjectEulerSolutions.P20.solve.loop n k acc))) heap := by
  induction measure : n + 1 - k using Nat.strong_induction_on generalizing k acc env with
  | h size ih =>
      rw [ProjectEulerSolutions.P20.solve.loop.eq_1]
      by_cases finished : k > n
      · rw [if_pos finished]
        apply HeapCalls.BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
        exact HeapCalls.BlockReturns.return_value (sumDigits.result acc env heap getFact)
      · rw [if_neg finished]
        apply HeapCalls.BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
        apply HeapCalls.BlockReturns.augAssign (afterOperation := heap) (left := .scalar (.int acc))
          (right := .scalar (.int k)) (value := .scalar (.int ((acc * k : Nat) : Int)))
          (by simpa [HeapCalls.setLocal] using getFact)
          (HeapCalls.Evaluates.lookup (by simp [HeapCalls.setLocal]))
          (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
        have next := ih (n + 1 - (k + 1)) (by omega) (k + 1) (acc * k)
          (HeapCalls.setLocal (HeapCalls.setLocal env "k" (.scalar (.int k)))
            "fact" (.scalar (.int ((acc * k : Nat) : Int))))
          (by simp [HeapCalls.setLocal]) rfl
        simpa only [Calls.loopContinuation, show Calls.loopControl body = false from rfl,
          Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using next

theorem factorial_digit_sum.result (n : Nat) (heap : Mutable.Heap) :
    HeapCalls.ReturnsValue program "factorial_digit_sum" [.scalar (.int n)] heap
      (.scalar (.int (ProjectEulerSolutions.P20.solve n))) heap := by
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.assign HeapCalls.Evaluates.constant
  apply HeapCalls.BlockReturns.for_range (start := 2) (stop := (n + 1)) (stride := 1)
    (values := [.scalar (.int 2), .scalar (.int (n + 1))])
  · exact .cons HeapCalls.Evaluates.constant (.cons
      (HeapCalls.Evaluates.binary (finalHeap := heap) (HeapCalls.Evaluates.lookup (value := .scalar (.int n)) (by simp [HeapCalls.setLocal,
        HeapCalls.functionEnv, HeapCalls.argumentEnv])) HeapCalls.Evaluates.constant
        (by simp [Mutable.binary, scalarBinary])) .nil)
  · rfl
  · exact loop_result n 2 1 _ heap (by simp [HeapCalls.setLocal])

theorem factorial_digit_sum.«Equiv(naive)» (n : Int) (condition : Calls.Condition program
    (fun name => if name = "n" then some (.int n) else none) Solvers.P20Ast.requires_factorial_digit_sum_0) :
    (HeapCalls.scalarFunction program "factorial_digit_sum" [.int n]).Equiv
      (HeapCalls.scalarFunction P20Ast.module "naive" [.int n]) := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
  lift n to Nat using nonnegative
  have reference := data.«project-euler-statements».data.py.project_euler_statements.p20.naive.result n []
  rw [ProjectEulerSolutions.P20.equiv] at reference
  have optimized := HeapCalls.ReturnsValue.scalar_finishes (values := [.int n]) (factorial_digit_sum.result n [])
  have referenceRun := HeapCalls.ReturnsValue.scalar_finishes (values := [.int n]) reference
  intro outcome
  constructor
  · intro run
    rw [run.unique optimized]
    exact referenceRun
  · intro run
    rw [run.unique referenceRun]
    exact optimized

theorem factorial_digit_sum.«Supported» :
    (∀ (n : Int), Calls.Condition program (fun name => if name = "n" then some (.int n) else none)
      Solvers.P20Ast.requires_factorial_digit_sum_0 → (HeapCalls.scalarFunction program "factorial_digit_sum" [.int n]).Supported) ∧
    (∀ (n : Int), Calls.Condition program (fun name => if name = "n" then some (.int n) else none)
      Solvers.P20Ast.requires_factorial_digit_sum_0 → (HeapCalls.scalarFunction P20Ast.module "naive" [.int n]).Supported) := by
  constructor
  · intro n condition
    have nonnegative : 0 ≤ n := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
    lift n to Nat using nonnegative
    exact (factorial_digit_sum.result n []).scalar_returns.supported
  · intro n condition
    have nonnegative : 0 ≤ n := by
      simpa [evalCmpOp] using Calls.Evaluates.condition_int_compare condition rfl
    lift n to Nat using nonnegative
    exact (data.«project-euler-statements».data.py.project_euler_statements.p20.naive.result n []).scalar_returns.supported


end solvers.«20»
