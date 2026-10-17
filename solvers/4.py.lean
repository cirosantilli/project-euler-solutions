import «.formalsw».solvers.«4.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p4.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P4

namespace solvers.«4»
open FormalSW.PyAst FormalSW.PyAst.Calls
private abbrev program := Solvers.P4Ast.module

namespace Proof

def Terminates (env : ScalarEnv) (body : List Stmt) : Prop :=
  ∃ value, BlockReturns program env body value

namespace Terminates

theorem assign (evaluated : Evaluates program env expression value)
    (rest : Terminates (env.set name value) body) :
    Terminates env (.assign [.name name] expression :: body) := by
  obtain ⟨result, rest⟩ := rest
  exact ⟨result, BlockReturns.assign evaluated rest⟩

theorem branch (evaluated : Evaluates program env condition value)
    (rest : Terminates env ((if scalarTruth value then yes else no) ++ body)) :
    Terminates env (.ifThen condition yes no :: body) := by
  obtain ⟨result, rest⟩ := rest
  exact ⟨result, BlockReturns.branch evaluated rest⟩

theorem range_step (active : rangeContinues next stop stride = true)
    (rest : Terminates (env.set name (.int next))
      (body ++ [loopContinuation body (.rangeStepLoop name (next + stride) stop stride body otherwise)] ++ following)) :
    Terminates env (.rangeStepLoop name next stop stride body otherwise :: following) := by
  obtain ⟨result, rest⟩ := rest
  exact ⟨result, BlockReturns.range_step active rest⟩

theorem range_done (inactive : rangeContinues next stop stride = false)
    (rest : Terminates env (otherwise ++ following)) :
    Terminates env (.rangeStepLoop name next stop stride body otherwise :: following) := by
  obtain ⟨result, rest⟩ := rest
  exact ⟨result, BlockReturns.range_done inactive rest⟩

theorem break_loop (target : loopTarget body = some (next, following))
    (rest : Terminates env following) : Terminates env (.breakLoop :: body) := by
  obtain ⟨result, rest⟩ := rest
  exact ⟨result, BlockReturns.break_loop target rest⟩

theorem resume (rest : Terminates env (next :: following)) :
    Terminates env (.loopResume next :: following) := by
  obtain ⟨result, rest⟩ := rest
  exact ⟨result, BlockReturns.loop_resume rest⟩

theorem for_range (arguments : Evaluates.Arguments program env expressions values)
    (bounds : rangeBounds values = .ok (start, stop, stride))
    (rest : Terminates env (.rangeStepLoop name start stop stride body otherwise :: following)) :
    Terminates env (.forIn (.name name) (.call (.name "range") expressions) body otherwise :: following) := by
  obtain ⟨result, rest⟩ := rest
  exact ⟨result, BlockReturns.for_range arguments bounds rest⟩

end Terminates

structure Bounds (env : ScalarEnv) (lo hi : Int) : Prop where
  lower : env "lo" = some (.int lo)
  upper : env "hi" = some (.int hi)
  best : ∃ value : Int, env "best" = some (.int value)

theorem Bounds.setAux (bounds : Bounds env lo hi) (name : String) (value : Constant)
    (hl : "lo" ≠ name) (hh : "hi" ≠ name) (hb : "best" ≠ name) :
    Bounds (env.set name value) lo hi := by
  obtain ⟨best, getBest⟩ := bounds.best
  exact ⟨by simpa [ScalarEnv.set, hl] using bounds.lower,
    by simpa [ScalarEnv.set, hh] using bounds.upper,
    best, by simpa [ScalarEnv.set, hb] using getBest⟩

theorem Bounds.setBest (bounds : Bounds env lo hi) (value : Int) :
    Bounds (env.set "best" (.int value)) lo hi :=
  ⟨by simpa [ScalarEnv.set] using bounds.lower,
   by simpa [ScalarEnv.set] using bounds.upper, value, by simp [ScalarEnv.set]⟩

private def palindromeTest : Expr :=
  .compare (.name "s") [(.eq, .subscript (.name "s")
    (.slice none none (some (.unaryOp .uSub (.constant (.int 1))))))]

private def innerBody : List Stmt :=
  [.assign [.name "prod"] (.binOp (.name "a") .mult (.name "b")),
   .ifThen (.compare (.name "prod") [(.ltE, .name "best")]) [.breakLoop] [],
   .assign [.name "s"] (.builtinCall .str [.name "prod"]),
   .ifThen palindromeTest [.assign [.name "best"] (.name "prod"), .breakLoop] []]

private def outerBody : List Stmt :=
  [.ifThen (.compare (.binOp (.name "a") .mult (.name "hi")) [(.lt, .name "best")]) [.breakLoop] [],
   .forIn (.name "b") (.call (.name "range") [.name "a",
     .binOp (.name "lo") .sub (.constant (.int 1)), .unaryOp .uSub (.constant (.int 1))]) innerBody []]

private theorem palindrome_evaluates (env : ScalarEnv) (s : String) (getS : env "s" = some (.string s)) :
    Evaluates program env palindromeTest (.bool (decide (s = String.ofList s.toList.reverse))) := by
  apply Evaluates.compare (Evaluates.lookup getS)
  · apply Evaluates.slice (stepValue := .int (-1))
    · exact .cons (Evaluates.lookup getS) (.cons Evaluates.constant (.cons Evaluates.constant
        (.cons (Evaluates.unary Evaluates.constant rfl) .nil)))
    · exact Strings.slice_reverse s
  · simp [scalarCompare]

private theorem inner_loop (b stop a lo hi : Int) (env : ScalarEnv) (following : List Stmt)
    (bounds : Bounds env lo hi) (getA : env "a" = some (.int a))
    (after : ∀ nextEnv, Bounds nextEnv lo hi → Terminates nextEnv following) :
    Terminates env (.rangeStepLoop "b" b stop (-1) innerBody [] :: following) := by
  by_cases active : stop < b
  · obtain ⟨best, getBest⟩ := bounds.best
    apply Terminates.range_step (by simp [rangeContinues]; omega)
    let envB := env.set "b" (.int b)
    have boundsB : Bounds envB lo hi := bounds.setAux "b" (.int b) (by decide) (by decide) (by decide)
    apply Terminates.assign (value := .int (a * b))
    · exact Evaluates.binary (Evaluates.lookup (by simpa [envB, ScalarEnv.set] using getA))
        (Evaluates.lookup rfl) rfl
    · let envP := envB.set "prod" (.int (a * b))
      have boundsP : Bounds envP lo hi := boundsB.setAux "prod" (.int (a * b)) (by decide) (by decide) (by decide)
      by_cases beaten : a * b ≤ best
      · apply Terminates.branch (value := .bool true)
        · exact Evaluates.compare (Evaluates.lookup rfl)
            (Evaluates.lookup (by simpa [envP, envB, ScalarEnv.set] using getBest))
            (by simp [scalarCompare, evalCmpOp, beaten])
        · apply Terminates.break_loop rfl
          exact after envP boundsP
      · apply Terminates.branch (value := .bool false)
        · exact Evaluates.compare (Evaluates.lookup rfl)
            (Evaluates.lookup (by simpa [envP, envB, ScalarEnv.set] using getBest))
            (by simp [scalarCompare, evalCmpOp, beaten])
        · apply Terminates.assign (value := .string (toString (a * b)))
          · apply Evaluates.builtin (values := [.int (a * b)])
            · exact .cons (Evaluates.lookup rfl) .nil
            · rfl
          · let envS := envP.set "s" (.string (toString (a * b)))
            have boundsS : Bounds envS lo hi := boundsP.setAux "s" (.string (toString (a * b))) (by decide) (by decide) (by decide)
            apply Terminates.branch (palindrome_evaluates envS (toString (a * b)) rfl)
            by_cases palindrome : toString (a * b) = String.ofList (toString (a * b)).toList.reverse
            · have test : decide (toString (a * b) = String.ofList (toString (a * b)).toList.reverse) = true := by
                simp only [decide_eq_true_eq]
                exact palindrome
              simp only [test, scalarTruth]
              apply Terminates.assign (value := .int (a * b))
              · exact Evaluates.lookup rfl
              · apply Terminates.break_loop rfl
                exact after _ (boundsS.setBest (a * b))
            · have test : decide (toString (a * b) = String.ofList (toString (a * b)).toList.reverse) = false := by
                simp only [decide_eq_false_iff_not]
                exact palindrome
              simp only [test, scalarTruth]
              apply Terminates.resume
              have next := inner_loop (b - 1) stop a lo hi envS following boundsS
                (by simpa [envS, envP, envB, ScalarEnv.set] using getA) after
              simpa only [show b + (-1) = b - 1 by omega] using next
  · apply Terminates.range_done (by simp [rangeContinues]; omega)
    exact after env bounds
termination_by (b - stop).toNat
decreasing_by omega

private theorem finish (bounds : Bounds env lo hi) : Terminates env [.return (.name "best")] := by
  obtain ⟨best, getBest⟩ := bounds.best
  exact ⟨.int best, BlockReturns.return_value (Evaluates.lookup getBest)⟩

private theorem outer_loop (a stop lo hi : Int) (env : ScalarEnv) (bounds : Bounds env lo hi) :
    Terminates env [.rangeStepLoop "a" a stop (-1) outerBody [], .return (.name "best")] := by
  by_cases active : stop < a
  · obtain ⟨best, getBest⟩ := bounds.best
    apply Terminates.range_step (by simp [rangeContinues]; omega)
    let envA := env.set "a" (.int a)
    have boundsA : Bounds envA lo hi := bounds.setAux "a" (.int a) (by decide) (by decide) (by decide)
    by_cases beaten : a * hi < best
    · apply Terminates.branch (value := .bool true)
      · exact Evaluates.compare
          (Evaluates.binary (Evaluates.lookup rfl) (Evaluates.lookup boundsA.upper) rfl)
          (Evaluates.lookup (by simpa [envA, ScalarEnv.set] using getBest))
          (by simp [scalarCompare, evalCmpOp, beaten])
      · apply Terminates.break_loop rfl
        exact finish boundsA
    · apply Terminates.branch (value := .bool false)
      · exact Evaluates.compare
          (Evaluates.binary (Evaluates.lookup rfl) (Evaluates.lookup boundsA.upper) rfl)
          (Evaluates.lookup (by simpa [envA, ScalarEnv.set] using getBest))
          (by simp [scalarCompare, evalCmpOp, beaten])
      · apply Terminates.for_range (values := [.int a, .int (lo - 1), .int (-1)])
        · exact .cons (Evaluates.lookup rfl) (.cons
            (Evaluates.binary (Evaluates.lookup boundsA.lower) Evaluates.constant rfl)
            (.cons (Evaluates.unary Evaluates.constant rfl) .nil))
        · rfl
        · apply inner_loop a (lo - 1) a lo hi envA _ boundsA rfl
          intro nextEnv nextBounds
          apply Terminates.resume
          simpa only [show a + (-1) = a - 1 by omega] using
            outer_loop (a - 1) stop lo hi nextEnv nextBounds
  · apply Terminates.range_done (by simp [rangeContinues]; omega)
    exact finish bounds
termination_by (a - stop).toNat
decreasing_by omega

end Proof

theorem largest_palindrome_product.result (lo hi : Int) :
    ∃ value, ReturnsValue program "largest_palindrome_product" [.int lo, .int hi] value := by
  have body : Proof.Terminates
      (argumentEnv [{ name := "lo", annotation := some (.name "int") },
        { name := "hi", annotation := some (.name "int") }] [.int lo, .int hi])
      [.assign [.name "best"] (.constant (.int 0)),
       .forIn (.name "a") (.call (.name "range") [.name "hi",
         .binOp (.name "lo") .sub (.constant (.int 1)), .unaryOp .uSub (.constant (.int 1))]) Proof.outerBody [],
       .return (.name "best")] := by
    apply Proof.Terminates.assign Evaluates.constant
    apply Proof.Terminates.for_range (values := [.int hi, .int (lo - 1), .int (-1)])
    · exact .cons (Evaluates.lookup rfl) (.cons
        (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant rfl)
        (.cons (Evaluates.unary Evaluates.constant rfl) .nil))
    · rfl
    · exact Proof.outer_loop hi (lo - 1) lo hi _ ⟨rfl, rfl, 0, rfl⟩
  obtain ⟨value, body⟩ := body
  exact ⟨value, BlockReturns.function rfl rfl body⟩

theorem largest_palindrome_product.«Returns» (lo hi : Int) :
    (Calls.function program "largest_palindrome_product" [.int lo, .int hi]).Returns := by
  obtain ⟨value, result⟩ := largest_palindrome_product.result lo hi
  exact result.returns


namespace Proof
open ProjectEulerSolutions.P4

private structure Values (env : ScalarEnv) (lo hi best : Nat) : Prop where
  lower : env "lo" = some (.int lo)
  upper : env "hi" = some (.int hi)
  maximum : env "best" = some (.int best)

private theorem Values.setAux (h : Values env lo hi best) (name : String) (value : Constant)
    (hl : "lo" ≠ name) (hh : "hi" ≠ name) (hb : "best" ≠ name) :
    Values (env.set name value) lo hi best :=
  ⟨by simpa [ScalarEnv.set, hl] using h.lower,
   by simpa [ScalarEnv.set, hh] using h.upper,
   by simpa [ScalarEnv.set, hb] using h.maximum⟩

private theorem Values.setBest (h : Values env lo hi best) (value : Nat) :
    Values (env.set "best" (.int value)) lo hi value :=
  ⟨by simpa [ScalarEnv.set] using h.lower,
   by simpa [ScalarEnv.set] using h.upper, by simp [ScalarEnv.set]⟩

/-- The inner Python loop implements the first component of the existing Lean
algorithm. The continuation retains the actual Python environment. -/
private theorem inner_correct (b a lo hi best bestA bestB : Nat) (env : ScalarEnv)
    (following : List Stmt) (result : Constant)
    (values : Values env lo hi best) (getA : env "a" = some (.int a))
    (after : ∀ nextEnv, Values nextEnv lo hi (loopB a b lo best bestA bestB).1 →
      BlockReturns program nextEnv following result) :
    BlockReturns program env
      (.rangeStepLoop "b" b ((lo : Int) - 1) (-1) innerBody [] :: following) result := by
  by_cases active : lo ≤ b
  · apply BlockReturns.range_step (by simp [rangeContinues]; omega)
    let envB := env.set "b" (.int b)
    have valuesB := values.setAux "b" (.int b) (by decide) (by decide) (by decide)
    apply BlockReturns.assign (value := .int (a * b))
    · exact Evaluates.binary (Evaluates.lookup (by simpa [envB, ScalarEnv.set] using getA))
        (Evaluates.lookup rfl) (by simp [scalarBinary])
    · let envP := envB.set "prod" (.int (a * b))
      have valuesP := valuesB.setAux "prod" (.int (a * b)) (by decide) (by decide) (by decide)
      by_cases beaten : a * b ≤ best
      · apply BlockReturns.branch (value := .bool true)
        · exact Evaluates.compare (Evaluates.lookup rfl)
            (Evaluates.lookup (by simpa [envP, envB, ScalarEnv.set] using values.maximum))
            (by simp [scalarCompare, evalCmpOp]; exact_mod_cast beaten)
        · apply BlockReturns.break_loop rfl
          apply after envP
          rw [loopB]
          simpa [Nat.not_lt.mpr active, beaten] using valuesP
      · have positive : 0 < b := by
          by_contra h
          have : b = 0 := by omega
          simp [this] at beaten
        apply BlockReturns.branch (value := .bool false)
        · exact Evaluates.compare (Evaluates.lookup rfl)
            (Evaluates.lookup (by simpa [envP, envB, ScalarEnv.set] using values.maximum))
            (by simp [scalarCompare, evalCmpOp]; exact_mod_cast Nat.lt_of_not_ge beaten)
        · apply BlockReturns.assign (value := .string (toString (a * b)))
          · apply Evaluates.builtin (values := [.int (a * b)])
            · exact .cons (Evaluates.lookup rfl) .nil
            · rfl
          · let envS := envP.set "s" (.string (toString (a * b)))
            have valuesS := valuesP.setAux "s" (.string (toString (a * b))) (by decide) (by decide) (by decide)
            apply BlockReturns.branch (palindrome_evaluates envS (toString (a * b)) rfl)
            by_cases palindrome : isPalindrome (a * b) = true
            · have test : decide (toString (a * b) = String.ofList (toString (a * b)).toList.reverse) = true := palindrome
              simp only [test, scalarTruth]
              apply BlockReturns.assign (value := .int (a * b))
              · exact Evaluates.lookup rfl
              · apply BlockReturns.break_loop rfl
                apply after _
                rw [loopB]
                simpa [Nat.not_lt.mpr active, Nat.ne_of_gt positive, beaten, palindrome]
                  using valuesS.setBest (a * b)
            · have test : decide (toString (a * b) = String.ofList (toString (a * b)).toList.reverse) = false :=
                Bool.eq_false_iff.mpr palindrome
              simp only [test, scalarTruth]
              apply BlockReturns.loop_resume
              have recurse := inner_correct (b - 1) a lo hi best bestA bestB envS following result valuesS
                (by simpa [envS, envP, envB, ScalarEnv.set] using getA) (by
                  intro nextEnv nextValues
                  apply after nextEnv
                  rw [loopB]
                  simpa [Nat.not_lt.mpr active, Nat.ne_of_gt positive, beaten, palindrome]
                    using nextValues)
              simpa only [show (b : Int) + (-1) = ((b - 1 : Nat) : Int) by omega] using recurse
  · apply BlockReturns.range_done (by simp [rangeContinues]; omega)
    apply after env
    simpa [loopB, show b < lo by omega] using values
termination_by b


private theorem outer_correct (a lo hi best bestA bestB : Nat) (env : ScalarEnv)
    (values : Values env lo hi best) :
    BlockReturns program env
      [.rangeStepLoop "a" a ((lo : Int) - 1) (-1) outerBody [], .return (.name "best")]
      (.int (loopA a lo hi best bestA bestB).1) := by
  by_cases active : lo ≤ a
  · apply BlockReturns.range_step (by simp [rangeContinues]; omega)
    let envA := env.set "a" (.int a)
    have valuesA := values.setAux "a" (.int a) (by decide) (by decide) (by decide)
    by_cases beaten : a * hi < best
    · apply BlockReturns.branch (value := .bool true)
      · exact Evaluates.compare
          (Evaluates.binary (Evaluates.lookup rfl) (Evaluates.lookup valuesA.upper) rfl)
          (Evaluates.lookup (by simpa [envA, ScalarEnv.set] using values.maximum))
          (by simp [scalarCompare, evalCmpOp]; exact_mod_cast beaten)
      · apply BlockReturns.break_loop rfl
        rw [loopA]
        simpa [Nat.not_lt.mpr active, beaten] using
          (BlockReturns.return_value (rest := []) (Evaluates.lookup valuesA.maximum))
    · apply BlockReturns.branch (value := .bool false)
      · exact Evaluates.compare
          (Evaluates.binary (Evaluates.lookup rfl) (Evaluates.lookup valuesA.upper) rfl)
          (Evaluates.lookup (by simpa [envA, ScalarEnv.set] using values.maximum))
          (by simp [scalarCompare, evalCmpOp]; exact_mod_cast Nat.le_of_not_gt beaten)
      · apply BlockReturns.for_range (values := [.int a, .int ((lo : Int) - 1), .int (-1)])
        · exact .cons (Evaluates.lookup rfl) (.cons
            (Evaluates.binary (Evaluates.lookup valuesA.lower) Evaluates.constant rfl)
            (.cons (Evaluates.unary Evaluates.constant rfl) .nil))
        · rfl
        · apply inner_correct a a lo hi best bestA bestB envA _ _ valuesA rfl
          intro nextEnv nextValues
          apply BlockReturns.loop_resume
          by_cases zero : a = 0
          · apply BlockReturns.range_done (by simp [rangeContinues]; omega)
            rw [loopA]
            simpa only [if_neg (Nat.not_lt.mpr active), if_neg beaten, if_pos zero, List.nil_append] using
              (BlockReturns.return_value (rest := []) (Evaluates.lookup nextValues.maximum))
          · rw [loopA]
            simp only [Nat.not_lt.mpr active, beaten, zero, ↓reduceIte]
            have recurse := outer_correct (a - 1) lo hi
              (loopB a a lo best bestA bestB).1 (loopB a a lo best bestA bestB).2.1
              (loopB a a lo best bestA bestB).2.2 nextEnv nextValues
            simpa only [show (a : Int) + (-1) = ((a - 1 : Nat) : Int) by omega] using recurse
  · apply BlockReturns.range_done (by simp [rangeContinues]; omega)
    rw [loopA]
    simpa [show a < lo by omega] using
      (BlockReturns.return_value (rest := []) (Evaluates.lookup values.maximum))
termination_by a

end Proof

/-- Link the parsed Python helper to the previously proved mathematical algorithm. -/
theorem largest_palindrome_product.correct (lo hi : Nat) :
    ReturnsValue program "largest_palindrome_product" [.int lo, .int hi]
      (.int (ProjectEulerSolutions.P4.largestPalindromeProduct lo hi).1) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.for_range (values := [.int hi, .int ((lo : Int) - 1), .int (-1)])
  · exact .cons (Evaluates.lookup rfl) (.cons
      (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant rfl)
      (.cons (Evaluates.unary Evaluates.constant rfl) .nil))
  · rfl
  · exact Proof.outer_correct hi lo hi 0 0 0 _ ⟨rfl, rfl, rfl⟩




private theorem lower_evaluates (n : Nat) (env : ScalarEnv)
    (getN : env "n" = some (.int n)) :
    Evaluates program env
      (.ifExp (.compare (.name "n") [(.eq, .constant (.int 0))]) (.constant (.int 0))
        (.binOp (.constant (.int 10)) .pow (.binOp (.name "n") .sub (.constant (.int 1)))))
      (.int (ProjectEulerStatements.P4.digitLower n)) := by
  cases n with
  | zero =>
      apply Evaluates.conditional (c := .bool true)
      · exact Evaluates.compare (Evaluates.lookup getN) Evaluates.constant rfl
      · exact Evaluates.constant
  | succ n =>
      apply Evaluates.conditional (c := .bool false)
      · exact Evaluates.compare (Evaluates.lookup getN) Evaluates.constant
          (by simp [scalarCompare, evalCmpOp]; omega)
      · apply Evaluates.binary Evaluates.constant (bv := .int n)
        · exact Evaluates.binary (Evaluates.lookup getN) Evaluates.constant (by simp [scalarBinary])
        · simp [scalarBinary, ProjectEulerStatements.P4.digitLower]

private theorem upper_evaluates (n : Nat) (env : ScalarEnv)
    (getN : env "n" = some (.int n)) :
    Evaluates program env
      (.binOp (.binOp (.constant (.int 10)) .pow (.name "n")) .sub (.constant (.int 1)))
      (.int (ProjectEulerStatements.P4.digitUpper n)) := by
  apply Evaluates.binary (av := .int (10 ^ n)) (bv := .int 1)
  · exact Evaluates.binary Evaluates.constant (Evaluates.lookup getN) (by simp [scalarBinary])
  · exact Evaluates.constant
  · have positive : 1 ≤ (10 : Nat) ^ n := Nat.one_le_pow n 10 (by decide)
    simp [scalarBinary, ProjectEulerStatements.P4.digitUpper, Int.natCast_sub positive]

theorem solve.result (n : Nat) :
    ReturnsValue program "solve" [.int n] (.int (ProjectEulerSolutions.P4.solve n)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (lower_evaluates n _ rfl)
  apply BlockReturns.assign (upper_evaluates n _ (by simp [ScalarEnv.set, argumentEnv]))
  apply BlockReturns.return_value
  apply Evaluates.call (values := [.int (ProjectEulerStatements.P4.digitLower n),
    .int (ProjectEulerStatements.P4.digitUpper n)])
  · exact .cons (Evaluates.lookup (by simp [ScalarEnv.set])) (.cons (Evaluates.lookup rfl) .nil)
  · exact largest_palindrome_product.correct _ _

/-- Reuse the mathematical correctness theorem after connecting the Python AST. -/
theorem solve.correct (n : Nat) :
    ReturnsValue program "solve" [.int n] (.int (ProjectEulerStatements.P4.naive n)) := by
  rw [ProjectEulerSolutions.P4.equiv]
  exact solve.result n

theorem solve.«Returns» (n : Int) (hp : Calls.Condition program
    (fun key => if key = "n" then some (.int n) else none) Solvers.P4Ast.requires_solve_0) :
    (Calls.function program "solve" [.int n]).Returns := by
  have hn : 0 ≤ n := by simpa [evalCmpOp] using Evaluates.condition_int_compare hp rfl
  simpa [Int.toNat_of_nonneg hn] using (solve.result n.toNat).returns

/-- The two Python interpreters reach the same mathematical result. Their
internal list allocations are local to the naive implementation. -/
theorem solve.«Equiv(naive)» (n : Int) (hp : Calls.Condition program
    (fun key => if key = "n" then some (.int n) else none) Solvers.P4Ast.requires_solve_0) :
    (Calls.function program "solve" [.int n]).Equiv
      (HeapCalls.scalarFunction P4Ast.module "naive" [.int n]) := by
  have nonnegative : 0 ≤ n := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare hp rfl
  lift n to Nat using nonnegative
  obtain ⟨heap, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p4.naive.correct n []
  have referenceRun := reference.scalar_finishes (values := [.int (n : Int)])
  have optimizedRun := (solve.correct n).finishes
  intro outcome
  constructor
  · intro run
    rw [run.unique optimizedRun]
    exact referenceRun
  · intro run
    rw [run.unique referenceRun]
    exact optimizedRun

theorem solve.«Supported» :
    (∀ (n : Int), Calls.Condition program
      (fun key => if key = "n" then some (.int n) else none) Solvers.P4Ast.requires_solve_0 →
      (Calls.function program "solve" [.int n]).Supported) ∧
    (∀ (n : Int), Calls.Condition program
      (fun key => if key = "n" then some (.int n) else none) Solvers.P4Ast.requires_solve_0 →
      (HeapCalls.scalarFunction P4Ast.module "naive" [.int n]).Supported) :=
  ⟨fun n hp => (solve.«Returns» n hp).supported,
   fun n hp => ((solve.«Equiv(naive)» n hp).returns_iff.mp (solve.«Returns» n hp)).supported⟩

end solvers.«4»
