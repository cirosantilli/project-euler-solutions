import «.formalsw».solvers.«53.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p53.py»
import ProjectEulerSolutions.Lib.BinomialSelections
import ProjectEulerSolutions.Lib.ScalarBlocks

/-! Source proofs for problem 53's exact binomial recurrence and symmetric early break. -/
set_option Elab.async false
namespace solvers.«53»
open FormalSW.PyAst FormalSW.PyAst.Calls ProjectEulerSolutions.BinomialSelections ProjectEulerSolutions.ScalarBlocks
private abbrev program := Solvers.P53Ast.module
private abbrev number (n : Int) : Constant := .int n
private abbrev flag (b : Bool) : Constant := .bool b
private def code : List Stmt := executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
private def rowBody : List Stmt := match (code[2]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def coefficientBody : List Stmt := match (rowBody[1]? : Option Stmt) with
  | some (.forIn _ _ body _) => body | _ => []
private def updateExpr : Expr := match (coefficientBody[0]? : Option Stmt) with
  | some (.assign _ value) => value | _ => .constant .none
private def massExpr : Expr := .binOp (.binOp (.name "n") .sub
  (.binOp (.constant (.int 2)) .mult (.name "r"))) .add (.constant (.int 1))

/-- The source expression's signed arithmetic and floor division give the exact next coefficient. -/
theorem coefficient_value (n r : Nat) (positive : 0 < r) (bounded : r ≤ n)
    (env : ScalarEnv) (getN : env "n" = some (number n))
    (getR : env "r" = some (number r)) (getC : env "c" = some (number (Nat.choose n (r - 1)))) :
    Evaluates program env updateExpr (number (Nat.choose n r)) := by
  have factor : Evaluates program env (.binOp (.binOp (.name "n") .sub (.name "r")) .add (.constant (.int 1)))
       (number ((n : Int) - r + 1)) := by
    apply Evaluates.binary (av := number ((n : Int) - r)) (bv := number 1)
    · exact Evaluates.binary (Evaluates.lookup getN) (Evaluates.lookup getR) (by simp [scalarBinary])
    · exact Evaluates.constant
    · simp [scalarBinary]
  have natural : (n : Int) - r + 1 = ((n - r + 1 : Nat) : Int) := by omega
  rw [natural] at factor
  have evaluated : Evaluates program env updateExpr (number (Nat.choose n (r - 1) * (n - r + 1) / r : Nat)) := by
    apply Evaluates.binary (av := number (Nat.choose n (r - 1) * (n - r + 1) : Nat)) (bv := number r)
    · exact Evaluates.binary (Evaluates.lookup getC) factor (by simp [scalarBinary])
    · exact Evaluates.lookup getR
    · simp [scalarBinary, number, positive, positive.ne']
  simpa only [update n r positive bounded] using evaluated

/-- The counted central interval is nonnegative, so Python subtraction agrees with natural arithmetic. -/
private theorem mass_value (n r : Nat) (middle : r ≤ n / 2) (env : ScalarEnv)
    (getN : env "n" = some (number n)) (getR : env "r" = some (number r)) :
    Evaluates program env massExpr (number (n - 2 * r + 1 : Nat)) := by
  have evaluated : Evaluates program env massExpr (number ((n : Int) - 2 * r + 1)) := by
    apply Evaluates.binary (av := number ((n : Int) - 2 * r)) (bv := number 1)
    · apply Evaluates.binary (av := number n) (bv := number ((2 : Int) * r))
      · exact Evaluates.lookup getN
      · exact Evaluates.binary Evaluates.constant (Evaluates.lookup getR) (by simp [scalarBinary])
      · simp [scalarBinary]
    · exact Evaluates.constant
    · simp [scalarBinary]
  have natural : (n : Int) - 2 * r + 1 = ((n - 2 * r + 1 : Nat) : Int) := by omega
  simpa only [natural] using evaluated

private def RowStable (before after : ScalarEnv) : Prop :=
  ∀ name, name ≠ "r" → name ≠ "c" → name ≠ "total" → after name = before name

/-- The actual inner range either breaks at its first hit or exhausts the rejected half-row. -/
theorem row_loop (n first count : Nat) (length : first + count = n / 2 + 1) (positive : 1 ≤ first)
    (threshold initial : Int) (earlier : ∀ r < first, (Nat.choose n r : Int) ≤ threshold)
    (env : ScalarEnv) (getN : env "n" = some (number n))
    (getThreshold : env "threshold" = some (number threshold)) (getTotal : env "total" = some (number initial))
    (getC : env "c" = some (number (Nat.choose n (first - 1)))) :
    ∃ finalEnv, Continues program env [.rangeStepLoop "r" first (n / 2 + 1 : Nat) 1 coefficientBody []] finalEnv ∧
      RowStable env finalEnv ∧ finalEnv "total" = some (number (initial + row n threshold)) := by
  induction count generalizing first env with
  | zero =>
      have empty := exhausted n first threshold (by omega) earlier
      refine ⟨env, Continues.range_done (by simp [Calls.rangeContinues]; omega) .nil, fun _ _ _ _ => rfl, ?_⟩
      simpa only [empty, Nat.cast_zero, Int.add_zero] using getTotal
  | succ count ih =>
      let bound := ScalarEnv.set env "r" (number first)
      let ready := ScalarEnv.set bound "c" (number (Nat.choose n first))
      have middle : first ≤ n / 2 := by omega
      have boundN : bound "n" = some (number n) := by simpa [bound, ScalarEnv.set] using getN
      have boundThreshold : bound "threshold" = some (number threshold) := by simpa [bound, ScalarEnv.set] using getThreshold
      have boundTotal : bound "total" = some (number initial) := by simpa [bound, ScalarEnv.set] using getTotal
      have assigned : Continues program bound [coefficientBody[0]'(by decide)] ready := by
        apply Continues.assign
        exact coefficient_value n first positive (by omega) bound boundN (by simp [bound, ScalarEnv.set]) (by simpa [bound, ScalarEnv.set] using getC)
      have readyN : ready "n" = some (number n) := by simpa [ready, ScalarEnv.set] using boundN
      have readyThreshold : ready "threshold" = some (number threshold) := by simpa [ready, ScalarEnv.set] using boundThreshold
      have readyTotal : ready "total" = some (number initial) := by simpa [ready, ScalarEnv.set] using boundTotal
      have tested : Evaluates program ready (.compare (.name "c") [(.gt, .name "threshold")])
          (flag (decide (threshold < (Nat.choose n first : Int)))) :=
        Evaluates.compare (av := number (Nat.choose n first)) (bv := number threshold) (Evaluates.lookup (by simp [ready, ScalarEnv.set])) (Evaluates.lookup readyThreshold) rfl
      by_cases hit : threshold < (Nat.choose n first : Int)
      · have mass := first_hit n first threshold middle earlier hit
        let finalEnv := ScalarEnv.set ready "total" (number (initial + row n threshold))
        refine ⟨finalEnv, ?_, ?_, by simp [finalEnv, ScalarEnv.set]⟩
        · apply Continues.range_step (by simp [Calls.rangeContinues]; omega)
          apply assigned.append
          apply Continues.branch_next (flag := true) (by simpa only [hit, decide_true] using tested) rfl
          have incremented : Continues program ready [.augAssign (.name "total") .add massExpr] finalEnv := by
            apply Continues.augAssign
            exact Evaluates.binary (av := number initial) (bv := number (row n threshold)) (Evaluates.lookup readyTotal)
              (by simpa only [mass] using mass_value n first middle ready readyN (by simp [ready, bound, ScalarEnv.set]))
              (by simp [scalarBinary])
          apply incremented.append
          exact Continues.break_loop rfl .nil
        · intro name notR notC notTotal
          simp [finalEnv, ready, bound, ScalarEnv.set, notR, notC, notTotal]
      · have nextEarlier : ∀ r < first + 1, (Nat.choose n r : Int) ≤ threshold := by
          intro r bound
          by_cases smaller : r < first
          · exact earlier r smaller
          · have same : r = first := by omega
            simpa only [same] using (show (Nat.choose n first : Int) ≤ threshold by omega)
        obtain ⟨finalEnv, continued, stable, total⟩ := ih (first + 1) (by omega) (by omega) nextEarlier ready readyN readyThreshold readyTotal
          (by simp [ready, ScalarEnv.set])
        refine ⟨finalEnv, ?_, ?_, total⟩
        · apply Continues.range_step (by simp [Calls.rangeContinues]; omega)
          apply assigned.append
          apply Continues.branch_next (flag := false) (by simpa only [hit, decide_false] using tested) rfl
          have resumed := Continues.loop_resume continued
          simpa [coefficientBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Nat.cast_add] using resumed
        · intro name notR notC notTotal
          exact (stable name notR notC notTotal).trans (by simp [ready, bound, ScalarEnv.set, notR, notC])
/-- Initializing the coefficient and scanning its half-row produces the complete reference row count. -/
theorem row_result (n : Nat) (threshold initial : Int) (high : 1 ≤ threshold)
    (env : ScalarEnv) (getN : env "n" = some (number n)) (getThreshold : env "threshold" = some (number threshold))
    (getTotal : env "total" = some (number initial)) :
    ∃ finalEnv, Continues program env rowBody finalEnv ∧ RowStable env finalEnv ∧
      finalEnv "total" = some (number (initial + row n threshold)) := by
  let ready := env.set "c" (number 1)
  have readyN : ready "n" = some (number n) := by simpa [ready, ScalarEnv.set] using getN
  obtain ⟨finalEnv, continued, stable, total⟩ := row_loop n 1 (n / 2) (by omega) (by decide) threshold initial
    (by
      intro r before
      have same : r = 0 := by omega
      subst r
      simpa using high) ready readyN
    (by simpa [ready, ScalarEnv.set] using getThreshold) (by simpa [ready, ScalarEnv.set] using getTotal)
    (by simp [ready, ScalarEnv.set])
  refine ⟨finalEnv, ?_, ?_, total⟩
  · apply Continues.append (body := [rowBody[0]'(by decide)]) (middle := ready) (Continues.assign Evaluates.constant)
    apply Continues.for_range (values := [number 1, number (n / 2 + 1 : Nat)]) (first := 1) (stop := ((n / 2 + 1 : Nat) : Int)) (stride := 1)
    · refine .cons Evaluates.constant (.cons ?_ .nil)
      apply Evaluates.binary (av := number (n / 2 : Nat)) (bv := number 1)
      · exact Evaluates.binary (Evaluates.lookup readyN) Evaluates.constant (by simp [scalarBinary, number])
      · exact Evaluates.constant
      · simp [scalarBinary, number]
    · rfl
    · exact continued
  · intro name notR notC notTotal
    exact (stable name notR notC notTotal).trans (by simp [ready, ScalarEnv.set, notC])

private theorem outer_loop (first count stop : Nat) (length : first + count = stop) (threshold initial : Int)
    (high : 1 ≤ threshold) (env : ScalarEnv) (getThreshold : env "threshold" = some (number threshold))
    (getTotal : env "total" = some (number initial)) :
    ∃ finalEnv, Continues program env [.rangeStepLoop "n" first stop 1 rowBody []] finalEnv ∧
      finalEnv "threshold" = some (number threshold) ∧ finalEnv "total" = some (number (initial + series first count threshold)) := by
  induction count generalizing first initial env with
  | zero =>
      exact ⟨env, Continues.range_done (by simp [Calls.rangeContinues]; omega) .nil, getThreshold, by simpa using getTotal⟩
  | succ count ih =>
      let bound := env.set "n" (number first)
      obtain ⟨ready, counted, stable, subtotal⟩ := row_result first threshold initial high bound (by simp [bound, ScalarEnv.set])
        (by simpa [bound, ScalarEnv.set] using getThreshold) (by simpa [bound, ScalarEnv.set] using getTotal)
      have readyThreshold : ready "threshold" = some (number threshold) :=
        (stable "threshold" (by decide) (by decide) (by decide)).trans (by simpa [bound, ScalarEnv.set] using getThreshold)
      obtain ⟨finalEnv, continued, finalThreshold, total⟩ := ih (first + 1) (by omega) (initial + row first threshold) ready readyThreshold subtotal
      refine ⟨finalEnv, ?_, finalThreshold, ?_⟩
      · apply Continues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [rowBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, Nat.cast_add] using counted.append continued
      · simpa only [series_succ, Nat.cast_add, ← Int.add_assoc] using total

private def lowExpr : Expr := match (code[0]? : Option Stmt) with
  | some (.ifThen _ [.return value] _) => value | _ => .constant .none

/-- The source shortcut counts both row endpoints and handles empty signed bounds. -/
private theorem low_value (limit threshold : Int) (low : threshold < 1) (env : ScalarEnv)
    (getLimit : env "limit_n" = some (number limit)) : Evaluates program env lowExpr (number (answer limit threshold)) := by
  rw [answer_low limit threshold low]
  by_cases positive : 0 < limit
  · lift limit to Nat using (by omega : 0 ≤ limit)
    simp only [Int.toNat_natCast]
    apply Evaluates.conditional (c := flag true)
    · exact Evaluates.compare (Evaluates.lookup getLimit) Evaluates.constant (by simp [scalarCompare, evalCmpOp, show 0 < limit by omega])
    · apply Evaluates.binary (av := number (limit * (limit + 3) : Nat)) (bv := number 2)
      · apply Evaluates.binary (av := number limit) (bv := number (limit + 3 : Nat))
        · exact Evaluates.lookup getLimit
        · exact Evaluates.binary (Evaluates.lookup getLimit) Evaluates.constant (by simp [scalarBinary, number])
        · simp [scalarBinary, number]
      · exact Evaluates.constant
      · simp [scalarBinary, number]
  · have empty : limit.toNat = 0 := by omega
    simp only [empty, Nat.zero_mul, Nat.zero_div]
    apply Evaluates.conditional (c := flag false)
    · exact Evaluates.compare (Evaluates.lookup getLimit) Evaluates.constant (by simp [scalarCompare, evalCmpOp, positive])
    · exact Evaluates.constant

namespace solve
/-- Every signed bound and threshold terminate with the exact reference coefficient count. -/
theorem result (limit threshold : Int) : ReturnsValue program "solve" [number limit, number threshold] (number (answer limit threshold)) := by
  let entry := argumentEnv ((findFunctionIn program.body "solve").getD ([], [])).1 [number limit, number threshold]
  let ready := entry.set "total" (number 0)
  apply BlockReturns.function rfl rfl
  by_cases low : threshold < 1
  · apply BlockReturns.branch (value := flag true)
      (Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant (by simp [scalarCompare, evalCmpOp, low]))
    exact BlockReturns.return_value (low_value limit threshold low entry rfl)
  · apply BlockReturns.branch (value := flag false)
      (Evaluates.compare (Evaluates.lookup rfl) Evaluates.constant (by simp [scalarCompare, evalCmpOp, low]))
    apply BlockReturns.assign Evaluates.constant
    by_cases nonnegative : 0 ≤ limit
    · lift limit to Nat using nonnegative
      obtain ⟨finalEnv, scanned, _, total⟩ := outer_loop 1 limit (limit + 1) (by omega) threshold 0 (by omega) ready rfl rfl
      apply BlockReturns.for_range (values := [number 1, number (limit + 1 : Nat)])
        (start := 1) (stop := ((limit + 1 : Nat) : Int)) (stride := 1)
      · exact .cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant (by simp [scalarBinary, number])) .nil)
      · rfl
      · apply scanned.returns
        exact BlockReturns.return_value (Evaluates.lookup (by simpa [answer] using total))
    · have empty : limit.toNat = 0 := by omega
      simp only [answer, empty, series_zero]
      apply BlockReturns.for_range (values := [number 1, number (limit + 1)])
        (.cons Evaluates.constant (.cons (Evaluates.binary (Evaluates.lookup rfl) Evaluates.constant rfl) .nil)) rfl
      apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup rfl)

theorem «Returns» (limit threshold : Int) : (Calls.function program "solve" [.int limit, .int threshold]).Returns :=
  (result limit threshold).returns

theorem «Equiv(naive)» (limit threshold : Int) :
    (Calls.function program "solve" [.int limit, .int threshold]).Equiv
      (HeapCalls.scalarFunction P53Ast.module "naive" [.int limit, .int threshold]) := by
  have optimized := (result limit threshold).finishes
  have original := (data.«project-euler-statements».data.py.project_euler_statements.p53.naive.result limit threshold []).scalar_finishes
    (values := [.int limit, .int threshold])
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact original
  · intro finished
    rw [finished.unique original]
    exact optimized

theorem «Supported» :
    (∀ limit threshold : Int, (Calls.function program "solve" [.int limit, .int threshold]).Supported) ∧
    (∀ limit threshold : Int, (HeapCalls.scalarFunction P53Ast.module "naive" [.int limit, .int threshold]).Supported) :=
  ⟨fun limit threshold => («Returns» limit threshold).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p53.naive.«Supported»⟩
end solve

end solvers.«53»
