import «.formalsw».solvers.«26.py.ast»
import ProjectEulerStatements.Lib.RecurringCycle
import ProjectEulerStatements.Lib.ArgMaximum
import lang.py.FormalSW.Calls
import data.«project-euler-statements».data.py.project_euler_statements.«p26.py»

namespace solvers.«26»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerStatements.P26

private abbrev program := Solvers.P26Ast.module

private def stripLoop (p : Nat) : Stmt :=
  .whileDo (.compare (.binOp (.name "d") .mod (.constant (.int p))) [(.eq, .constant (.int 0))])
    [.augAssign (.name "d") .floorDiv (.constant (.int p))] []

/-- Each factor-removal loop decreases its positive integer argument. -/
private theorem strip_loop (p n : Nat) (large : 1 < p) (positive : 0 < n)
    (env : ScalarEnv) (lookup : env "d" = some (.int n)) (following : List Stmt) (answer : Constant)
    (finish : ∀ finalEnv, finalEnv "d" = some (.int (stripFactor p n)) →
      BlockReturns program finalEnv following answer) :
    BlockReturns program env (stripLoop p :: following) answer := by
  induction n using Nat.strong_induction_on generalizing env with
  | h n ih =>
      by_cases divisible : n % p = 0
      · have decreased : n / p < n := Nat.div_lt_self positive large
        have quotientPos : 0 < n / p := Nat.div_pos (Nat.le_of_dvd positive (Nat.dvd_of_mod_eq_zero divisible)) (by omega)
        have same : stripFactor p n = stripFactor p (n / p) := by
          rw [stripFactor]
          simp [show ¬ p ≤ 1 by omega, show n ≠ 0 by omega, divisible]
        apply BlockReturns.while_loop (value := .bool true)
        · exact Evaluates.compare (Evaluates.binary (value := .int (n % p)) (Evaluates.lookup lookup) Evaluates.constant
            (by simp [scalarBinary, show p ≠ 0 by omega, show 0 < p by omega])) Evaluates.constant
            (by simp [scalarCompare, evalCmpOp, Int.natCast_dvd_natCast, Nat.dvd_iff_mod_eq_zero, divisible])
        · apply BlockReturns.augAssign (value := .int (n / p))
          · exact Evaluates.binary (Evaluates.lookup lookup) Evaluates.constant
              (by simp [scalarBinary, show p ≠ 0 by omega])
          · apply ih (n / p) decreased quotientPos _ (by simp [ScalarEnv.set])
            intro finalEnv finalLookup
            exact finish finalEnv (by simpa [same] using finalLookup)
      · have same : stripFactor p n = n := by
          rw [stripFactor]
          simp [show ¬ p ≤ 1 by omega, show n ≠ 0 by omega, divisible]
        apply BlockReturns.while_loop (value := .bool false)
        · exact Evaluates.compare (Evaluates.binary (value := .int (n % p)) (Evaluates.lookup lookup) Evaluates.constant
            (by simp [scalarBinary, show p ≠ 0 by omega, show 0 < p by omega])) Evaluates.constant
            (by simp [scalarCompare, evalCmpOp, Int.natCast_dvd_natCast, Nat.dvd_iff_mod_eq_zero, divisible])
        · exact finish env (by simpa [same] using lookup)


open ProjectEulerStatements.RecurringCycle

private def periodLoop : List Stmt :=
  (executableBody ((findFunctionIn program.body "recurring_cycle_length").getD ([], [])).2).drop 5

private theorem period_loop (d k : Nat) (large : 1 < d) (coprime : Nat.Coprime 10 d)
    (positive : 0 < k) (bound : k ≤ first d) (env : ScalarEnv)
    (getD : env "d" = some (.int d)) (getR : env "r" = some (.int (10 ^ k % d)))
    (getK : env "k" = some (.int k)) :
    BlockReturns program env periodLoop (.int (first d)) := by
  induction measure : first d - k using Nat.strong_induction_on generalizing k env with
  | h measure ih =>
      by_cases same : k = first d
      · apply BlockReturns.while_loop (value := .bool false)
        · exact Evaluates.compare (Evaluates.lookup getR) Evaluates.constant
            (by
              have hit : (10 : Int) ^ first d % d = 1 := by exact_mod_cast first_spec d large coprime
              simp [scalarCompare, evalCmpOp, same, hit])
        · exact BlockReturns.return_value (Evaluates.lookup (by simpa [same] using getK))
      · have before : k < first d := by omega
        have absent := before_first d k large coprime positive before
        apply BlockReturns.while_loop (value := .bool true)
        · exact Evaluates.compare (Evaluates.lookup getR) Evaluates.constant
            (by
              have miss : (10 : Int) ^ k % d ≠ 1 := by exact_mod_cast absent
              simp [scalarCompare, evalCmpOp, miss])
        · apply BlockReturns.assign (value := .int (10 ^ (k + 1) % d))
          · apply Evaluates.binary (av := .int (10 ^ k % d * 10)) (bv := .int d)
            · exact Evaluates.binary (Evaluates.lookup getR) Evaluates.constant (by simp [scalarBinary])
            · exact Evaluates.lookup getD
            · have next : ((10 : Int) ^ k % d * 10) % d = 10 ^ (k + 1) % d := by exact_mod_cast next_remainder d k
              simpa [scalarBinary, show d ≠ 0 by omega, show 0 < d by omega] using next
          · apply BlockReturns.augAssign (value := .int (k + 1))
            · exact Evaluates.binary (Evaluates.lookup (by simpa [ScalarEnv.set] using getK))
                Evaluates.constant (by simp [scalarBinary])
            · apply ih (first d - (k + 1)) (by omega) (k + 1) (by omega) (by omega)
              all_goals simp [ScalarEnv.set, getD]

/-- The actual factor-removal and remainder loops return the statement's bounded cycle length. -/
theorem recurring_cycle_length.result (d : Nat) (positive : 0 < d) :
    ReturnsValue program "recurring_cycle_length" [.int d] (.int (cycleLen d)) := by
  apply BlockReturns.function rfl rfl
  apply strip_loop 2 d (by decide) positive _ rfl
  intro second getSecond
  apply strip_loop 5 (stripFactor 2 d) (by decide) (stripFactor_pos 2 d (by decide) positive) second getSecond
  intro env getD
  change env "d" = some (.int (reducedDenom d)) at getD
  have denPos := reducedDenom_pos d positive
  by_cases one : reducedDenom d = 1
  · apply BlockReturns.branch (value := .bool true)
    · exact Evaluates.compare (Evaluates.lookup getD) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp, one])
    · simpa [cycleLen, one] using (BlockReturns.return_value (module := program) (env := env)
        (rest := (executableBody ((findFunctionIn program.body "recurring_cycle_length").getD ([], [])).2).drop 3)
        (Evaluates.constant (value := .int 0)))
  · have large : 1 < reducedDenom d := by omega
    have coprime := reducedDenom_coprime d positive
    have correct : cycleLen d = first (reducedDenom d) := by
      simp [cycleLen, one, cycleLenAux_eq_first _ large coprime]
    rw [correct]
    apply BlockReturns.branch (value := .bool false)
    · exact Evaluates.compare (Evaluates.lookup getD) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp, one])
    · apply BlockReturns.assign Evaluates.constant
      apply BlockReturns.assign (value := .int (10 % reducedDenom d))
      · exact Evaluates.binary Evaluates.constant (Evaluates.lookup (by simpa [ScalarEnv.set] using getD))
          (by simp [scalarBinary, show reducedDenom d ≠ 0 by omega, denPos])
      · exact period_loop (reducedDenom d) 1 large coprime (by decide)
          (first_pos _ large coprime) _ (by simpa [ScalarEnv.set] using getD)
          (by simp [ScalarEnv.set]) (by simp [ScalarEnv.set])

theorem recurring_cycle_length.«Returns» (d : Int)
    (requires : Calls.Condition program (fun key => if key = "d" then some (.int d) else none)
      Solvers.P26Ast.requires_recurring_cycle_length_0) :
    (Calls.function program "recurring_cycle_length" [.int d]).Returns := by
  have positive : 1 ≤ d := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare requires rfl
  lift d to Nat using (by omega : 0 ≤ d)
  exact (recurring_cycle_length.result d (by omega)).returns


private def searchBody : List Stmt :=
  [.assign [.name "L"] (.call (.name "recurring_cycle_length") [.name "d"]),
   .ifThen (.compare (.name "L") [(.gt, .name "best_len")])
     [.assign [.name "best_len"] (.name "L"), .assign [.name "best_d"] (.name "d")] []]

/-- Maximum selection keeps the earlier denominator when cycle lengths tie. -/
private theorem search_loop (count start limit winner : Nat) (stop : start + count = limit)
    (positive : 0 < start) (env : ScalarEnv)
    (getBest : env "best_d" = some (.int winner))
    (getLength : env "best_len" = some (.int (cycleLen winner))) :
    BlockReturns program env
      [.rangeStepLoop "d" start limit 1 searchBody [], .return (.name "best_d")]
      (.int (↑((List.range' start count).foldl (ProjectEulerStatements.ArgMaximum.select cycleLen) winner) : Int)) := by
  induction count generalizing start winner env with
  | zero =>
      apply BlockReturns.range_done (by simp [rangeContinues]; omega)
      exact BlockReturns.return_value (Evaluates.lookup (by simpa using getBest))
  | succ count ih =>
      apply BlockReturns.range_step (by simp [rangeContinues]; omega)
      apply BlockReturns.assign
        (Evaluates.call (.cons (Evaluates.lookup (by simp [ScalarEnv.set])) .nil)
          (recurring_cycle_length.result start positive))
      by_cases improve : cycleLen winner < cycleLen start
      · apply BlockReturns.branch (value := .bool true)
        · exact Evaluates.compare (av := .int (cycleLen start)) (Evaluates.lookup (by simp [ScalarEnv.set]))
            (Evaluates.lookup (by simpa [ScalarEnv.set] using getLength))
            (by simp [scalarCompare, evalCmpOp, improve])
        · apply BlockReturns.assign (value := .int (cycleLen start)) (Evaluates.lookup (by simp [ScalarEnv.set]))
          apply BlockReturns.assign (value := .int start) (Evaluates.lookup (by simp [ScalarEnv.set]))
          simpa only [List.range'_succ, List.foldl_cons, ProjectEulerStatements.ArgMaximum.select, if_pos improve,
            Nat.cast_add, Nat.cast_one] using
            ih (start + 1) start (by omega) (by omega) _ (by simp [ScalarEnv.set]) (by simp [ScalarEnv.set])
      · apply BlockReturns.branch (value := .bool false)
        · exact Evaluates.compare (av := .int (cycleLen start)) (Evaluates.lookup (by simp [ScalarEnv.set]))
            (Evaluates.lookup (by simpa [ScalarEnv.set] using getLength))
            (by simp [scalarCompare, evalCmpOp, improve])
        · simpa only [List.range'_succ, List.foldl_cons, ProjectEulerStatements.ArgMaximum.select, if_neg improve,
            Nat.cast_add, Nat.cast_one] using
            ih (start + 1) winner (by omega) (by omega) _
              (by simpa [ScalarEnv.set] using getBest) (by simpa [ScalarEnv.set] using getLength)

private theorem zero_score : cycleLen 0 = 0 := by
  have two : stripFactor 2 0 = 0 := by rw [stripFactor]; norm_num
  have five : stripFactor 5 0 = 0 := by rw [stripFactor]; norm_num
  simp [cycleLen, reducedDenom, two, five, cycleLenAux]
private theorem one_score : cycleLen 1 = 0 := by
  simp [cycleLen, reducedDenom, stripFactor_done 2 1 (by decide) (by decide),
    stripFactor_done 5 1 (by decide) (by decide)]
private theorem two_score : cycleLen 2 = 0 := by
  have two : stripFactor 2 2 = 1 := by
    rw [stripFactor_step 2 2 (by decide) (by decide) (by decide)]
    exact stripFactor_done 2 1 (by decide) (by decide)
  simp [cycleLen, reducedDenom, two, stripFactor_done 5 1 (by decide) (by decide)]
private theorem three_score : cycleLen 3 = 1 := by
  simp [cycleLen, reducedDenom, stripFactor_done 2 3 (by decide) (by decide),
    stripFactor_done 5 3 (by decide) (by decide), cycleLenAux, cycleLenAux.go]

private theorem winner_eq (limit : Nat) (large : 4 ≤ limit) :
    ((List.range' 3 (limit - 3)).foldl (ProjectEulerStatements.ArgMaximum.select cycleLen) 2) = naive limit := by
  obtain ⟨extra, rfl⟩ := Nat.exists_eq_add_of_le large
  simp [naive, List.range_eq_range', List.range'_succ, ProjectEulerStatements.ArgMaximum.argmax_cons,
    ProjectEulerStatements.ArgMaximum.select, zero_score, one_score, two_score, three_score,
    show 4 + extra = extra + 4 by omega]

/-- The source's range starts at two; index three supplies a positive score before later maxima. -/
theorem solve.result (limit : Nat) (large : 4 ≤ limit) :
    ReturnsValue program "solve" [.int limit] (.int (naive limit)) := by
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply BlockReturns.assign (value := .int (-1))
    (Evaluates.unary Evaluates.constant rfl)
  apply BlockReturns.for_range (.cons Evaluates.constant (.cons (Evaluates.lookup rfl) .nil)) rfl
  apply BlockReturns.range_step (by simp [rangeContinues]; omega)
  apply BlockReturns.assign (Evaluates.call (.cons (Evaluates.lookup rfl) .nil)
    (recurring_cycle_length.result 2 (by decide)))
  apply BlockReturns.branch (value := .bool true)
  · exact Evaluates.compare (Evaluates.lookup rfl) (Evaluates.lookup rfl)
      (by simp [scalarCompare, evalCmpOp, two_score])
  · apply BlockReturns.assign (Evaluates.lookup rfl)
    apply BlockReturns.assign (Evaluates.lookup rfl)
    rw [← winner_eq limit large]
    exact search_loop (limit - 3) 3 limit 2 (by omega) (by decide) _ rfl rfl

theorem solve.«Returns» (limit : Int)
    (requires : Calls.Condition program (fun key => if key = "limit" then some (.int limit) else none)
      Solvers.P26Ast.requires_solve_0) :
    (Calls.function program "solve" [.int limit]).Returns := by
  have large : 4 ≤ limit := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare requires rfl
  lift limit to Nat using (by omega : 0 ≤ limit)
  exact (solve.result limit (by omega)).returns


theorem solve.«Equiv(naive)» (limit : Int)
    (requires : Calls.Condition program (fun key => if key = "limit" then some (.int limit) else none)
      Solvers.P26Ast.requires_solve_0) :
    (Calls.function program "solve" [.int limit]).Equiv
      (Calls.function P26Ast.module "naive" [.int limit]) := by
  have large : 4 ≤ limit := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare requires rfl
  lift limit to Nat using (by omega : 0 ≤ limit)
  have optimized := (solve.result limit (by omega)).finishes
  have reference := (data.«project-euler-statements».data.py.project_euler_statements.p26.naive.result limit).finishes
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem solve.«Supported» :
    (∀ limit : Int, Calls.Condition program
      (fun key => if key = "limit" then some (.int limit) else none) Solvers.P26Ast.requires_solve_0 →
      (Calls.function program "solve" [.int limit]).Supported) ∧
    (∀ limit : Int, Calls.Condition program
      (fun key => if key = "limit" then some (.int limit) else none) Solvers.P26Ast.requires_solve_0 →
      (Calls.function P26Ast.module "naive" [.int limit]).Supported) := by
  constructor
  · intro limit requires
    exact (solve.«Returns» limit requires).supported
  · intro limit _
    exact (data.«project-euler-statements».data.py.project_euler_statements.p26.naive.«Returns» limit).supported

end solvers.«26»
